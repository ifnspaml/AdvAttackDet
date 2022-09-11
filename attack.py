import torch
import torch.nn.functional as functional
import numpy as np
import sys
import dataloader.pt_data_loader.mytransforms as tf

from losses import DepthLosses, SegLosses
from losses.segmentation import RemappingScore
from perspective_resample import PerspectiveResampler


class Attack:
    KEY_FRAME_CUR = ('color_aug', 0, 0)

    def __init__(self, mode, eps, device, eval_disparity_smoothness, eval_disparity_smoothness_seg,
                 eval_segmentation_smoothness, scale_factors=(1.0, 0.0)):
        self.eps = eps / 255.0  # scaling necessary as inputs are normalized to [0, 1]
        self.mode = mode
        self.normalizer = tf.NormalizeZeroMean()
        self.device = device

        self.scale_factors = scale_factors  # index 0: segmentation, index 1: depth
        self.resample = PerspectiveResampler()

        # the smoothness terms should point in the other direction as the attack is aiming at
        assert eval_disparity_smoothness >= 0 or eval_disparity_smoothness_seg >= 0 or eval_segmentation_smoothness >= 0, \
            'The smoothness values are not allowed to be smaller than 0 as they are multiplied by -1'
        self.depth_loss = DepthLosses(device,
                                      disparity_smoothness=eval_disparity_smoothness * -1.0,
                                      disparity_smoothness_seg=eval_disparity_smoothness_seg * -1.0,
                                      depth_regularization_type="ssim")
        self.seg_loss = SegLosses(device,
                                  segmentation_smoothness=eval_segmentation_smoothness * -1.0,
                                  segmentation_regularization_type="ssim",
                                  use_weights=False)

        self.single_im_score = RemappingScore()

    def perturb(self, batch, model):
        if self.mode == 'gaussian':
            return self._add_gaussian_noise(batch)
        elif self.mode == 'saltandpepper':
            return self._add_salt_and_pepper(batch)
        elif self.mode == 'fgsm':
            return self._untargeted(batch, model)
        elif self.mode == 'bim':
            return self._untargeted_bim(batch, model)
        elif self.mode == 'pgd':
            return self._untargeted_pgd(batch, model)
        elif self.mode == 'o-pgd':
            return self._untargeted_opgd(batch, model)
        else:
            raise NotImplementedError("Attack not implemented")

    def _batch_to_device(self, batch_cpu):
        batch_gpu = list()

        for dataset_cpu in batch_cpu:
            dataset_gpu = dict()

            for k, ipt in dataset_cpu.items():
                if isinstance(ipt, torch.Tensor):
                    dataset_gpu[k] = ipt.to(self.device)

                else:
                    dataset_gpu[k] = ipt

            batch_gpu.append(dataset_gpu)

        return tuple(batch_gpu)

    def _add_gaussian_noise(self, batch):

        # Calculate the gaussian noise onto the image
        noise = np.random.normal(0, self.eps, batch[0][self.KEY_FRAME_CUR].shape)
        noise = torch.from_numpy(noise.astype(np.float32))
        seg_perturbation = np.sqrt((noise * noise).mean())

        # Calculate the gaussian noise onto the image adapt the input image inside the batch
        batch[0][self.KEY_FRAME_CUR] += noise
        batch[0][self.KEY_FRAME_CUR] = batch[0][self.KEY_FRAME_CUR].clamp(0.0, 1.0)
        batch[0][self.KEY_FRAME_CUR][0] = self.normalizer(
            {self.KEY_FRAME_CUR: batch[0][self.KEY_FRAME_CUR][0]})[self.KEY_FRAME_CUR]

        # return the batch with the perturbed input and the perturbation strength
        return batch, seg_perturbation

    def _add_salt_and_pepper(self, batch):
        s_vs_p = 0.5
        amount = self.eps * self.eps * 4
        image = np.copy(batch[0][self.KEY_FRAME_CUR])
        num_salt = int(np.ceil(amount * np.prod(image.shape) * s_vs_p))
        num_pepper = int(np.ceil(amount * np.prod(image.shape) * (1. - s_vs_p)))

        # determine coords to be jittered
        grid = np.meshgrid(*[np.linspace(0, len, len, endpoint=False) for len in image.shape])
        grid = [g.flatten() for g in grid]
        idxs = np.random.randint(0, len(grid[0]), num_salt + num_pepper)
        idxs_salt = idxs[:num_salt]
        idxs_pepper = idxs[num_salt:]
        coords_salt = [np.take(g, idxs_salt) for g in grid]
        coords_pepper = [np.take(g, idxs_pepper) for g in grid]

        # set the determined coordinates to 0 or 1
        batch[0][self.KEY_FRAME_CUR][tuple(coords_salt)] = 1.0
        batch[0][self.KEY_FRAME_CUR][tuple(coords_pepper)] = 0.0

        # clip to the valid range
        batch[0][self.KEY_FRAME_CUR] = batch[0][self.KEY_FRAME_CUR].clamp(0.0, 1.0)

        # calculate the noise from the updated image and the original image
        noise = np.array(batch[0][self.KEY_FRAME_CUR]) - image
        seg_perturbation = np.sqrt((noise * noise).mean())

        # normalize to zero mean
        batch[0][self.KEY_FRAME_CUR][0] = self.normalizer(
            {self.KEY_FRAME_CUR: batch[0][self.KEY_FRAME_CUR][0]})[self.KEY_FRAME_CUR]

        # return the batch with the perturbed input and the perturbation strength
        return batch, seg_perturbation

    def _untargeted(self, batch, model):
        # prepare the input such that it requires a gradient
        batch_gpu = self._batch_to_device(batch)
        image = batch_gpu[0][self.KEY_FRAME_CUR].clone()

        image.requires_grad = True
        batch_gpu[0][self.KEY_FRAME_CUR][0] = self.normalizer(
            {self.KEY_FRAME_CUR: image[0].clone()})[self.KEY_FRAME_CUR]

        # prepare the labels to have the same size as the image
        labels = batch_gpu[0][('segmentation', 0, 0)]
        labels = functional.interpolate(labels, image[0, 0, :, :].shape, mode='nearest')

        # feed the input forward through the model
        # model.set_gradient_scales(1, 1, 1)
        model.to(self.device).zero_grad()
        outputs = model(batch_gpu)

        # calculate the loss
        predictions_depth = self.resample.warp_images(batch_gpu[0], outputs[0], None)
        outputs[0].update(predictions_depth)
        loss_depth = self.depth_loss.compute_losses(batch_gpu[0], outputs[0], None,
                                                    {('segmentation', 0, 0): outputs[0][('segmentation_logits', 0)]})
        loss_seg = self.seg_loss.seg_losses({('segmentation', 0, 0): labels,
                                             ('color', 0, 0): batch_gpu[0][self.KEY_FRAME_CUR]}, outputs[0])
        loss = self.scale_factors[0] * loss_seg['loss_seg'] + self.scale_factors[1] * loss_depth['loss_depth']

        loss.backward()

        # calculate the noise from the gradients
        im_grad = image.grad.data
        noise = self.eps * torch.sign(im_grad)

        # Calculate the fgsm noise onto the image and adapt the input image inside the batch
        batch[0][self.KEY_FRAME_CUR] = (image + noise).cpu()
        batch[0][self.KEY_FRAME_CUR] = batch[0][self.KEY_FRAME_CUR].clamp(0.0, 1.0)
        batch[0][self.KEY_FRAME_CUR][0] = self.normalizer(
            {self.KEY_FRAME_CUR: batch[0][self.KEY_FRAME_CUR][0]})[self.KEY_FRAME_CUR]

        # return the batch with the perturbed input and the perturbation strength
        seg_perturbation = np.sqrt((noise * noise).detach().cpu().mean())
        return batch, seg_perturbation

    def _untargeted_bim(self, batch, model):
        iters = 40
        alpha = 0.01

        # prepare the input such that it requires a gradient
        batch_gpu = self._batch_to_device(batch)
        image = batch_gpu[0][self.KEY_FRAME_CUR].clone()
        ori_image = image.data

        # prepare the labels to have the same size as the image
        labels = batch_gpu[0][('segmentation', 0, 0)]
        labels = functional.interpolate(labels, image[0, 0, :, :].shape, mode='nearest')

        # prepare the model
        # model.set_gradient_scales(1, 1, 1)
        model.to(self.device).zero_grad()

        # calculate the pgd attack
        for i in range(iters):
            # prepare the model input
            image.requires_grad = True
            batch_gpu = self._batch_to_device(batch)
            batch_gpu[0][self.KEY_FRAME_CUR][0] = self.normalizer(
                {self.KEY_FRAME_CUR: image[0].clone()})[self.KEY_FRAME_CUR]

            # calculate the loss and backprop the gradient
            model.zero_grad()
            outputs = model(batch_gpu)

            # calculate the loss
            if self.scale_factors[1] != 0:
                predictions_depth = self.resample.warp_images(batch_gpu[0], outputs[0], None)
                outputs[0].update(predictions_depth)
                loss_depth = self.depth_loss.compute_losses(batch_gpu[0], outputs[0], None,
                                                            {('segmentation', 0, 0): outputs[0][
                                                                ('segmentation_logits', 0)]})
                loss_seg = self.seg_loss.seg_losses({('segmentation', 0, 0): labels,
                                                     ('color', 0, 0): batch_gpu[0][self.KEY_FRAME_CUR]}, outputs[0])
                loss = self.scale_factors[0] * loss_seg['loss_seg'] + self.scale_factors[1] * loss_depth['loss_depth']
            else:
                loss = self.seg_loss.seg_losses({('segmentation', 0, 0): labels,
                                                 ('color', 0, 0): batch_gpu[0][self.KEY_FRAME_CUR]}, outputs[0])['loss_seg']

            loss.backward(retain_graph=True)

            adv_image = image + alpha * torch.sign(image.grad.data)
            eta = torch.clamp(adv_image - ori_image, min=-self.eps, max=self.eps)
            image = torch.clamp(ori_image + eta, 0, 1).detach_()

        noise = image.detach_().cpu() - ori_image.detach_().cpu()
        batch[0][self.KEY_FRAME_CUR] = image.detach_().cpu()
        batch[0][self.KEY_FRAME_CUR][0] = self.normalizer(
            {self.KEY_FRAME_CUR: batch[0][self.KEY_FRAME_CUR][0]})[self.KEY_FRAME_CUR]

        seg_perturbation = np.sqrt((noise * noise).detach().cpu().mean())
        return batch, seg_perturbation

    def _untargeted_pgd(self, batch, model):
        iters = 40
        alpha = 0.01

        # prepare the input such that it requires a gradient
        batch_gpu = self._batch_to_device(batch)
        image = batch_gpu[0][self.KEY_FRAME_CUR].clone()
        ori_image = image.data

        # add noise from the start to implement PGD instead of BIM
        initial_noise = np.random.uniform(-self.eps, self.eps, image.shape).astype(np.float32)
        image = image + torch.from_numpy(initial_noise).to(self.device)

        # prepare the labels to have the same size as the image
        labels = batch_gpu[0][('segmentation', 0, 0)]
        labels = functional.interpolate(labels, image[0, 0, :, :].shape, mode='nearest')

        # prepare the model
        model.to(self.device).zero_grad()

        # calculate the pgd attack
        for i in range(iters):
            # prepare the model input
            image.requires_grad = True
            batch_gpu = self._batch_to_device(batch)
            batch_gpu[0][self.KEY_FRAME_CUR][0] = self.normalizer(
                {self.KEY_FRAME_CUR: image[0].clone()})[self.KEY_FRAME_CUR]

            # calculate the loss and backprop the gradient
            model.zero_grad()
            outputs = model(batch_gpu)

            # calculate the loss
            if self.scale_factors[1] != 0:
                predictions_depth = self.resample.warp_images(batch_gpu[0], outputs[0], None)
                outputs[0].update(predictions_depth)
                loss_depth = self.depth_loss.compute_losses(batch_gpu[0], outputs[0], None,
                                                            {('segmentation', 0, 0): outputs[0][
                                                                ('segmentation_logits', 0)]})
                loss_seg = self.seg_loss.seg_losses({('segmentation', 0, 0): labels,
                                                     ('color', 0, 0): batch_gpu[0][self.KEY_FRAME_CUR]}, outputs[0])
                loss = self.scale_factors[0] * loss_seg['loss_seg'] + self.scale_factors[1] * loss_depth['loss_depth']
            else:
                loss = self.seg_loss.seg_losses({('segmentation', 0, 0): labels,
                                                     ('color', 0, 0): batch_gpu[0][self.KEY_FRAME_CUR]}, outputs[0])['loss_seg']

            loss.backward(retain_graph=True)

            adv_image = image + alpha * torch.sign(image.grad.data)
            eta = torch.clamp(adv_image - ori_image, min=-self.eps, max=self.eps)
            image = torch.clamp(ori_image + eta, 0, 1).detach_()

        noise = image.detach_().cpu() - ori_image.detach_().cpu()
        batch[0][self.KEY_FRAME_CUR] = image.detach_().cpu()
        batch[0][self.KEY_FRAME_CUR][0] = self.normalizer(
            {self.KEY_FRAME_CUR: batch[0][self.KEY_FRAME_CUR][0]})[self.KEY_FRAME_CUR]

        seg_perturbation = np.sqrt((noise * noise).detach().cpu().mean())
        return batch, seg_perturbation

    def _untargeted_opgd(self, batch, model):
        iters = 100
        alpha = 0.01

        # prepare the input such that it requires a gradient
        batch_gpu = self._batch_to_device(batch)
        image = batch_gpu[0][self.KEY_FRAME_CUR].clone()
        ori_image = image.data

        # prepare the labels to have the same size as the image
        labels = batch_gpu[0][('segmentation', 0, 0)]
        labels = functional.interpolate(labels, image[0, 0, :, :].shape, mode='nearest')

        # prepare the model
        model.to(self.device).zero_grad()

        # calculate the pgd attack
        for i in range(iters):
            # prepare the model input
            image.requires_grad = True
            batch_gpu = self._batch_to_device(batch)
            batch_gpu[0][self.KEY_FRAME_CUR][0] = self.normalizer(
                {self.KEY_FRAME_CUR: image[0].clone()})[self.KEY_FRAME_CUR]

            # calculate the loss and backprop the gradient
            model.zero_grad()
            outputs = model(batch_gpu)

            # calculate the mIoU for each image to be able to decide how to project gradients
            segs_gt = batch[0]['segmentation', 0, 0].squeeze(1).long()
            segs_pred = outputs[0]['segmentation_logits', 0].clone()
            segs_pred = functional.interpolate(segs_pred, segs_gt[0, :, :].shape, mode='nearest')
            score = list()
            for j in range(segs_pred.shape[0]):
                seg_gt = segs_gt[j].unsqueeze(0)
                seg_pred = segs_pred[j].unsqueeze(0)
                self.single_im_score.update(seg_gt, seg_pred)
                score.append(self.single_im_score['none'].get_scores()['meaniou'])
                self.single_im_score.reset()

            # Indicator, if the mIoU is below 5, needed for selection process in O-PGD
            has_attack_succeeded = torch.from_numpy(np.array(score)) < 0.05

            # calculate the loss
            predictions_depth = self.resample.warp_images(batch_gpu[0], outputs[0], None)
            outputs[0].update(predictions_depth)
            self.depth_loss.disparity_smoothness = 1.0
            self.depth_loss.disparity_smoothness_seg = 1.0
            self.seg_loss.segmentation_smoothness = 1.0
            loss_depth = self.depth_loss.compute_losses(batch_gpu[0], outputs[0], None,
                                                        {('segmentation', 0, 0): outputs[0][
                                                            ('segmentation_logits', 0)]})
            loss_seg = self.seg_loss.seg_losses({('segmentation', 0, 0): labels,
                                                 ('color', 0, 0): batch_gpu[0][self.KEY_FRAME_CUR]}, outputs[0])
            loss_perf = 0
            loss_det = 0
            for key in loss_seg.keys():
                if "cross" in key:
                    loss_perf += self.scale_factors[0] * loss_seg[key]
                if "smth" in key:
                    loss_det += loss_seg[key]
            for key in loss_depth.keys():
                if "photometric" in key:
                    loss_perf += self.scale_factors[1] * loss_depth[key]
                if "smth" in key:
                    loss_det += loss_depth[key]

            # calculate the gradients
            loss_perf.backward(retain_graph=True)
            grad_perf = image.grad.data

            image.grad = None
            model.zero_grad()
            loss_det.backward(retain_graph=True)
            grad_det = image.grad.data
            batch_size = image.shape[0]

            # using Orthogonal Projected Gradient Descent
            # projection of gradient of detector on gradient of classifier
            # then grad_d' = grad_d - (project grad_d onto grad_c)
            grad_det_proj = grad_det - torch.bmm(
                (torch.bmm(grad_det.view(batch_size, 1, -1), grad_det.view(batch_size, -1, 1))) / (
                            1e-20 + torch.bmm(grad_perf.view(batch_size, 1, -1),
                                              grad_perf.view(batch_size, -1, 1))).view(-1, 1, 1),
                grad_perf.view(batch_size, 1, -1)).view(grad_det.shape)

            # using Orthogonal Projected Gradient Descent
            # projection of gradient of detector on gradient of classifier
            # then grad_c' = grad_c - (project grad_c onto grad_d)
            grad_perf_proj = grad_perf - torch.bmm(
                (torch.bmm(grad_perf.view(batch_size, 1, -1), grad_det.view(batch_size, -1, 1))) / (
                            1e-20 + torch.bmm(grad_det.view(batch_size, 1, -1),
                                              grad_det.view(batch_size, -1, 1))).view(-1, 1, 1),
                grad_det.view(batch_size, 1, -1)).view(grad_perf.shape)

            grad = grad_perf_proj * (1 - int(has_attack_succeeded)) + grad_det_proj * int(has_attack_succeeded)

            adv_image = image + alpha * torch.sign(grad)
            eta = torch.clamp(adv_image - ori_image, min=-self.eps, max=self.eps)
            image = torch.clamp(ori_image + eta, 0, 1).detach_()

        noise = image.detach_().cpu() - ori_image.detach_().cpu()
        batch[0][self.KEY_FRAME_CUR] = image.detach_().cpu()
        batch[0][self.KEY_FRAME_CUR][0] = self.normalizer(
            {self.KEY_FRAME_CUR: batch[0][self.KEY_FRAME_CUR][0]})[self.KEY_FRAME_CUR]

        seg_perturbation = np.sqrt((noise * noise).detach().cpu().mean())
        return batch, seg_perturbation
