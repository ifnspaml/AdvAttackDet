#!/usr/bin/env python3

# Python standard library
import json
import os

# Public libraries
import numpy as np
import torch
import torch.nn.functional as functional

# IfN libraries
from dataloader.eval.metrics import DepthRunningScore

# Local imports
import loaders, loaders.joint, loaders.fns

from losses.segmentation import RemappingScore
from state_manager import StateManager
from perspective_resample import PerspectiveResampler


class Harness(object):
    def __init__(self, opt):
        print('Starting initialization', flush=True)

        self._init_device(opt)
        self._init_resampler(opt)
        self._init_log_dir(opt)
        self._init_state(opt)
        self._init_validation_loaders(opt)
        self._init_validation(opt)
        self._save_opts(opt)

        print('Summary:')
        print(f'  - Model name: {opt.model_name}')
        print(f'  - Logging directory: {self.log_path}')
        print(f'  - Using device: {self._pretty_device_name()}')

    def _init_device(self, opt):
        if not torch.cuda.is_available():
            self.device = torch.device("cpu")
        else:
            self.device = torch.device(opt.sys_device_name)

    def _init_resampler(self, opt):
        if hasattr(opt, 'depth_min_sampling_res'):
            self.resample = PerspectiveResampler(opt.model_depth_max, opt.model_depth_min, opt.depth_min_sampling_res)
        else:
            self.resample = PerspectiveResampler(opt.model_depth_max, opt.model_depth_min)

    def _init_log_dir(self, opt):
        self.log_path = os.path.join("pretrained", opt.model_name)

        self.run_detection = False

        os.makedirs(self.log_path, exist_ok=True)

    def _init_state(self, opt):
        self.state = StateManager(
            opt.model_name, self.device,
            opt.model_split_pos, opt.model_num_layers, opt.train_depth_grad_scale, opt.train_segmentation_grad_scale,
            opt.train_domain_grad_scale, opt.train_weights_init, opt.model_depth_resolutions, opt.train_mode)
        if opt.model_load is not None:
            self.state.load(opt.model_load)

    def _init_validation_loaders(self, opt):
        print('Loading validation dataset metadata:', flush=True)

        if hasattr(opt, 'joint_validation_loaders'):
            self.joint_validation_loader = loaders.ChainedLoaderList(
                getattr(loaders.joint, loader_name)(
                    resize_height=opt.joint_validation_resize_height,
                    resize_width=opt.joint_validation_resize_width,
                    batch_size=opt.joint_validation_batch_size,
                    num_workers=opt.sys_num_workers
                )
                for loader_name in opt.joint_validation_loaders.split(',') if (loader_name != '')
            )

    def _init_validation(self, opt):
        self.fixed_depth_scaling = opt.depth_validation_fixed_scaling

    def _pretty_device_name(self):
        dev_type = self.device.type

        dev_idx = (
            f',{self.device.index}'
            if (self.device.index is not None)
            else ''
        )

        dev_cname = (
            f' ({torch.cuda.get_device_name(self.device)})'
            if (dev_type == 'cuda')
            else ''
        )

        return f'{dev_type}{dev_idx}{dev_cname}'

    def _log_gpu_memory(self):
        if self.device.type == 'cuda':
            max_mem = torch.cuda.max_memory_allocated(self.device)

            print('Maximum bytes of GPU memory used:')
            print(max_mem)

    def _save_opts(self, opt):
        opt_path = os.path.join(self.log_path, 'opt.json')

        with open(opt_path, 'w') as fd:
            json.dump(vars(opt), fd, indent=2)

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

    def _validate_batch_depth(self, model, batch, score, ratios, images):
        if len(batch) != 1:
            raise Exception('Can only run validation on batches containing only one dataset')

        im_scores = list()
        single_im_score = DepthRunningScore()

        batch_gpu = self._batch_to_device(batch)
        outputs = model(batch_gpu)

        colors_gt = batch[0]['color', 0, -1]
        depths_gt = batch[0]['depth', 0, 0][:, 0]

        disps_pred = outputs[0]["disp", 0]
        disps_scaled_pred = self.resample.scale_disp(disps_pred)
        disps_scaled_pred = disps_scaled_pred.cpu()[:, 0]

        if self.run_detection:
            depth_image = disps_scaled_pred.detach()

        # Process each image from the batch separately
        for i in range(depths_gt.shape[0]):
            # If you are here due to an exception, make sure that your loader uses
            # AddKeyValue('domain', domain_name), AddKeyValue('validation_mask', mask_fn)
            # and AddKeyValue('validation_clamp', clamp_fn) to add these keys to each input sample.
            # There is no sensible default, that works for all datasets,
            # so you have have to define one on a per-set basis.
            domain = batch[0]['domain'][i]
            mask_fn = loaders.fns.get(batch[0]['validation_mask'][i])
            clamp_fn = loaders.fns.get(batch[0]['validation_clamp'][i])

            color_gt = colors_gt[i].unsqueeze(0)
            depth_gt = depths_gt[i].unsqueeze(0)
            disp_scaled_pred = disps_scaled_pred[i].unsqueeze(0)

            img_height = depth_gt.shape[1]
            img_width = depth_gt.shape[2]
            disp_scaled_pred = functional.interpolate(
                disp_scaled_pred.unsqueeze(1),
                (img_height, img_width),
                align_corners=False,
                mode='bilinear'
            ).squeeze(1)
            depth_pred = 1 / disp_scaled_pred

            images.append((color_gt, depth_gt, depth_pred))

            # Datasets/splits define their own masking rules
            # delegate masking to functions defined in the loader
            mask = mask_fn(depth_gt)
            depth_pred = depth_pred[mask]
            depth_gt = depth_gt[mask]

            if self.fixed_depth_scaling != 0:
                ratio = self.fixed_depth_scaling

            else:
                median_gt = np.median(depth_gt.numpy())
                median_pred = np.median(depth_pred.numpy())

                ratio = (median_gt / median_pred).item()

            ratios.append(ratio)
            depth_pred *= ratio

            # Datasets/splits define their own prediction clamping rules
            # delegate clamping to functions defined in the loader
            depth_pred = clamp_fn(depth_pred)

            score.update(
                depth_gt.numpy(),
                depth_pred.numpy()
            )
            single_im_score.update(
                depth_gt.numpy(),
                depth_pred.numpy()
            )
            im_scores.append(single_im_score.get_scores())
            single_im_score.reset()

        if self.run_detection:
            self.detection_model.enter_depth(depth_image.detach().cpu().unsqueeze(1), scale_factor=ratio)

        return im_scores

    def _validate_batch_segmentation(self, model, batch, score, images):
        if len(batch) != 1:
            raise Exception('Can only run validation on batches containing only one dataset')

        im_scores = list()
        single_im_score = RemappingScore()

        batch_gpu = self._batch_to_device(batch)
        outputs = model(batch_gpu)  # forward the data through the network

        # needed for video evaluation
        if ('segmentation', 0, 0) not in batch[0].keys():
            return im_scores

        colors_gt = batch[0]['color', 0, -1]
        segs_gt = batch[0]['segmentation', 0, 0].squeeze(1).long()
        segs_pred = outputs[0]['segmentation_logits', 0]

        if self.run_detection:
            self.detection_model.enter_segmentation(segs_pred.detach().cpu())

        segs_pred = functional.interpolate(segs_pred, segs_gt[0, :, :].shape, mode='nearest')

        for i in range(segs_pred.shape[0]):
            color_gt = colors_gt[i].unsqueeze(0)
            seg_gt = segs_gt[i].unsqueeze(0)
            seg_pred = segs_pred[i].unsqueeze(0)

            images.append((color_gt, seg_gt, seg_pred.argmax(1).cpu()))

            score.update(seg_gt, seg_pred)
            single_im_score.update(seg_gt, seg_pred)
            im_scores.append(single_im_score['none'].get_scores())
            single_im_score.reset()

        return im_scores

    def _validate_batch_joint(self, model, batch, depth_score, depth_ratios, depth_images,
                              seg_score, seg_images, seg_perturbations,
                              seg_im_scores, depth_im_scores):
        # apply a perturbation onto the input image
        batch, seg_perturbation = self.attack_model.perturb(batch, model)
        seg_perturbations.append(seg_perturbation)

        # insert the color image saving after the perturbation has been applied
        # this requires that the option output_filenames in the dataset is set to True

        if self.run_detection:
            color_image = batch[0]['color_aug', 0, 0]
            self.detection_model.enter_color(color_image.detach().cpu(),
                                             mean=(0.485, 0.456, 0.406),
                                             std=(0.229, 0.224, 0.225)
                                             )

        # pass the evaluation to the single evaluation routines
        with torch.no_grad():
            seg_im_scores.extend(
                self._validate_batch_segmentation(model, batch, seg_score, seg_images)
            )
            depth_im_scores.extend(
                self._validate_batch_depth(model, batch, depth_score, depth_ratios, depth_images)
            )

        if self.run_detection:
            self.detection_model.calculate_scores()

    def _run_joint_validation(self, images_to_keep=0, class_remaps=('none',)):
        depth_scores = dict()
        depth_ratios = dict()
        depth_images = dict()
        seg_scores = dict()
        seg_images = dict()
        seg_perturbations = dict()

        seg_im_scores = dict()
        depth_im_scores = dict()

        with self.state.model_manager.get_eval() as model:
            for batch in self.joint_validation_loader:

                domain = batch[0]['domain'][0]

                if domain not in seg_scores:
                    seg_scores[domain] = RemappingScore(class_remaps)
                    seg_images[domain] = list()
                    seg_perturbations[domain] = list()
                    depth_scores[domain] = DepthRunningScore()
                    depth_ratios[domain] = list()
                    depth_images[domain] = list()
                    seg_im_scores[domain] = list()
                    depth_im_scores[domain] = list()

                self._validate_batch_joint(model, batch, depth_scores[domain], depth_ratios[domain],
                                           depth_images[domain], seg_scores[domain], seg_images[domain],
                                           seg_perturbations[domain], seg_im_scores[domain], depth_im_scores[domain])

                depth_images[domain] = depth_images[domain][:images_to_keep]
                seg_images[domain] = seg_images[domain][:images_to_keep]

        return depth_scores, depth_ratios, depth_images, seg_scores, seg_images, seg_perturbations
