import torch
import torch.nn as nn
import torch.nn.functional as functional

import losses as trn_losses


class DepthLosses(object):
    def __init__(self, device, disable_automasking=False, avg_reprojection=False, disparity_smoothness=0,
                 disparity_smoothness_seg=0, depth_regularization_type="smooth", supervision_mode="self"):
        self.automasking = not disable_automasking
        self.avg_reprojection = avg_reprojection
        self.disparity_smoothness = disparity_smoothness
        self.disparity_smoothness_seg = disparity_smoothness_seg
        self.supervision_mode = supervision_mode
        self.scaling_direction = "up"
        self.masked_supervision = True

        # noinspection PyUnresolvedReferences
        self.ssim = trn_losses.SSIM().to(device)
        self.softmax = nn.Softmax(dim=1)

        if depth_regularization_type == "smooth":
            self.smoothness = trn_losses.SmoothnessLoss()
            self.smoothness_seg = trn_losses.SmoothnessLoss()
        elif depth_regularization_type == "ssim":
            self.smoothness = trn_losses.SSIMGradLoss().to(device)
            self.smoothness_seg = trn_losses.SSIMGradLoss().to(device)
        elif depth_regularization_type == "abs":
            self.smoothness = trn_losses.MAEGradLoss()
            self.smoothness_seg = trn_losses.MAEGradLoss()

    def _supervision_losses(self, inputs, outputs):
        """Calculates an L1-Loss between target and prediction
            scaling direction:
                up: (up)scales the prediction to the size of the target before applying the loss
                    (uses bilinear sampling to not break gradient)
                down: (down)scales the target to the size of the prediction before applying the loss
                    (uses nearest neighbor to preserve sparse ground truth values)

            masked: indicates whether target value of 0 should be ignored (masked)
        """
        losses = dict()
        prediction = outputs[('depth', 0, 0)]
        target = inputs[('depth', 0, 0)]

        if self.scaling_direction == "up":
            prediction = functional.interpolate(prediction, target[0, 0].shape, mode="bilinear", align_corners=False)
        elif self.scaling_direction == "down":
            target = functional.interpolate(target, prediction[0, 0].shape, mode="nearest")

        if self.masked_supervision:
            mask = target > 0
            mask = mask.float()
            supervision_loss = torch.abs(target * mask - prediction * mask)
        else:
            supervision_loss = torch.abs(target - prediction)

        supervision_loss = supervision_loss.mean()
        losses['loss_depth_supervised'] = supervision_loss
        return losses

    def _combined_reprojection_loss(self, pred, target):
        """Computes reprojection losses between a batch of predicted and target images
        """

        # Calculate the per-color difference and the mean over all colors
        l1 = (pred - target).abs().mean(1, True)

        ssim = self.ssim(pred, target).mean(1, True)

        reprojection_loss = 0.85 * ssim + 0.15 * l1

        return reprojection_loss

    def _reprojection_losses(self, inputs, outputs, outputs_masked, outputs_seg):
        """Compute the reprojection and smoothness losses for a minibatch
        """

        frame_ids = tuple(frozenset(k[1] for k in outputs if k[0] == 'color'))

        resolutions = tuple(frozenset(k[2] for k in outputs if k[0] == 'color'))

        losses = dict()

        color = inputs["color", 0, 0]
        target = inputs["color", 0, 0]
        if self.disparity_smoothness_seg != 0:
            segmentation = self.softmax(outputs_seg['segmentation', 0, 0])

        # Compute reprojection losses for the unwarped input images
        identity_reprojection_loss = tuple(
            self._combined_reprojection_loss(inputs["color", frame_id, 0], target)
            for frame_id in frame_ids
        )
        identity_reprojection_loss = torch.cat(identity_reprojection_loss, 1)

        if self.avg_reprojection:
            identity_reprojection_loss = identity_reprojection_loss.mean(1, keepdim=True)

        for resolution in resolutions:
            # Compute reprojection losses (prev frame to cur and next frame to cur)
            reprojection_loss = tuple(
                self._combined_reprojection_loss(outputs["color", frame_id, resolution], target)
                for frame_id in frame_ids
            )
            reprojection_loss = torch.cat(reprojection_loss, 1)

            # If avg_reprojection is disabled and automasking is enabled
            # there will be four "loss  images" stacked in the end and
            # the per-pixel minimum will be selected for optimization.
            # Cases where this is relevant are, for example, image borders,
            # where information is missing, or areas occluded in one of the
            # input images but not all of them.
            # If avg_reprojection is enabled the number of images to select
            # the minimum loss from is reduced by average-combining them.
            if self.avg_reprojection:
                reprojection_loss = reprojection_loss.mean(1, keepdim=True)

            # Pixels that are equal in the (unwarped) source image
            # and target image (e.g. no motion) are not that helpful
            # and can be masked out.
            if self.automasking:
                reprojection_loss = torch.cat(
                    (identity_reprojection_loss, reprojection_loss), 1
                )
                # Select the per-pixel minimum loss from
                # (prev_unwarped, next_unwarped, prev_unwarped, prev_warped).
                # Pixels where the unwarped input images are selected
                # act as gradient black holes, as nothing is backpropagated
                # into the network.
                reprojection_loss, idxs = torch.min(reprojection_loss, dim=1)

            # Segmentation moving mask to mask DC objects
            if outputs_masked is not None:
                moving_mask = outputs_masked['moving_mask']
                reprojection_loss = reprojection_loss * moving_mask

            loss = reprojection_loss.mean()
            losses[f'photometric/{resolution}'] = loss

            if self.disparity_smoothness != 0 or self.disparity_smoothness_seg != 0:
                disp = outputs["disp", resolution]
                mean_disp = disp.mean((2, 3), True)
                norm_disp = disp / (mean_disp + 1e-7)
            else:
                norm_disp = None

            if self.disparity_smoothness != 0:
                ref_color = functional.interpolate(
                    color, disp.shape[2:], mode='bilinear', align_corners=False
                )

                disp_smth_loss = self.smoothness(norm_disp, ref_color)
                disp_smth_loss = disp_smth_loss / (2 ** resolution)

                losses[f'disp_smth_loss/{resolution}'] = disp_smth_loss
                loss += self.disparity_smoothness * disp_smth_loss

            if self.disparity_smoothness_seg != 0:
                ref_seg = functional.interpolate(
                    segmentation, disp.shape[2:], mode='bilinear', align_corners=False
                )

                disp_smth_loss_seg = self.smoothness_seg(norm_disp, ref_seg)
                disp_smth_loss_seg = disp_smth_loss_seg / (2 ** resolution)

                losses[f'disp_smth_loss_seg/{resolution}'] = disp_smth_loss_seg
                loss += self.disparity_smoothness_seg * disp_smth_loss_seg

            losses[f'loss/{resolution}'] = loss

        losses['loss_depth_reprojection'] = sum(
            losses[f'loss/{resolution}']
            for resolution in resolutions
        ) / len(resolutions)

        return losses

    def compute_losses(self, inputs, outputs, outputs_masked, outputs_seg):
        if self.supervision_mode == 'self':
            losses = self._reprojection_losses(inputs, outputs, outputs_masked, outputs_seg)
            losses['loss_depth'] = losses['loss_depth_reprojection']
        elif self.supervision_mode == 'semi':
            losses_reprojection = self._reprojection_losses(inputs, outputs, outputs_masked, outputs_seg)
            losses_supervised = self._supervision_losses(inputs, outputs)
            losses = {**losses_reprojection, **losses_supervised}
            losses['loss_depth'] = losses['loss_depth_reprojection'] + losses['loss_depth_supervised']
        elif self.supervision_mode == 'sup':
            losses = self._supervision_losses(inputs, outputs)
            losses['loss_depth'] = losses['loss_depth_supervised']
        else:
            raise ValueError('Unsupported depth supervision mode')
        return losses
