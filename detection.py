import sys

import torch
import numpy as np

from utils.ssim import SSIM


class EdgeConsistencyDetector(object):
    def __init__(self, thresholds_rs, threshold_rd, threshold_sd, device, grad_percentile=0.95,
                 rgb_grad_method="abs", seg_grad_method="unequal", depth_grad_method="abs",
                 cons_method="ssim"):
        if None not in [thresholds_rs, threshold_rd, threshold_sd]:
            self.thresholds = [thresholds_rs, threshold_rd, threshold_sd]
        else:
            self.thresholds = None
        self.grad_percentile = grad_percentile
        self.rgb_grad_method = rgb_grad_method
        self.seg_grad_method = seg_grad_method
        self.depth_grad_method = depth_grad_method
        self.cons_method = cons_method
        self.ssim = SSIM().to(device)

        self.color = None
        self.segmentation = None
        self.depth = None
        self.scores = {"det_rgb_seg": list(), "det_rgb_depth": list(), "det_seg_depth": list(),
                       "det_overall": list(), "det_rates": list()}

        self.cons_rgb_seg = list()
        self.cons_rgb_depth = list()
        self.cons_seg_depth = list()

    def _calc_gradients(self, img, method='abs'):
        if method == "abs":
            grad_x = torch.abs(img[:, :, :, :-1] - img[:, :, :, 1:]).mean(dim=1, keepdim=True)
            grad_y = torch.abs(img[:, :, :-1, :] - img[:, :, 1:, :]).mean(dim=1, keepdim=True)
        elif method == "norm_abs":
            grad_x = torch.abs(img[:, :, :, :-1] - img[:, :, :, 1:]).mean(dim=1, keepdim=True)
            for i in range(grad_x.shape[0]):
                grad_x[i] = grad_x[i] / grad_x[i].mean()
            grad_y = torch.abs(img[:, :, :-1, :] - img[:, :, 1:, :]).mean(dim=1, keepdim=True)
            for i in range(grad_y.shape[0]):
                grad_y[i] = grad_y[i] / grad_y[i].mean()
        elif method == "unequal":
            grad_x = torch.logical_not(torch.eq(img[:, :, :, :-1], img[:, :, :, 1:])).float().mean(dim=1, keepdim=True)
            grad_y = torch.logical_not(torch.eq(img[:, :, :-1, :], img[:, :, 1:, :])).float().mean(dim=1, keepdim=True)
        elif method == "percentile":
            grad_x = torch.abs(img[:, :, :, :-1] - img[:, :, :, 1:]).mean(dim=1, keepdim=True)
            for i in range(grad_x.shape[0]):
                threshold = torch.quantile(grad_x[i].flatten(), self.grad_percentile)
                grad_x[i] = torch.ge(grad_x[i], threshold).float()

            grad_y = torch.abs(img[:, :, :-1, :] - img[:, :, 1:, :]).mean(dim=1, keepdim=True)
            for i in range(grad_y.shape[0]):
                threshold = torch.quantile(grad_y[i].flatten(), self.grad_percentile)
                grad_y[i] = torch.ge(grad_y[i], threshold).float()
        else:
            raise NotImplementedError('The gradient calculation method does not exist')
        grad_x = (grad_x - grad_x.min()) / (grad_x.max() - grad_x.min())
        grad_y = (grad_y - grad_y.min()) / (grad_y.max() - grad_y.min())
        return grad_x, grad_y

    def _calc_consistency(self, grad_1, grad_2, method="ssim"):
        if method == "mae":
            error = torch.abs(grad_1 - grad_2).mean(dim=(1, 2, 3))
        elif method == "ssim":
            error = self.ssim(grad_1, grad_2)
        else:
            raise NotImplementedError('The consistency method is not implemented, remember to adapt all if conditions')
        return error

    def _generate_detection(self, cons_rs, cons_rd, cons_sd, method):
        detection = list()
        cons = (cons_rs, cons_rd, cons_sd)
        if method == "ssim":
            for c, t in zip(cons, self.thresholds):
                detection.append(torch.lt(c, t).float())
        elif method == "mae":
            for c, t in zip(cons, self.thresholds):
                detection.append(torch.gt(c, t).float())
        else:
            raise NotImplementedError("Method not implemented")

        detection = torch.stack(detection, dim=0)
        detection_tot = torch.gt(detection.mean(dim=0), 0.5).float()
        return detection.numpy(), detection_tot.numpy()

    def reset(self):
        self.color = None
        self.segmentation = None
        self.depth = None

    def enter_color(self, color, mean=(0, 0, 0), std=(1, 1, 1)):
        color = torch.mul(color, torch.from_numpy(np.array(std)).float().unsqueeze(-1).unsqueeze(-1).unsqueeze(0))
        color = torch.sub(color, torch.from_numpy(-1 * np.array(mean)).float().unsqueeze(-1).unsqueeze(-1).unsqueeze(0))
        self.color = color

    def enter_segmentation(self, segmentation):
        softmax = torch.nn.Softmax(dim=1)
        segmentation = softmax(segmentation)
        self.segmentation = torch.argmax(segmentation, dim=1, keepdim=True)

    def enter_depth(self, depth, scale_factor=1.0):
        self.depth = depth/depth.mean((2, 3))

    def calculate_scores(self):
        if self.color is None or self.segmentation is None or self.depth is None:
            raise RuntimeError('Color, depth or segmentation not entered correctly')
        grad_x_rgb, grad_y_rgb = self._calc_gradients(self.color, method=self.rgb_grad_method)
        grad_x_seg, grad_y_seg = self._calc_gradients(self.segmentation, method=self.seg_grad_method)
        grad_x_depth, grad_y_depth = self._calc_gradients(self.depth, method=self.depth_grad_method)

        cons_rgb_seg = self._calc_consistency(grad_x_rgb, grad_x_seg, method=self.cons_method) + \
                       self._calc_consistency(grad_y_rgb, grad_y_seg, method=self.cons_method)
        cons_rgb_depth = self._calc_consistency(grad_x_rgb, grad_x_depth, method=self.cons_method) + \
                         self._calc_consistency(grad_y_rgb, grad_y_depth, method=self.cons_method)
        cons_seg_depth = self._calc_consistency(grad_x_seg, grad_x_depth, method=self.cons_method) + \
                         self._calc_consistency(grad_y_seg, grad_y_depth, method=self.cons_method)

        cons_rgb_seg /= 2.0
        cons_rgb_depth /= 2.0
        cons_seg_depth /= 2.0
        self.cons_rgb_seg.extend(list(cons_rgb_seg.numpy()))
        self.cons_rgb_depth.extend(list(cons_rgb_depth.numpy()))
        self.cons_seg_depth.extend(list(cons_seg_depth.numpy()))

        if self.thresholds is not None:
            detection, detection_tot = self._generate_detection(cons_rgb_seg,
                                                                cons_rgb_depth,
                                                                cons_seg_depth,
                                                                self.cons_method)

            det_rgb_seg, det_rgb_depth, det_seg_depth = detection
            self.scores["det_rgb_seg"].extend(list(det_rgb_seg))
            self.scores["det_rgb_depth"].extend(list(det_rgb_depth))
            self.scores["det_seg_depth"].extend(list(det_seg_depth))
            self.scores["det_overall"].extend(list(detection_tot))

        return cons_rgb_seg, cons_rgb_depth, cons_seg_depth

    def return_threshs(self, percentile=0.1):
        percentile = np.round(percentile * 100)
        measures = [self.cons_rgb_seg, self.cons_rgb_depth, self.cons_seg_depth]
        if self.cons_method == "ssim":
            low = True
        elif self.cons_method == "mae":
            low = False
        else:
            raise NotImplementedError("Method not implemented")
        if low:
            self.thresholds = [np.percentile(np.array(m), percentile) for m in measures]
        else:
            self.thresholds = [np.percentile(np.array(m), 100 - percentile) for m in measures]
        return self.thresholds

    def return_scores(self):
        if len(self.scores["det_overall"]) > 0:
            self.scores["det_rates"] = np.array(self.scores["det_overall"]).mean()
        return self.scores
