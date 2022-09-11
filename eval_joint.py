#!/usr/bin/env python3

# Python standard library
import os
import pickle
import numpy as np

# Public libraries
import torch
from torchvision.utils import save_image

# Local imports
import colors
from arguments import JointEvaluationArguments
from harness import Harness
from attack import Attack
from detection import EdgeConsistencyDetector


class JointEvaluator(Harness):
    def _init_validation(self, opt):
        self.fixed_depth_scaling = opt.joint_depth_validation_fixed_scaling
        self.val_num_log_images = opt.eval_num_images
        self.eval_name = opt.model_name
        self.remaps = opt.eval_segmentation_remaps.split(',')
        self.attack_model = Attack(opt.joint_validation_perturbation_type, opt.joint_validation_perturbation_strength,
                                   self.device, opt.eval_disparity_smoothness, opt.eval_disparity_smoothness_seg,
                                   opt.eval_segmentation_smoothness, (opt.eval_seg_loss_scale, opt.eval_depth_loss_scale))
        self.joint_validation_perturbation_type = opt.joint_validation_perturbation_type
        self.joint_validation_perturbation_strength = opt.joint_validation_perturbation_strength
        self.run_detection = opt.eval_run_detection
        self.eval_dont_save_threshs = opt.eval_dont_save_threshs
        if self.run_detection:
            if not self.eval_dont_save_threshs:
                thresh = [None, None, None]
            else:
                thresh_file = os.path.join(self.log_path, "threshs.pickle")
                with open(thresh_file, 'rb') as handle:
                    thresh = pickle.load(handle)
            self.detection_model = EdgeConsistencyDetector(thresh[0], thresh[1], thresh[2], self.device,
                                                           rgb_grad_method=opt.eval_edge_extraction,
                                                           seg_grad_method="unequal",
                                                           depth_grad_method=opt.eval_edge_extraction,
                                                           cons_method=opt.eval_consistency_calculation)

    def evaluate(self):
        print('Evaluate joint predictions:', flush=True)

        depth_scores, depth_ratios, depth_images, seg_scores, seg_images, seg_perturbations = self._run_joint_validation(
            self.val_num_log_images, self.remaps
        )

        depth_metrics = dict()
        for domain in depth_scores:
            print(f'  - Results for domain {domain}:')

            if len(depth_ratios[domain]) > 0:
                ratios_np = np.array(depth_ratios[domain])

                ratio_median = np.median(ratios_np)
                ratio_norm_std = np.std(ratios_np / ratio_median)

                print(" Scaling ratios | med: {:0.3f} | std: {:0.3f}".format(ratio_median, ratio_norm_std))

            depth_metrics = depth_scores[domain].get_scores()

            print("\n  " + ("{:>8} | " * 7).format("abs_rel", "sq_rel", "rmse", "rmse_log", "a1", "a2", "a3"))
            print(("&{: 8.3f}  " * 7).format(depth_metrics['abs_rel'], depth_metrics['sq_rel'],
                                             depth_metrics['rmse'], depth_metrics['rmse_log'],
                                             depth_metrics['delta1'], depth_metrics['delta2'],
                                             depth_metrics['delta3']) + "\\\\")

        seg_metrics = dict()
        for domain in seg_scores:
            print('eval_name    | domain               | remap    |     miou | accuracy')

            for remap in seg_scores[domain]:
                seg_metrics = seg_scores[domain][remap].get_scores()

                miou = seg_metrics['meaniou']
                acc = seg_metrics['meanacc']

                print(f'{self.eval_name:12} | {domain:20} | {remap:8} | {miou:8.3f} | {acc:8.3f}', flush=True)

        if self.run_detection:
            if not self.eval_dont_save_threshs:
                thresh = self.detection_model.return_threshs(percentile=0.05)
                thresh_file = os.path.join(self.log_path, "threshs.pickle")
                with open(thresh_file, 'wb') as handle:
                    pickle.dump(thresh, handle, protocol=pickle.HIGHEST_PROTOCOL)
            scores = self.detection_model.return_scores()
            print(scores)

        self._log_gpu_memory()


if __name__ == "__main__":
    opt = JointEvaluationArguments().parse()

    if opt.model_load is None:
        raise Exception('You must use --model-load to select a model state directory to run evaluation on')

    if opt.sys_best_effort_determinism:
        import random
        import numpy as np

        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        np.random.seed(1)
        torch.manual_seed(1)
        torch.cuda.manual_seed(1)
        random.seed(1)

    evaluator = JointEvaluator(opt)
    evaluator.evaluate()
