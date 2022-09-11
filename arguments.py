from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter


class ArgumentsBase(object):
    DESCRIPTION = 'Gsusdn Arguments'

    def __init__(self):
        self.ap = ArgumentParser(
            description=self.DESCRIPTION,
            formatter_class=ArgumentDefaultsHelpFormatter
        )

    def _harness_init_system(self):
        self.ap.add_argument(
            '--sys-device-name', type=str, default='cuda:0',
            help='Disable Hardware acceleration'
        )

        self.ap.add_argument(
            '--sys-num-workers', type=int, default=3,
            help='Number of worker processes to spawn per DataLoader'
        )

        self.ap.add_argument(
            '--sys-best-effort-determinism', default=False, action='store_true',
            help='Try and make some parts of the training/validation deterministic'
        )

    def _harness_init_model(self):
        self.ap.add_argument(
            '--model-name', type=str, default='gsusdn_base',
            help='A nickname for this model'
        )

        self.ap.add_argument(
            '--model-load', type=str, default=None,
            help='Load a model state from a state directory containing *.pth files'
        )

        self.ap.add_argument(
            '--model-num-layers', type=int, default=18, choices=(18, 34, 50, 101, 152),
            help='Number of ResNet Layers in the depth and segmentation encoder'
        )

        self.ap.add_argument(
            '--model-num-layers-pose', type=int, default=18, choices=(18, 34, 50, 101, 152),
            help='Number of ResNet Layers in the pose encoder'
        )

        self.ap.add_argument(
            '--model-split-pos', type=int, default=1, choices=(0, 1, 2, 3, 4, 5),
            help='Position in the decoder to split from common to separate depth/segmentation decoders'
        )

        self.ap.add_argument(
            '--model-depth-min', type=float, default=0.1,
            help='Depth Estimates are scaled according to this min/max',
        )

        self.ap.add_argument(
            '--model-depth-max', type=float, default=100.0,
            help='Depth Estimates are scaled according to this min/max',
        )

        self.ap.add_argument(
            '--model-depth-resolutions', type=int, default=4, choices=(1, 2, 3, 4),
            help='Number of depth resolutions to generate in the network'
        )

    def _harness_init_joint(self):
        self.ap.add_argument(
            '--joint-validation-resize-height', type=int, default=512,
            help='Segmentation images are resized to this height prior to cropping'
        )

        self.ap.add_argument(
            '--joint-validation-resize-width', type=int, default=1024,
            help='Segmentation images are resized to this width prior to cropping'
        )

        self.ap.add_argument(
            '--joint-validation-loaders', type=str, default='',
            help='Comma separated list of segmentation dataset loaders from loaders/segmentation.py '
                 'to use for validation'
        )

        self.ap.add_argument(
            '--joint-validation-batch-size', type=int, default=1,
            help='Batch size for segmentation validation'
        )

        self.ap.add_argument(
            '--joint-validation-perturbation-strength', type=float, default=0.0,
            help='strength of the perturbation'
        )

        self.ap.add_argument(
            '--joint-validation-perturbation-type', type=str, default='gaussian',
            help='Type of the perturbation'
        )

        self.ap.add_argument(
            '--joint-depth-validation-fixed-scaling', type=float, default=0,
            help='Use this fixed scaling ratio (from another run) for validation outputs'
        )

    def _eval_init(self):
        self.ap.add_argument(
            '--eval-num-images', type=int, default=20,
            help='Number of generated images to store to disk'
        )

        self.ap.add_argument(
            '--eval-segmentation-remaps', type=str, default='none',
            help='Segmentation label remap modes for reduced number of classes'
        )

        self.ap.add_argument(
            '--eval-edge-extraction', type=str, default='abs',
            help='mode for edge extraction'
        )

        self.ap.add_argument(
            '--eval-consistency-calculation', type=str, default='ssim',
            help='mode for consistency calculation'
        )

        self.ap.add_argument(
            '--eval-run-detection', action='store_true',
            help='Run an additional detection model in the joint evaluation'
        )

        self.ap.add_argument(
            '--eval-dont-save-threshs', action='store_true',
            help='Do not save the thresholds but load them'
        )

        self.ap.add_argument(
            '--eval-disparity-smoothness', type=float, default=0.0,
            help='Disparity RGB smoothness to the input image to consider during evaluation'
        )

        self.ap.add_argument(
            '--eval-disparity-smoothness-seg', type=float, default=0.0,
            help='Disparity segmentation smoothness to consider during evaluation'
        )

        self.ap.add_argument(
            '--eval-segmentation-smoothness', type=float, default=0.0,
            help='Segmentation RGB smoothness to consider during evaluation'
        )

        self.ap.add_argument(
            '--eval-seg-loss-scale', type=float, default=1.0,
            help='Segmentation loss scale to consider during evaluation'
        )

        self.ap.add_argument(
            '--eval-depth-loss-scale', type=float, default=0.0,
            help='Depth loss scale to consider during evaluation'
        )

    def _parse(self):
        return self.ap.parse_args()


class JointEvaluationArguments(ArgumentsBase):
    DESCRIPTION = 'Gsusdn Segmentation Evaluation'

    def __init__(self):
        super().__init__()

        self._harness_init_system()
        self._harness_init_model()
        self._harness_init_joint()
        self._eval_init()

    def parse(self):
        opt = self._parse()

        # These options are required by the StateManager
        # but are effectively ignored when evaluating so
        # they can be initialized to arbitrary values
        opt.train_optimizer = 'adam'
        opt.train_learning_rate = 0
        opt.train_scheduler_step_size = 1000
        opt.train_weight_decay = 0
        opt.train_mode = 'train_all'
        opt.train_weights_init = 'scratch'
        opt.train_depth_grad_scale = 1
        opt.train_segmentation_grad_scale = 1
        opt.train_domain_grad_scale = 1
        opt.train_depth_loss_scale = 0
        opt.train_segmentation_loss_scale = 0

        return opt
