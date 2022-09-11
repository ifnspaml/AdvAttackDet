import os

import torch

from models.sgdepth import SGDepth



class ModelContext(object):
    def __init__(self, model, mode, train_mode):
        self.model = model
        self.mode_wanted = mode
        self.train_mode = train_mode

    def _set_mode(self, mode):
        if mode == 'train':
            if self.train_mode == 'train_all':
                self.model.train()
            elif self.train_mode == 'freeze_encoder':
                self.model.train()
                self.model.common.eval()
            else:
                raise ValueError('Invalid train mode')

        elif mode == 'eval':
            self.model.eval()

    def __enter__(self):
        self.mode_was = 'train' if self.model.training else 'eval'

        self._set_mode(self.mode_wanted)

        return self.model

    def __exit__(self, *_):
        self._set_mode(self.mode_was)


class ModelManager(object):
    def __init__(self, model, train_mode):
        self.model = model
        self.train_mode = train_mode

    def get_eval(self):
        return ModelContext(self.model, 'eval', self.train_mode)

    def get_train(self):
        return ModelContext(self.model, 'train', self.train_mode)


class StateManager(object):
    def __init__(self, model_name, device, split_pos, num_layers, grad_scale_depth,
                 grad_scale_seg, grad_scale_domain, weights_init, resolutions_depth, train_mode):

        self.device = device

        self.log_base = "pretrained"
        self.log_path = os.path.join(self.log_base, model_name)

        self._init_training()
        self._init_model(
            split_pos, num_layers, grad_scale_depth, grad_scale_seg, grad_scale_domain,
            weights_init, resolutions_depth, train_mode,
        )

    def _init_training(self):
        self.epoch = 0
        self.step = 0

    def _init_model(self, split_pos, num_layers, grad_scale_depth, grad_scale_seg, grad_scale_domain,
                    weights_init, resolutions_depth, train_mode):

        model = SGDepth(
            split_pos, num_layers, grad_scale_depth, grad_scale_seg, grad_scale_domain,
            weights_init, resolutions_depth
        )

        model = model.to(self.device)

        self.model_manager = ModelManager(model, train_mode)

    def _state_dir_paths(self, state_dir):
        return {
            'model': os.path.join(self.log_base, state_dir, "model.pth"),
        }

    def store_state(self, state_dir):
        print(f"Storing model state to {state_dir}:")
        os.makedirs(state_dir, exist_ok=True)

        paths = self._state_dir_paths(state_dir)

        with self.model_manager.get_train() as model:
            torch.save(model.state_dict(), paths['model'])

    def store_checkpoint(self):
        state_dir = os.path.join(self.log_path, "checkpoints", f"epoch_{self.epoch}")
        self.store_state(state_dir)

    # Store Checkpoints during the sequential training of batches
    def store_batch_checkpoint(self, directory_naming):
        state_dir = os.path.join(self.log_path, "checkpoints", directory_naming)
        self.store_state(state_dir)

    def _load_model_state(self, path):
        with self.model_manager.get_train() as model:
            state = model.state_dict()
            to_load = torch.load(path, map_location=self.device)

            for (k, v) in to_load.items():
                if k not in state:
                    print(f"    - WARNING: Model file contains unknown key {k} ({list(v.shape)})")

            for (k, v) in state.items():
                if k not in to_load:
                    print(f"    - WARNING: Model file does not contain key {k} ({list(v.shape)})")

                else:
                    state[k] = to_load[k]

            model.load_state_dict(state)

    def load(self, state_dir):
        """Load model(s) from a state directory on disk
        """

        print(f"Loading checkpoint from {os.path.join(self.log_base, state_dir)}:")

        paths = self._state_dir_paths(state_dir)

        print(f"  - Loading model state from {paths['model']}:")
        try:
            self._load_model_state(paths['model'])
        except FileNotFoundError:
            print("   - Could not find model state file")
