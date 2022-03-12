import os
import sys
from typing import Dict, Optional, List, overload
import copy
import argparse

import yaml
import numpy as np

import torch
from torch.utils.data import DataLoader

from .models import model_by_params
from . import data
from .utils import debug_print

def build_model(options: dict, dataset: data.ImageDataset):

    if isinstance(options["hyperparameters"], str):
        hyperparameters = yaml.safe_load(options["hyperparameters"])
    else:
        hyperparameters = options["hyperparameters"]
    
    # Loader for detecting target dims (immediately discarded)
    x, t = dataset.test_set.__getitem__(0)
    hyperparameters["t_dim"] = t.shape[0]
    debug_print(f"""hyperparameter 't_dim' is set to {hyperparameters["t_dim"]}""")

    # model
    model = model_by_params(
        model_name=options["model_name"],
        hyperparameters=hyperparameters
    )
    model = model.cuda()

    return model

def load_checkpoint(checkpoint_path: str):

    state_dict = torch.load(checkpoint_path)

    state_dict["options"]["logger_path"] = os.path.dirname(checkpoint_path)

    checkpoint = Checkpoint(state_dict["options"])
    checkpoint.load_state_dict(state_dict)

    return checkpoint

LOAD_DATASET_CACHE = {}
def load_dataset(dataset_name: str) -> data.ImageDataset:

    if dataset_name not in LOAD_DATASET_CACHE:

        if dataset_name != "image_dataset" and dataset_name in data.__dict__:

            LOAD_DATASET_CACHE[dataset_name] = data.__dict__[dataset_name]()

        else:

            LOAD_DATASET_CACHE[dataset_name] = data.image_dataset(dataset_name)

    return LOAD_DATASET_CACHE[dataset_name]

class Checkpoint:
    """ A training context including a model, options, training states and history """

    def __init__(self, options: dict):
        """
        Args:
            options (dict): training options. The keys are follows:

                :obj:`"model_name"` (str)
                    model name registred in :func:`vaetc.models.register_model`

                :obj:`"hyperparameters"` (str)
                    hyperparameters in a YAML-formatted string

                :obj:`"dataset"` (str)
                    dataset name that can be called by `vaetc.data.${dataset}()`

                :obj:`"logger_path"` (str)
                    path to the directory to save the checkpoint and results in

                :obj:`"epochs"` (int)
                    # of epochs to train, ignored if `"until_convergence"` is True

                :obj:`"batch_size"` (int)
                    size of mini-batches to train

                :obj:`"cuda_sync"` (bool)
                    if True, :func:`torch.cuda.synchronize` and
                    :func:`torch.cuda.empty_cache` are called in the end of each epoch

                :obj:`"very_verbose"` (bool)
                    if True, reconstructions and latent traversals are saved in the end of each epoch

                :obj:`"until_convergence"` (Optional, :obj:`bool`)
                    if True, early stopping with :obj:`"patience"` of patience epochs, watching history["val_loss"]
                
                :obj:`"patience"` (Optional, :obj:`int`)
                    if :obj:`"until_convergence"` is True, early stopping with patience
        """
        
        self.options = {}
        self.options.update(options)

        dataset_name = self.options["dataset"]
        self.dataset = load_dataset(dataset_name)

        self.model = build_model(options, self.dataset)

        debug_print(f"""WARNING: Using all the {torch.cuda.device_count()} CUDA devices""")
        self.model_parallel = torch.nn.DataParallel(self.model)
        self.model = self.model_parallel.module

        self.optimizers = self.model.build_optimizers()

        self.history: Dict[str, List[float]] = {}
        
        self.training_state = {
            "epochs": 0,
        }

    def entire_state_dict(self):

        return {
            "options": copy.deepcopy(self.options),
            "model_state_dict": self.model.state_dict(),
            "optimizers_state_dict": {name: self.optimizers[name].state_dict() for name in self.optimizers},
            "history": copy.deepcopy(self.history),
            "training_state": copy.deepcopy(self.training_state),
        }

    def load_state_dict(self, state_dict):

        self.options = state_dict["options"]
        self.model.load_state_dict(state_dict["model_state_dict"])
        for name in self.optimizers:
            self.optimizers[name].load_state_dict(state_dict["optimizers_state_dict"][name])
        self.history = state_dict["history"]
        self.training_state = state_dict["training_state"]

    def save(self, file_name: str = "checkpoint.pth"):

        path_to_save = os.path.join(self.options["logger_path"], file_name)
        torch.save(self.entire_state_dict(), path_to_save)

    def save_if_better(self, file_name: str = "checkpoint.pth", criterion: str = "val_loss", lower_is_better: bool = True):

        path_to_save = os.path.join(self.options["logger_path"], file_name)

        if not os.path.exists(path_to_save):
            self.save(file_name)
            return

        previous_state = torch.load(path_to_save)
        previous_criterion_value = previous_state["history"][criterion][-1]
        current_criterion_value  = self.history[criterion][-1]

        if lower_is_better:
            is_current_better = previous_criterion_value > current_criterion_value
        else:
            is_current_better = previous_criterion_value < current_criterion_value

        if is_current_better:
            self.save(file_name)

    def append_epoch_history(self, epoch_history: Dict[str, float], prefix: Optional[str] = None):

        for key in epoch_history:

            dest_key = f"{prefix}_{key}" if prefix is not None else key

            if dest_key not in self.history:
                self.history[dest_key] = []

            self.history[dest_key] += [epoch_history[key]]
    
    def last_history(self):
        assert self.epochs > 0

def investigate(checkpoint_path):

    state_dict = torch.load(checkpoint_path)

    shown_keys = {"options", "training_state"}
    shown_state = {name: state_dict[name] for name in shown_keys}

    yaml.safe_dump(shown_state, sys.stderr, default_flow_style=False)

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint_path", type=str, default="runs/current/checkpoint_last.pth", help="path to the checkpoint to be investigated")
    args = parser.parse_args()

    investigate(args.checkpoint_path)