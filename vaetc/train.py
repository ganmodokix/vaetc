import os
import argparse
import math
import random

import numpy as np
from tqdm import tqdm
import yaml

import torch
from torch.utils.data import DataLoader

import torchinfo

from .data.utils import IMAGE_SHAPE, IMAGE_HEIGHT, IMAGE_WIDTH
from .utils import debug_print, mean_dict
from .checkpoint import Checkpoint
from .models import RLModel, AutoEncoderRLModel, GaussianEncoderAutoEncoderRLModel
from .evaluation import visualizations as vis

def non_deterministic():
    """ Speed up training at the cost of determinacy """

    torch.use_deterministic_algorithms(False)
    torch.backends.cudnn.benchmark = True

def deterministic(seed: int):
    """ Manually set the random seed and *make training deterministic again*

    Args:
        seed (int): seed value, 64bit signed int
    """

    # make it deterministic
    torch.backends.cudnn.benchmark = False
    torch.use_deterministic_algorithms(True)
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"

    # seeds
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

def show_summary(model: RLModel):
    """Shows the model summary using `torchinfo`

    Args:
        model (RLModel): An instance of `RLModel` whose summary is shown

    Returns:
        torchinfo.ModelStatistics: statictics about `model`
    """
    
    if model.inputs_include_targets:
        t_dim = model.t_dim
        stat = torchinfo.summary(model, input_size=[(1, *IMAGE_SHAPE), (1, t_dim)])
    else:
        stat = torchinfo.summary(model, input_size=(1, *IMAGE_SHAPE))
    
    return stat

def very_verbose_output(checkpoint: Checkpoint, epoch: int):
    """ Verbose output during the training if --very_verbose option specified

    Args:
        checkpoint (Checkpoint): An instance of training checkpoint
        epoch (int): the current epoch (0-indexed) for file name prefix
    """

    os.makedirs(os.path.join(checkpoint.options["logger_path"], "verbose"), exist_ok=True)

    checkpoint.model.eval()

    if isinstance(checkpoint.model, AutoEncoderRLModel):
        debug_print("Plotting latent traversals...")
        vis.traversal.visualize(checkpoint, traversal_path=f"verbose/traversal_{epoch:04d}.png")
    else:
        debug_print("Latent Traversal skipped; no decoder")

    if checkpoint.options["cuda_sync"]:
        torch.cuda.synchronize()
        torch.cuda.empty_cache()

    if isinstance(checkpoint.model, AutoEncoderRLModel):
        debug_print("Reconstructing examples...")
        vis.reconstruction.visualize(checkpoint, reconstruction_path=f"verbose/reconstruction_{epoch:04d}.png")
    else:
        debug_print("Reconstructions skipped; no decoder")

    if checkpoint.options["cuda_sync"]:
        torch.cuda.synchronize()
        torch.cuda.empty_cache()

    if isinstance(checkpoint.model, GaussianEncoderAutoEncoderRLModel):
        debug_print("Sampling images...")
        vis.sample.visualize(checkpoint, out_path=f"verbose/samples_{epoch:04d}.png")
    else:
        debug_print("Sampling skipped; no decoder")

    if checkpoint.options["cuda_sync"]:
        torch.cuda.synchronize()
        torch.cuda.empty_cache()

def train_one_epoch(checkpoint: Checkpoint, loader_train: DataLoader, epoch: int) -> dict[str, float]:
    """ Trains one epoch and returns the training loss value

    Args:
        checkpoint (Checkpoint): An instance of training checkpoint
        loader_train (DataLoader): The data loader of the training set
        epoch (int): the current epoch (0-indexed) for file name prefix

    Returns:
        epoch_history (Dict[str, float])
            The keys are the names of losses
    """

    # Switch the model module to training mode
    checkpoint.model.train()

    history_steps = []

    # training for each minibatch
    iter_loader = tqdm(loader_train, desc="train")
    for batch_index, batch in enumerate(iter_loader):

        # debug_print(f"x in [{batch[0].min().item():.3f}, {batch[0].max().item():.3f}]")

        # 0 in the first batch of the first epoch, 1 in the last batch of the last epoch
        batches_per_epoch = len(loader_train)
        batch_index_entire = epoch * batches_per_epoch + batch_index
        num_batches_entire = max(1, checkpoint.options["epochs"] * batches_per_epoch - 1)
        progress = batch_index_entire / num_batches_entire

        # # LR decay
        if not checkpoint.options.get("until_convergence", False) and "lr_decay" in checkpoint.options:
            lr_decay = float(checkpoint.options["lr_decay"])
            lr_multiplier = lr_decay ** progress
            lr_original = float(yaml.safe_load(checkpoint.options["hyperparameters"])["lr"])
            lr_current = lr_original * lr_multiplier
            for g in checkpoint.optimizers["main"].param_groups:
                g["lr"] = lr_current
        
        loss_dict = checkpoint.model.train_batch(batch, checkpoint.optimizers, progress)
        history_steps.append(loss_dict)
        
        loss_to_show = loss_dict["loss"] if "loss" in loss_dict else math.nan
        iter_loader.set_description(desc=f'train loss {loss_to_show: 9.3f}')

        # Crashes if NaN has been encountered
        if any(map(math.isnan, loss_dict.values())):
            raise ValueError("NaN detected!")

        if checkpoint.options["cuda_sync"]:
            torch.cuda.synchronize()

    # mean
    epoch_history = mean_dict(history_steps)

    if checkpoint.options["cuda_sync"]:
        torch.cuda.synchronize()
        torch.cuda.empty_cache()

    return epoch_history

def validate_one_epoch(checkpoint: Checkpoint, loader_valid: DataLoader) -> dict[str, float]:
    """ Computes validation losses for one epoch

    Args:
        checkpoint (Checkpoint): An instance of checkpoint
        loader_valid (DataLoader): The data loader of the validation set
        epoch (int): the current epoch (0-indexed) for file name suffices

    Returns:
        epoch_history (Dict[str, float])
            The keys are the names of losses
    """

    history_steps = []

    checkpoint.model.eval()

    for batch in tqdm(loader_valid, desc="valid"):

        loss_dict = checkpoint.model.eval_batch(batch)
        history_steps.append(loss_dict)

        if checkpoint.options["cuda_sync"]:
            torch.cuda.synchronize()
    
    epoch_history_val = mean_dict(history_steps)

    if checkpoint.options["cuda_sync"]:
        torch.cuda.synchronize()
        torch.cuda.empty_cache()

    return epoch_history_val

def one_epoch(checkpoint: Checkpoint, loader_train: DataLoader, loader_valid: DataLoader):
    """ One execution of training routines 

    Args:
        checkpoint (Checkpoint): An instance of checkpoint
        loader_valid (DataLoader): The data loader of the validation set
        loader_train (DataLoader): The data loader of the training set
        epoch (int): the number of previously trained epochs (i.e., 0-indexed)
    """

    epoch = checkpoint.training_state["epochs"]

    debug_print(f'Epoch {epoch+1}/{checkpoint.options["epochs"]}')

    # Training
    epoch_history = train_one_epoch(checkpoint, loader_train, epoch)

    # Validation
    epoch_history_val = validate_one_epoch(checkpoint, loader_valid)

    # increments the training history
    checkpoint.append_epoch_history(epoch_history)
    checkpoint.append_epoch_history(epoch_history_val, prefix="val")
    checkpoint.training_state["epochs"] += 1

    # save checkpoint
    checkpoint.save_if_better("checkpoint_best.pth")
    checkpoint.save("checkpoint_last.pth")

    # print losses for debugging
    debug_print(f"train: {epoch_history}")
    debug_print(f"valid: {epoch_history_val}")

    # visualize if --very_verbose specified
    if checkpoint.options["very_verbose"]:
        very_verbose_output(checkpoint, epoch)

    return checkpoint

def train(checkpoint: Checkpoint):
    """ train model *in place*

    Args:
        checkpoint: Checkpoint
            A checkpoint instance to train

    Raises:
        ValueError
            raises error if training/validation losses include NaN

    Returns:
        Checkpoint
            the same instance as :obj:`checkpoint`
    """

    # Creating the logger directory
    debug_print(f'Directory set to {checkpoint.options["logger_path"]}')
    os.makedirs(checkpoint.options["logger_path"], exist_ok=True)
    
    # Data Loaders
    debug_print(f'Preparing data loaders...')
    loader_train = DataLoader(
        dataset=checkpoint.dataset.training_set,
        batch_size=checkpoint.options["batch_size"],
        shuffle=True,
        num_workers=os.cpu_count() - 1,
        pin_memory=True)
    loader_valid = DataLoader(
        dataset=checkpoint.dataset.validation_set,
        batch_size=checkpoint.options["batch_size"],
        shuffle=False,
        num_workers=os.cpu_count() - 1,
        pin_memory=True)

    # Show the model summary
    show_summary(checkpoint.model)

    # Training Loop
    debug_print("Training starting with hyperparameters below:")
    debug_print(yaml.safe_dump(yaml.safe_load(checkpoint.options["hyperparameters"])))
    
    while True:

        if checkpoint.options.get("until_convergence", False):
            
            # until convergenve (early stopping)
            history_val_loss = checkpoint.history.get("val_loss", [])
            best_epoch = np.argmin([float("inf"), *history_val_loss]) # 1-indexed
            non_best_recent_epochs = len(history_val_loss) - best_epoch
            patience = checkpoint.options["patience"]

            if non_best_recent_epochs >= patience:
                debug_print(f"Training stopped because of {non_best_recent_epochs} epochs (>= patience {patience}) without any best update in val_loss")
                break
            else:
                checkpoint.options["epochs"] = checkpoint.training_state["epochs"] + 1
                debug_print(f"(--until_convergence) {non_best_recent_epochs} epochs without any update in val_loss (patience: {patience})")

        else:
            
            # constant number of epochs
            trained_epochs = checkpoint.training_state["epochs"]
            required_epochs = checkpoint.options["epochs"]

            if trained_epochs >= required_epochs:
                debug_print(f"""Already {trained_epochs} epochs trained (>= {required_epochs} epochs)""")
                break

        one_epoch(checkpoint, loader_train, loader_valid)

    return checkpoint

if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    parser.add_argument("model_name", type=str, help="model name (e.g., 'vae')")
    parser.add_argument("hyperparameters", type=str, help="hyper-parameters in the YAML format")
    parser.add_argument("--dataset", type=str, default="celeba", help="dataset to train with")
    parser.add_argument("--logger_path", type=str, default="runs/current", help="epochs to train")
    parser.add_argument("--epochs", type=int, default=50, help="epochs to train")
    parser.add_argument("--batch_size", type=int, default=64, help="batch size")
    parser.add_argument("--cuda_sync", action="store_true", help="syncronize CUDA for each batch (not recommended; it's slow)")
    parser.add_argument("--very_verbose", action="store_true", help="visualize the model at the end of each epoch")

    args = parser.parse_args()

    options = vars(args)

    checkpoint = Checkpoint(options)

    train(checkpoint)
