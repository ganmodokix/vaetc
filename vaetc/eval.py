
import sys
import os

import argparse
import copy
from typing import Optional

import numpy as np
import yaml
import matplotlib.pyplot as plt
from tqdm import tqdm
import cv2

import torch
from torch.utils.data import DataLoader, Subset

from vaetc.utils import debug_print
from . import data
from .evaluation import visualizations as vis
from .evaluation import metrics as met
from .checkpoint import Checkpoint, load_checkpoint
from .models import AutoEncoderRLModel, GaussianEncoderAutoEncoderRLModel

def options(checkpoint_last: Checkpoint):

    debug_print("Visualizing the options...")

    with open(os.path.join(checkpoint_last.options["logger_path"], "options.yaml"), "w") as fp:
        yaml.safe_dump(checkpoint_last.options, fp)

def history(checkpoint_best: Checkpoint, checkpoint_last: Checkpoint):

    debug_print("Visualizing the training history...")
    vis.history.visualize(checkpoint_last)

    if isinstance(checkpoint_best.model, AutoEncoderRLModel):
        debug_print("Plotting latent traversals...")
        vis.traversal.visualize(checkpoint_best)
    else:
        debug_print("Latent Traversal skipped; no decoder")

    torch.cuda.synchronize()
    torch.cuda.empty_cache()

def reconstruction(checkpoint_best: Checkpoint):

    if isinstance(checkpoint_best.model, AutoEncoderRLModel):
        debug_print("Reconstructing examples...")
        vis.reconstruction.visualize(checkpoint_best)
    else:
        debug_print("Reconstructions skipped; no decoder")

    torch.cuda.synchronize()
    torch.cuda.empty_cache()

def marginal_distribution(checkpoint_best: Checkpoint):

    debug_print("Plotting empirical distributions...")
    vis.distribution.visualize(checkpoint_best)

    torch.cuda.synchronize()
    torch.cuda.empty_cache()

def manifold(checkpoint_best: Checkpoint):

    if isinstance(checkpoint_best.model, AutoEncoderRLModel):
        debug_print("Plotting manifold...")
        vis.manifold.visualize(checkpoint_best)
    else:
        debug_print("Manifold traversal skipped; no decoder")

    torch.cuda.synchronize()
    torch.cuda.empty_cache()

def sampling(checkpoint_best: Checkpoint):

    if isinstance(checkpoint_best.model, GaussianEncoderAutoEncoderRLModel):
        debug_print("Sampling data ...")
        vis.sample.visualize(checkpoint_best)
    else:
        debug_print("Generative Sampling skipped; no generative model")
    
    torch.cuda.synchronize()
    torch.cuda.empty_cache()

def metrics(checkpoint_best: Checkpoint):

    debug_print("Calculating metric values...")
    met.evaluate(checkpoint_best)

    torch.cuda.synchronize()
    torch.cuda.empty_cache()

def success_status(checkpoint_last: Checkpoint):

    logger_path = checkpoint_last.options["logger_path"]
    with open(os.path.join(logger_path, "exit_status.yaml"), "w") as fp:
        yaml.safe_dump({"exit_status": "ok"}, fp)
    
def visualize(
    checkpoint_best: Checkpoint,
    checkpoint_last: Optional[Checkpoint] = None,
    logging: bool = True,
    qualitative: bool = True,
    quantitative: bool = True):

    if checkpoint_last is None:
        checkpoint_last = checkpoint_best

    if logging:
        options(checkpoint_last)
        history(checkpoint_best, checkpoint_last)
    
    if qualitative:
        reconstruction(checkpoint_best)
        marginal_distribution(checkpoint_best)
        manifold(checkpoint_best)
        sampling(checkpoint_best)
    
    if quantitative:
        metrics(checkpoint_best)
    
    success_status(checkpoint_last)

def main(logger_path: str):

    assert os.path.isdir(logger_path), "logger_path must specify a directory"

    debug_print(f"Loading from {logger_path} ...")

    checkpoint_last = load_checkpoint(os.path.join(logger_path, "checkpoint_last.pth"))
    checkpoint_best = load_checkpoint(os.path.join(logger_path, "checkpoint_best.pth"))
    visualize(checkpoint_best, checkpoint_last)

if __name__ == "__main__":
    
    parser = argparse.ArgumentParser()

    parser.add_argument("--logger_path", type=str, default="runs/current",
        help="path to the run (e.g., runs/current)")

    args = parser.parse_args()

    main(args.logger_path)
