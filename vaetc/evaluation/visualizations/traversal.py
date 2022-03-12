import os

import numpy as np
import yaml
from tqdm import tqdm
import cv2

import torch

from vaetc.checkpoint import Checkpoint
from vaetc.utils import write_video

def render(options, model, i: int, n: int = 15, radius: float = 3.0):

    if "decode" in model.__dict__:
        raise ValueError("The model has no decoder")

    model.eval()

    hyperparameters = yaml.safe_load(options["hyperparameters"])
    l = hyperparameters["z_dim"]
    
    z = np.zeros(shape=(n, l))
    z[:,i] = np.linspace(-radius, radius, n)
    
    with torch.no_grad():
        z = torch.FloatTensor(z).cuda()
        x = model.decode(z)
        x = x.detach().cpu().numpy()

    return np.concatenate(x, axis=2)

def video(checkpoint: Checkpoint, video_dir_name: str = "traversal_video"):
    
    video_dir_path = os.path.join(checkpoint.options["logger_path"], video_dir_name)
    hyperparameters = yaml.safe_load(checkpoint.options["hyperparameters"])

    for i in tqdm(range(hyperparameters["z_dim"])):
        traversal = render(checkpoint.options, checkpoint.model, i, n=30) # (N, C, H, W)
        traversal = np.concatenate([traversal, traversal[-2:0:-1]], axis=0)
        output_path = os.path.join(video_dir_path, f"z_{i:03d}.mp4")
        write_video(output_path, traversal, framerate=60)

def visualize(checkpoint: Checkpoint, traversal_path: str = "traversal.png"):

    hyperparameters = yaml.safe_load(checkpoint.options["hyperparameters"])

    traversals = []
    for i in tqdm(range(hyperparameters["z_dim"])):
        traversals += [render(checkpoint.options, checkpoint.model, i)]
    
    img = np.concatenate(traversals, axis=1)
    img = (img.transpose(1, 2, 0)[...,::-1] * 255).astype(np.uint8)
    cv2.imwrite(os.path.join(checkpoint.options["logger_path"], traversal_path), img)