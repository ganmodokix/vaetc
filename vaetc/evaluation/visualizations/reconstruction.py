import os

import numpy as np
import cv2

import torch
from torch.utils.data import DataLoader, Subset

from vaetc.checkpoint import Checkpoint

def render(loader, model):

    if "decode" in model.__dict__:
        raise ValueError("The model has no decoder")

    model.eval()

    with torch.no_grad():

        x, t = loader.__iter__().next()
        
        x = x.to(model.device)
        z = model.encode(x)
        x2 = model.decode(z)

        x  = x .detach().cpu().numpy() # (N, C, H, W)
        x2 = x2.detach().cpu().numpy() # (N, C, H, W)

    return x, x2

def convert_to_cv2image(x, x2):

    img = np.concatenate([x, x2], axis=2)  # (N, C, H*2, W)
    img = np.concatenate(img, axis=2) # (C, H*2, W*N)
    img = img.transpose(1, 2, 0) # (H*2, W*N, C)
    img = (img[...,::-1] * 255).astype(np.uint8) # RGB -> BGR, float->uint8

    return img

def save_image(options, img, reconstruction_path):

    logger_path = options["logger_path"]

    path_to_save = os.path.join(logger_path, reconstruction_path)

    cv2.imwrite(path_to_save, img)


def visualize(checkpoint: Checkpoint, reconstruction_path: str = "reconstructions.png"):

    n = 10
    
    loader_test = DataLoader(
        dataset=Subset(checkpoint.dataset.test_set, range(n)),
        batch_size=n,
        shuffle=False,
        num_workers=os.cpu_count() - 1,
        pin_memory=True)
        
    x, x2 = render(loader_test, checkpoint.model)
    img = convert_to_cv2image(x, x2)
    save_image(checkpoint.options, img, reconstruction_path)
