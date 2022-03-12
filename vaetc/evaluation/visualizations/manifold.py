import os

import numpy as np
import cv2

import torch
from torch.utils.data import DataLoader, Subset

from vaetc.checkpoint import Checkpoint

def decode_batch(model, z: torch.Tensor, batch_size: int = 16) -> torch.Tensor:

    data_size = z.shape[0]

    results = []
    
    for i in range((data_size + batch_size - 1) // batch_size):
        
        ib = batch_size * i
        ie = ib + batch_size

        z_batch = z[ib:ie]

        results += [model.decode(z_batch)]

    return torch.cat(results, dim=0)

def render(checkpoint: Checkpoint, n: int = 10):
    
    loader = DataLoader(
        dataset=Subset(checkpoint.dataset.test_set, range(4)),
        batch_size=4,
        shuffle=False,
        num_workers=os.cpu_count() - 1)

    checkpoint.model.eval()

    with torch.no_grad():

        x, t = loader.__iter__().next()
        x = x.to(checkpoint.model.device)
        z = checkpoint.model.encode(x).detach()
        z_dim = z.shape[1]

        u = torch.linspace(0., 1., steps=n).to(checkpoint.model.device)
        u, v = torch.meshgrid(u, u, indexing="ij")

        z_grid = z[None,None,0,:] * (1 - u[:,:,None]) * (1 - v[:,:,None]) \
               + z[None,None,1,:] * (1 - u[:,:,None]) * v[:,:,None] \
               + z[None,None,2,:] * u[:,:,None] * (1 - v[:,:,None]) \
               + z[None,None,3,:] * u[:,:,None] * v[:,:,None]

        z_seq = z_grid.view(n*n, z_dim)
        x2_seq = decode_batch(checkpoint.model, z_seq).detach()
        x2_grid = x2_seq.view(n, n, x.shape[1], x.shape[2], x.shape[3])
        x2_grid[ 0, 0,...] = x[0]
        x2_grid[ 0,-1,...] = x[1]
        x2_grid[-1, 0,...] = x[2]
        x2_grid[-1,-1,...] = x[3]

        img = torch.cat(torch.cat(x2_grid.unbind(), dim=-2).unbind(), dim=-1)
        img = img.cpu().numpy()

    return img

def convert_to_cv2image(img):

    img = img.transpose(1, 2, 0) # (H*2, W*N, C)
    img = (img[...,::-1] * 255).astype(np.uint8) # RGB -> BGR, float->uint8

    return img

def save_image(options, img):

    logger_path = options["logger_path"]
    file_name = "manifold.png"

    path_to_save = os.path.join(logger_path, file_name)

    cv2.imwrite(path_to_save, img)


def visualize(checkpoint: Checkpoint):

    img = render(checkpoint)
    img = convert_to_cv2image(img)
    save_image(checkpoint.options, img)
