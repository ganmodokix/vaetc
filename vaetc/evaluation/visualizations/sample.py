import os
import numpy as np
import torch

import cv2

from vaetc.checkpoint import Checkpoint

def visualize(checkpoint: Checkpoint, out_path: str = "samples.png", rows=16, cols=16):

    with torch.no_grad():
        
        if "sample_prior" in checkpoint.model.__dict__:
            z = checkpoint.model.sample_prior(rows * cols)
        else:
            z = torch.randn([rows * cols, checkpoint.model.z_dim]).cuda()
        
        x2 = checkpoint.model.decode(z)

        img = x2.detach().cpu().numpy()
    
    img = np.reshape(img, [rows, cols, *img.shape[1:]])
    img = np.concatenate(img, axis=0)
    img = np.concatenate(img, axis=0)
    img = np.transpose(img, [0, 2, 3, 1])[...,::-1]
    img = (img * 255).astype(np.uint8)

    out_path = os.path.join(checkpoint.options["logger_path"], out_path)
    cv2.imwrite(img, out_path)

