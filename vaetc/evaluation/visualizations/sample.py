import os
import numpy as np
import torch

import cv2

from vaetc.checkpoint import Checkpoint
from vaetc.models.abstract import GaussianEncoderAutoEncoderRLModel
from vaetc.utils.debug import debug_print

def visualize(checkpoint: Checkpoint, out_path: str = "samples.png", rows=16, cols=16):

    if not hasattr(checkpoint.model, "sample_prior"):
        debug_print("Skipped; no prior")
        return

    with torch.no_grad():

        if isinstance(checkpoint.model, GaussianEncoderAutoEncoderRLModel):
            z = checkpoint.model.sample_prior(rows * cols)
            print(z)
        else:
            z = torch.randn([rows * cols, checkpoint.model.z_dim]).cuda()
        
        x2 = checkpoint.model.decode(z)

        img = x2.detach().cpu().numpy()
    
    img = np.transpose(img, [0, 2, 3, 1])[...,::-1]
    img = np.reshape(img, [rows, cols, *img.shape[1:]])
    img = np.concatenate(img, axis=1)
    img = np.concatenate(img, axis=1)
    img = (img * 255).astype(np.uint8)

    out_path = os.path.join(checkpoint.options["logger_path"], out_path)
    cv2.imwrite(out_path, img)

