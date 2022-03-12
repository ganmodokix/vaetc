import sys
import os

import numpy as np
import cv2
import torch

sys.path.append(os.path.dirname(__file__) + '/../')
from vaetc.data import smallnorb

if __name__ == "__main__":

    # ds = dsprites().training_set
    ds = smallnorb().training_set

    idx = torch.randint(len(ds), size=(1, )).item()
    x, t = ds[idx]

    print(f"min x: {x.min().item()}")
    print(f"max x: {x.max().item()}")
    print(f"avg x: {x.mean().item()}")
    print(f"std x: {x.std().item()}")
    print(f"t: {t.detach().cpu().numpy()}")

    path_to_sandbox = os.path.join(os.path.dirname(os.path.dirname(__file__)), "sandbox")
    os.makedirs(path_to_sandbox, exist_ok=True)
    path_to_save = os.path.join(path_to_sandbox, "smallnorb.png")
    img = x.detach().cpu().numpy()
    img = img.transpose(1, 2, 0) # (H*2, W*N, C)
    img = (img[...,::-1] * 255).astype(np.uint8) # RGB -> BGR, float->uint8
    cv2.imwrite(path_to_save, img)