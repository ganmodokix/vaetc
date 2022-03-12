import os
import subprocess

import numpy as np
import torch
from torch.utils.data import Dataset
from tqdm import tqdm

from .utils import ImageDataset, cache_path
from vaetc.utils import debug_print

TEAPOT_URL = "https://www.dropbox.com/s/woeyomxuylqu7tx/edinburgh_teapots.zip"
TEAPOT_NAME = TEAPOT_URL.split("/")[-1]

def download_teapot(root_path: str):

    os.makedirs(os.path.join(root_path), exist_ok=True)

    zip_path = os.path.join(root_path, TEAPOT_NAME)
    teapot_path = os.path.join(root_path, "teapots.npz")
    images_path = os.path.join(root_path, "images.npy")
    gts_path = os.path.join(root_path, "gts.npy")

    if os.path.exists(images_path) and os.path.exists(gts_path):
        debug_print("downloading process already completed")
        return

    # download
    if not os.path.exists(zip_path):
        subprocess.run(["wget", TEAPOT_URL], cwd=root_path)
    else:
        debug_print(f"{TEAPOT_NAME} already downloaded; skipped")

    # extract
    if not os.path.exists(teapot_path):
        pipe1 = subprocess.Popen(["pv", zip_path], stdout=subprocess.PIPE, cwd=root_path)
        pipe2 = subprocess.Popen(["busybox", "unzip", "-"], stdin=pipe1.stdout, stdout=subprocess.DEVNULL, cwd=root_path)
        pipe1.stdout.close()
        pipe2.communicate()
    else:
        debug_print(f"{TEAPOT_NAME} already unzipped")
    
    # extract npz
    if not (os.path.exists(images_path) and os.path.exists(gts_path)):
        debug_print("unzipping teapots.npz ...")
        subprocess.run(["unzip", teapot_path], cwd=root_path)
        subprocess.run(["rm", teapot_path])
    else:
        debug_print("teapots.npz already unzipped")


LOAD_TEAPOT_CACHE = None
def load_teapot(root_path):

    global LOAD_TEAPOT_CACHE

    images_path = os.path.join(root_path, "images.npy")
    gts_path = os.path.join(root_path, "gts.npy")

    if LOAD_TEAPOT_CACHE is None:

        try:
            images_mmap = np.load(images_path, mmap_mode="r")
            images = np.empty_like(images_mmap)
            images_size = len(images)
            chunk_size = 1024
            for i in tqdm(range((images_size + chunk_size - 1) // chunk_size)):
                idx_begin = i * chunk_size
                idx_end = (i + 1) * chunk_size
                images[idx_begin:idx_end] = images_mmap[idx_begin:idx_end]
        finally:
            del images_mmap

        LOAD_TEAPOT_CACHE = {
            "images": images,
            "gts": np.load(gts_path),
        }

    return LOAD_TEAPOT_CACHE

class TeapotDataset():
    """ Teapot Dataset by [Eastwood and Williams, ICLR 2018]
    (https://github.com/cianeastwood/qedr)
    """

    def __init__(self, root_path: str, download=False) -> None:
        
        self.root_path = str(root_path)
        if download:
            download_teapot(self.root_path)

        debug_print("Warning: teapot dataset requires >10GB RAM")

        self.data = load_teapot(self.root_path)

        debug_print(self.data["images"].shape, self.data["images"].dtype)
        debug_print(self.data["gts"].shape, self.data["gts"].dtype)

    def __len__(self):

        return self.data["gts"].shape[0]

    def __getitem__(self, index: int):

        x = self.data["images"][index]
        x = x.transpose(2, 1, 0)
        x = x.astype(np.float32) / 255

        t = self.data["gts"][index]
        t[0] /= np.pi * 2
        t[1] /= np.pi / 2

        return torch.tensor(x), torch.tensor(t)

def teapot(download=True):

    root_path = cache_path("teapot")

    dataset = TeapotDataset(root_path, download=download)
    return ImageDataset(dataset)