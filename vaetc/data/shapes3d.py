import os

import numpy as np
import h5py

import torch
from torch.utils.data import Dataset

from .utils import cache_path, ImageDataset, one_hot, file_md5, download_file
from vaetc.utils import debug_print

class Shapes3D(Dataset):
    """ Deepmind 3D Shapes dataset loader class
    (https://github.com/deepmind/3d-shapes) """

    def __init__(self, root, download=False) -> None:
        
        self.dataset_path = os.path.join(root, "3dshapes.h5")
        if not os.path.exists(self.dataset_path):
            debug_print(f"File not downloaded yet at {self.dataset_path}")
            if not download:
                raise RuntimeError(f"Dataset not found in {self.dataset_path}")
            self._download()
        else:
            debug_print(f"File already downloaded at {self.dataset_path}")

        self._check_integrity()
        dataset = h5py.File(self.dataset_path, "r")

        debug_print("Loading Images...")
        self.images = dataset["images"][...]
        debug_print("Loading Labels...")
        self.labels = dataset["labels"][...]
        debug_print("Loaded")

        dataset.close()

    def __len__(self) -> int:

        return 480000

    def __getitem__(self, index):
    
        x = self.images[index]
        x = x.transpose(2, 0, 1)
        x = (x / 255).astype(np.float32)

        t = np.array([
            self.labels[index,0],
            self.labels[index,1],
            self.labels[index,2],
            (self.labels[index,3] - 0.75) / (1.25 - 0.75),
            # *one_hot(int(0.5 + self.labels[index,4]), 4),
            self.labels[index, 4] / 3.0,
            self.labels[index,5] / 60 + 0.5,
        ], dtype=np.float32)

        return torch.tensor(x), torch.tensor(t)

    def _download(self):

        if os.path.exists(self.dataset_path):
            raise RuntimeError(f"Already exists: {self.dataset_path}")

        url = "https://storage.googleapis.com/3d-shapes/3dshapes.h5"
        download_file(url, self.dataset_path, make_dirs=True)

    def _check_integrity(self) -> bool:

        if not os.path.exists(self.dataset_path):
            raise RuntimeError(f"File not found in {self.dataset_path}")
        
        md5_true = "099a2078d58cec4daad0702c55d06868"
        md5_downloaded = file_md5(self.dataset_path)
        
        if md5_downloaded != md5_true:
            raise RuntimeError(f"File corrupted: {self.dataset_path}")

        debug_print(f"MD5 check: {md5_downloaded} OK")

def shapes3d():
    """ Shapes 3D dataset [Kim and Mnih, 2018 (http://proceedings.mlr.press/v80/kim18b.html)]

    URL: https://github.com/deepmind/3d-shapes
    
    Returns:
        ImageDataset
    """

    path_to_dataset = cache_path("3dshapes")
    return ImageDataset(Shapes3D(root=path_to_dataset, download=True))