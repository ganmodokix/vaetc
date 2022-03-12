import torch
from torch.utils.data import Dataset
import os

import cv2
import numpy as np

from .utils import cache_path, ImageDataset, one_hot, download_file, file_md5

from torch.nn import functional as F
from torchvision import transforms, datasets

from vaetc.utils import debug_print


def create_target(
    object_color: int,
    object_shape: int,
    object_size: int,
    camera_height: int,
    background_color: int,
    horizontal_axis: int,
    vertical_axis: int,
):
    return np.concatenate([
        one_hot(object_color, 6),
        one_hot(object_shape, 6),
        one_hot(object_size, 6),
        one_hot(camera_height, 6),
        one_hot(background_color, 6),
        np.array([horizontal_axis / 39]),
        np.array([vertical_axis / 39]),
    ], axis=0)

class MPI3D(Dataset):
    """ MPI3D Dataset class
    [Gondal+, 2019 (https://openreview.net/forum?id=Hke7CHHxUB)] """

    def __init__(self, set_name = "real", root_dir = cache_path("mpi3d"), download=False) -> None:
        super().__init__()
        
        self.set_name = str(set_name)
        self.root_dir = str(root_dir)

        self.root_path = os.path.join(self.root_dir, f"mpi3d_{self.set_name}")

        self.image_path = os.path.join(self.root_dir, f"mpi3d_{self.set_name}.npz")
        self.images = np.load(self.image_path)["images"]

        if not os.path.exists(self.image_path):
            
            debug_print(f"File not downloaded yet at {self.image_path}")

            if not download:
                raise RuntimeError(f"Dataset not found in {self.image_path}")
            
            self._download()
        
        else:

            debug_print(f"File already downloaded at {self.image_path}")

        self._check_integrity()

    @staticmethod
    def factor_shape():
        return (6, 6, 2, 3, 3, 40, 40)

    @staticmethod
    def factor_indices(index: int):

        indices = []
        i = index
        for f in MPI3D.factor_shape()[::-1]:
            indices.append(i % f)
            i //= f
        
        return indices[::-1]

    def __len__(self):
        return int(np.prod(MPI3D.factor_shape()))

    def __getitem__(self, index):
        
        if torch.is_tensor(index):
            index = index.item()

        # x_path = os.path.join(self.root_path, f"images", f"{index:07d}.jpg")
        # t_path = os.path.join(self.root_path, f"targets", f"{index:07d}.npz")

        # x = cv2.imread(x_path)[:,:,::-1]
        # t = np.load(t_path)["target"]

        x = self.images[index]
        fc = MPI3D.factor_indices(index)
        t = create_target(*fc)

        # transform
        x = x.astype(np.float32) / 255 # [0,255] -> [0, 1]
        x = x.transpose(2, 0, 1) # HWC -> CHW
        t = t.astype(np.float32)

        return torch.tensor(x), torch.tensor(t)

    def _download(self):

        if os.path.exists(self.image_path):
            raise RuntimeError(f"Already exists: {self.image_path}")

        url = f"https://storage.googleapis.com/disentanglement_dataset/Final_Dataset/mpi3d_{self.set_name}.npz"
        download_file(url, self.image_path, make_dirs=True)

    def _check_integrity(self) -> bool:

        if not os.path.exists(self.image_path):
            raise RuntimeError(f"File not found in {self.image_path}")
        
        # md5_true = {
        #    "toy": "",
        #    "realistic": "",
        #    "real": "",
        #    "real_complex": "",
        # }[self.set_name]
        # md5_downloaded = file_md5(self.image_path)
        
        # if md5_downloaded != md5_true:
        #     raise RuntimeError(f"File corrupted: {self.image_path}")

        # debug_print(f"MD5 check: {md5_downloaded} OK")

def mpi3d_real():
    """ MPI3D-real
    [Gondal+, 2019 (https://openreview.net/forum?id=Hke7CHHxUB)]
    
    Returns:
        ImageDataset
    """

    path_to_dataset = cache_path("mpi3d")

    return ImageDataset(MPI3D("real", root_dir=path_to_dataset))