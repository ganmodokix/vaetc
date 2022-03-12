import os

import torch
from torch.utils.data import Dataset
import numpy as np

from .utils import cache_path, ImageDataset, one_hot, file_md5, download_file
from vaetc.utils import debug_print

class DSprites(Dataset):
    """ Dsprites Dataset class [Metthey+, 2017]
    (https://github.com/deepmind/dsprites-dataset)
    
    Returns:
        ImageDataset
    """

    def __init__(self, root, download=False) -> None:
        
        self.data_path = os.path.join(root, "dsprites_ndarray_co1sh3sc6or40x32y32_64x64.npz")
        if not os.path.exists(self.data_path):
            
            debug_print(f"File not downloaded yet at {self.data_path}")

            if not download:
                raise RuntimeError(f"Dataset not found in {self.data_path}")
            
            self._download()
        
        else:

            debug_print(f"File already downloaded at {self.data_path}")

        self._check_integrity()
        self.data = np.load(self.data_path)
        self.nums_categories = self.data["latents_classes"].max(axis=0) + 1

        self.images = self.data["imgs"]
        self.latents_classes = self.data["latents_classes"]
        self.latents_values = self.data["latents_values"]

    def __len__(self) -> int:

        return 737280

    def __getitem__(self, index):

        if torch.is_tensor(index):
            index = index.item()

        x = self.images[index]
        x = np.tile(x[None,:,:], (3, 1, 1)).astype(np.float32)

        t = []
        for i in range(3):
            t.append(one_hot(
                self.latents_classes[index,i],
                self.nums_categories[i]))
        t_ori = (self.latents_values[index,3:4] / np.pi / 2).astype(np.float32)
        t.append(t_ori)
        t_pos = self.latents_values[index,4:].astype(np.float32)
        t.append(t_pos)

        t = np.concatenate(t)

        return torch.tensor(x), torch.tensor(t)

    def _download(self):

        if os.path.exists(self.data_path):
            raise RuntimeError(f"Already exists: {self.data_path}")

        url = "https://github.com/deepmind/dsprites-dataset/blob/master/dsprites_ndarray_co1sh3sc6or40x32y32_64x64.npz?raw=true"
        download_file(url, self.data_path, make_dirs=True)

    def _check_integrity(self) -> bool:

        if not os.path.exists(self.data_path):
            raise RuntimeError(f"File not found in {self.data_path}")
        
        md5_true = "7da33b31b13a06f4b04a70402ce90c2e"
        md5_downloaded = file_md5(self.data_path)
        
        if md5_downloaded != md5_true:
            raise RuntimeError(f"File corrupted: {self.data_path}")

        debug_print(f"MD5 check: {md5_downloaded} OK")

def dsprites(download=True):

    path_to_dataset = cache_path("dsprites")
    return ImageDataset(DSprites(root=path_to_dataset, download=download))