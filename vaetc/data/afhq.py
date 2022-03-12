import os
from typing import Literal

import numpy as np
from tqdm import tqdm
import torch
from torch.utils.data import Dataset
import cv2

from vaetc.utils.debug import debug_print

from .utils import ImageDataset, cache_path, IMAGE_CHANNELS, IMAGE_HEIGHT, IMAGE_WIDTH, IMAGE_SHAPE

class AFHQv2(Dataset):
    """ AFHQ-v2 Dataset (https://github.com/clovaai/stargan-v2) """

    def __init__(self,
        root_path: str,
        set_name: Literal["dog", "cat", "wild"],
        split: Literal["train", "test"],
        download=False
    ) -> None:
    
        super().__init__()

        self.root_path = root_path
        self.set_name = set_name
        self.split = split
        assert self.set_name in ["dog", "cat", "wild"]
        assert self.split in ["train", "test"]
        
        if download and self._download_required():
            self._download()

        debug_print("Scanning directories...")
        self.image_paths = []
        with os.scandir(os.path.join(self.root_path, self.split, self.set_name)) as scd:
            entries = [x.path for x in scd]
        for entry in tqdm(entries):
            self.image_paths.append(entry)
        self.image_paths.sort()

        debug_print("Pre-processing...")
        self.cache = np.empty(shape=(len(self.image_paths), *IMAGE_SHAPE), dtype=np.float32)
        for i, image_path in enumerate(tqdm(self.image_paths)):
            self.cache[i] = self._load_image(image_path)

    def _download_required(self) -> bool:

        return False

    def _download(self):

        print("This dataset (FFHQ Thumbnail) must be downloaded and placed manually as:")
        print(f"1. Download afhq-v2.zip from https://github.com/clovaai/stargan-v2 and place it under {self.root_path}/")
        print("2. Extract it with `7z x afhq-v2.zip` instead of `unzip` because afhq-v2.zip has a corrupted header")

        raise NotImplementedError("This dataset is currently available by manual download")

    def __len__(self) -> int:
        
        return len(self.image_paths)

    def _load_image(self, image_path: str) -> np.ndarray:
        
        img = cv2.imread(image_path)
        img = cv2.resize(img, [IMAGE_WIDTH, IMAGE_HEIGHT], interpolation=cv2.INTER_LANCZOS4)
        img = img[...,::-1] # BGR -> RGB
        img = img.transpose(2, 0, 1)
        img = img.astype(np.float32) / 255

        return img

    def __getitem__(self, index):

        return torch.tensor(self.cache[index]), torch.empty(size=(0, ), dtype=torch.float)

def afhq_v2_dog(download=True):

    root_path = cache_path("afhq-v2")

    set_name = "dog"
    ds_train = AFHQv2(root_path=root_path, download=download, set_name=set_name, split="train")
    ds_test  = AFHQv2(root_path=root_path, download=download, set_name=set_name, split="test")

    return ImageDataset(ds_train, ds_test)

def afhq_v2_cat(download=True):

    root_path = cache_path("afhq-v2")

    set_name = "cat"
    ds_train = AFHQv2(root_path=root_path, download=download, set_name=set_name, split="train")
    ds_test  = AFHQv2(root_path=root_path, download=download, set_name=set_name, split="test")

    return ImageDataset(ds_train, ds_test)

def afhq_v2_wild(download=True):

    root_path = cache_path("afhq-v2")

    set_name = "wild"
    ds_train = AFHQv2(root_path=root_path, download=download, set_name=set_name, split="train")
    ds_test  = AFHQv2(root_path=root_path, download=download, set_name=set_name, split="test")

    return ImageDataset(ds_train, ds_test)