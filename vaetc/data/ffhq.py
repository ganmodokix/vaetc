import os
import sys
import subprocess

import numpy as np
from tqdm import tqdm
import cv2
import torch
import orjson
from torch.utils.data import Dataset

from vaetc.utils.debug import debug_print

from .utils import IMAGE_HEIGHT, IMAGE_WIDTH, ImageDataset, cache_path

class FFHQThumbnails(Dataset):
    """ FFHQ-Dataset (https://github.com/NVlabs/ffhq-dataset)
    Because we resize images small, we use 128x128 thumbnails
    """

    def __init__(self, root_path: str, download=False, train=True) -> None:
        super().__init__()

        self.root_path = root_path
        self.category = "training" if train else "validation"
        
        if download and self._download_required():
            self._download()
        
        with open(os.path.join(self.root_path, "ffhq-dataset-v2.json"), "r", encoding="utf8") as fp:
            
            debug_print("Loading metadata...")
            json_str = ""
            with tqdm() as pbar:
                while True:
                    chunk = fp.read(4096)
                    if len(chunk) == 0:
                        break
                    json_str += chunk
                    pbar.update(1)

            debug_print("Parsing metadata...")
            self.metadata = orjson.loads(json_str)

        debug_print("Splitting ...")
        self.keys = []
        for index in tqdm(range(70000)):

            metadata_instance = self.metadata[str(index)]

            if metadata_instance["category"] == self.category:
                if os.path.isfile(os.path.join(self.root_path, metadata_instance["thumbnail"]["file_path"])):
                    self.keys.append(index)

    def _download_required(self) -> bool:

        if not os.path.isfile(os.path.join(self.root_path, "ffhq-dataset-v2.json")):
            return True

        if not os.path.isdir(os.path.join(self.root_path, "thumbnails128x128")):
            return True

        return False

    def _download(self):

        print("This dataset (FFHQ Thumbnail) must be downloaded and placed manually as:")
        print("1. Download ffhq-dataset-v2.json from https://github.com/NVlabs/ffhq-dataset#overview")
        print("2. Download thumbnails128x128 from https://github.com/NVlabs/ffhq-dataset#overview")
        print(f"3. Put them in {self.root_path}/")

        raise NotImplementedError("This dataset is currently available by manual download")

    def __len__(self) -> int:
        
        return len(self.keys)

    def _load_image(self, image_path: str) -> np.ndarray:
        
        img = cv2.imread(image_path)
        img = cv2.resize(img, [IMAGE_WIDTH, IMAGE_HEIGHT], interpolation=cv2.INTER_LANCZOS4)
        img = img[...,::-1] # BGR -> RGB
        img = img.transpose(2, 0, 1)
        img = img.astype(np.float32) / 255

        return img

    def __getitem__(self, index):

        index_entire = self.keys[index]
        metadata_instance = self.metadata[str(index_entire)]
        
        image_path = os.path.join(
            self.root_path,
            metadata_instance["thumbnail"]["file_path"]
        )

        img = self._load_image(image_path)

        t = np.array(metadata_instance["image"]["face_landmarks"], dtype=np.float32) # (M, 2)
        t[:,0] /= metadata_instance["image"]["pixel_size"][0]
        t[:,1] /= metadata_instance["image"]["pixel_size"][1]

        return torch.tensor(img), torch.tensor(t).view(-1)

def ffhq(download=True):

    root_path = cache_path("ffhq")

    train_set = FFHQThumbnails(root_path=root_path, download=download, train=True)
    test_set  = FFHQThumbnails(root_path=root_path, download=download, train=False)

    return ImageDataset(train_set, test_set)