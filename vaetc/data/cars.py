import os
import sys
import subprocess
from typing import Literal

from tqdm import tqdm
import numpy as np
from scipy.io import loadmat
import cv2

import torch
from torch.nn import functional as F
from torch.utils.data import Dataset
from torchvision import datasets, transforms

from vaetc.utils.debug import debug_print

from .utils import ImageDataset, cache_path, IMAGE_HEIGHT, IMAGE_WIDTH, one_hot

NUM_CLASSES = 196

def download_if_not_exist(url: str, cwd: str):

    name = url.split("/")[-1]

    if not os.path.exists(os.path.join(cwd, name)):

        debug_print(f"Downloading {name} from {url} ...")
        subprocess.run(["wget", url], cwd=cwd)

    else:

        debug_print(f"{name} already downloaded; skipped")

def extract_tgz(src_path, dst_path):

    # subprocess.run(["pv", src_path, "|", "tar", "-xzvf", "-", ">", "/dev/null"], cwd=dst_path)
    pipe1 = subprocess.Popen(["pv", src_path], stdout=subprocess.PIPE, cwd=dst_path)
    pipe2 = subprocess.Popen(["tar", "-xzvf", "-"], stdin=pipe1.stdout, stdout=subprocess.DEVNULL, cwd=dst_path)
    pipe1.stdout.close()
    pipe2.communicate()

def download_cars(root_path: str):

    download_path = os.path.join(root_path, "download")
    os.makedirs(download_path, exist_ok=True)

    # downloadation

    download_if_not_exist(
        url="http://ai.stanford.edu/~jkrause/car196/cars_annos.mat",
        cwd=download_path,
    )
    download_if_not_exist(
        url="http://ai.stanford.edu/~jkrause/car196/car_ims.tgz",
        cwd=download_path,
    )

def extract_cars(root_path: str):
    
    if not os.path.exists(os.path.join(root_path, "car_ims")):
        extract_tgz(os.path.join(root_path, "download", "car_ims.tgz"), root_path)
    else:
        debug_print("car_ims.tgz already extracted to car_ims/")

def square_crop(img: np.ndarray, x: int, y: int, w: int, h: int):

    imgh, imgw, imgc = img.shape

    if w < h:
        x += w // 2 - h // 2
        w = h
    else:
        y += h // 2 - w // 2
        h = w

    margin = max(0, -x, -y, x + w - imgw, y + h - imgh)
    padded = cv2.copyMakeBorder(img, margin, margin, margin, margin, cv2.BORDER_REPLICATE)

    x += margin
    y += margin
    cropped = padded[y:y+h,x:x+w]

    return cropped

def prepare_cars(root_path: str):

    download_cars(root_path)
    extract_cars(root_path)

class CarsDataset(Dataset):

    def __init__(self, root_path: str, split: Literal["train","test"], download=False) -> None:
        super().__init__()

        self.root_path = str(root_path)

        if download:
            prepare_cars(self.root_path)
        
        annos_path = os.path.join(root_path, "download", "cars_annos.mat")
        annos = loadmat(annos_path, mat_dtype=True)
        self.class_names = annos["class_names"][0]
        annotations = annos["annotations"][0]
        
        annotations_interpreted = []
        for record in annotations:

            relative_im_path = str(record[0][0])
            bbox_x1 = int(record[1][0][0])
            bbox_y1 = int(record[2][0][0])
            bbox_x2 = int(record[3][0][0])
            bbox_y2 = int(record[4][0][0])
            class_id = int(record[5][0][0]) - 1
            split = "test" if record[6] else "train"

            annotations_interpreted.append({
                "file_path": relative_im_path,
                "bbox": [bbox_x1, bbox_y1, bbox_x2 - bbox_x1 + 1, bbox_y2 - bbox_y1 + 1],
                "class_id": class_id,
                "split_name": split
            })

        self.records = [
            record for record in annotations_interpreted
            if record["split_name"] == split
        ]

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:

        record = self.records[index]

        file_path = os.path.join(self.root_path, record["file_path"])
        class_id = record["class_id"]
        bbox_x, bbox_y, bbox_w, bbox_h = record["bbox"]

        x = cv2.imread(file_path)
        x = square_crop(x, bbox_x, bbox_y, bbox_w, bbox_h)
        x = cv2.resize(x, [IMAGE_WIDTH, IMAGE_HEIGHT], interpolation=cv2.INTER_LINEAR)
        x = x[...,::-1] # bgr -> rgb
        x = x.transpose(2, 0, 1)
        x = x.astype(np.float32) / 255
        t = one_hot(class_id, num_classes=NUM_CLASSES)

        return torch.tensor(x), torch.tensor(t)

    def __len__(self):

        return len(self.records)


def cars(download=True):
    """ Stanford Cars Dataset
    (http://ai.stanford.edu/~jkrause/cars/car_dataset.html)
    
    Returns:
        ImageDataset
    """

    path_to_dataset = cache_path("cars")

    train_set = CarsDataset(path_to_dataset, split="train", download=download)
    test_set = CarsDataset(path_to_dataset, split="test", download=download)

    return ImageDataset(training_set=train_set, test_set=test_set)
