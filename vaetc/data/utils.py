import os
import sys
from typing import Optional, Tuple
import hashlib

import numpy as np
from tqdm import tqdm
import requests

from torch.utils.data import random_split, Dataset, Subset

IMAGE_CHANNELS = 3
IMAGE_HEIGHT = 64
IMAGE_WIDTH = 64
IMAGE_SHAPE = (IMAGE_CHANNELS, IMAGE_WIDTH, IMAGE_HEIGHT)

DEFAULT_VAETC_ROOT = os.path.join(os.path.expanduser("~"), ".vaetc")
VAETC_ROOT = os.getenv("VAETC_PATH") if os.getenv("VAETC_PATH") is not None else DEFAULT_VAETC_ROOT
DATASET_ROOT = os.path.join(VAETC_ROOT, "datasets")

def cache_path(dataset_name: str):
    """ Dataset cache path by dataset name

    Args:
        dataset_name (str): dataset name `celeba` `mnist` etc

    Returns:
        str: Directory path to the cache root
    """

    return os.path.join(DATASET_ROOT, dataset_name)

def one_hot(i: int, num_classes: int):
    """ Create one-hot vector
    
    Args:
        i (int): class id
        num_classes (int): # of classes

    Returns:
        np.ndarray: one-hot vector, shape (num_classes, )
    """

    r = np.zeros(shape=(num_classes, ), dtype=np.float32)
    r[i] = 1
    return r

def file_md5(file_path: str):

    hash = hashlib.md5(usedforsecurity=True)
    with open(file_path, "rb") as fp:
        while True:
            chunk = fp.read(2048 * hash.block_size)
            if len(chunk) == 0: break
            hash.update(chunk)
    
    return hash.hexdigest()

def download_file(url: str, file_path: str, make_dirs: bool = False):

    response = requests.get(url, stream=True)
    if response.status_code != 200:
        raise RuntimeError(f"URL responded {response.status_code}")

    if make_dirs:
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
    
    with open(file_path, "wb") as fp:
        for chunk in tqdm(response.iter_content(chunk_size=4096)):
            if chunk:
                fp.write(chunk)
        fp.flush()

    response.close()

class ImageDataset:
    """ An image dataset wrapper for splitting """

    # a data set of (x, y)
    def __init__(self, training_set: Dataset, test_set: Optional[Dataset] = None, validation_set: Optional[Dataset] = None) -> None:
        """
        Args:
            training_set (:class:`Dataset`): training set
            test_set (:class:`Dataset`, optional): test set if already split
            validation_set (:class:`Dataset`, optional): validation set if already split
        """

        assert training_set is not None

        if test_set is not None and validation_set is not None:

            # use pre-defined splits
            self.training_set   = training_set
            self.validation_set = validation_set
            self.test_set       = test_set

        elif test_set is not None and validation_set is None:

            # split training for valid
            total = len(training_set)
            num_training = int(total * 0.9)
            num_validation = total - num_training

            self.training_set, self.validation_set \
                = random_split(
                    dataset=training_set,
                    lengths=[num_training, num_validation])
            self.test_set = test_set

        elif test_set is None and validation_set is None:

            # split training for valid and test
            total = len(training_set)
            num_training = int(total * 0.8)
            num_validation = int(total * 0.1)
            num_test = total - num_training - num_validation

            self.training_set, self.validation_set, self.test_set = random_split(
                dataset=training_set,
                lengths=[num_training, num_validation, num_test])

        elif test_set is None and validation_set is not None:

            # wrong split
            raise ValueError("valid exists and test does not exist??")

        else:

            # unreachable
            raise ValueError("something went wrong")
