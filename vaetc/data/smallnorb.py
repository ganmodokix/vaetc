import os
import gzip

import torch
from torch.utils.data import Dataset
import numpy as np
import cv2

from .utils import IMAGE_CHANNELS, IMAGE_HEIGHT, IMAGE_WIDTH, cache_path, ImageDataset, one_hot, file_md5, download_file
from vaetc.utils import debug_print

from struct import unpack

def read_binary_matrix(file_path: str) -> np.ndarray:

    SINGLE_PRECISION_MATRIX = 0x1E3D4C51
    PACKED_MATRIX           = 0x1E3D4C52
    DOUBLE_PRECISION_MATRIX = 0x1E3D4C53
    INTEGER_MATRIX          = 0x1E3D4C54
    BYTE_MATRIX             = 0x1E3D4C55
    SHORT_MATRIX            = 0x1E3D4C56

    with open(file_path, "rb") as fp:

        magic = unpack("<i", fp.read(4))[0]
        ndim = unpack("<i", fp.read(4))[0]
        dim = tuple(unpack("<i", fp.read(4))[0] for _ in range(max(3, ndim)))[:ndim]

        debug_print(ndim, dim)

        if magic == INTEGER_MATRIX:
            
            data = np.frombuffer(fp.read(), dtype=np.dtype(np.int32).newbyteorder("<")).reshape(dim)

        elif magic == BYTE_MATRIX:

            data = np.frombuffer(fp.read(), dtype=np.uint8).reshape(dim)

        else:
            raise NotImplementedError(f"magic 0x{magic:X}")

    return data

class SmallNORB(Dataset):
    """ THE small NORB DATASET, V1.0 [huang and LeCun, 2005]
    (https://cs.nyu.edu/~ylclab/data/norb-v1.0-small/)
    
    Returns:
        ImageDataset
    """

    def __init__(self, root, download=False, split="train") -> None:
        
        self.root = root
        if self._file_exists():
        
            debug_print(f"File already downloaded at {self.root}")
            
        else:
            
            debug_print(f"File not downloaded yet at {self.root}")

            if not download:
                raise RuntimeError(f"Dataset not found in {self.root}")
            
            self._download()

        self._check_integrity()
        
        if split == "train":
            self.dat = read_binary_matrix(os.path.join(self.root, "smallnorb-5x46789x9x18x6x2x96x96-training-dat.mat"))
            self.cat = read_binary_matrix(os.path.join(self.root, "smallnorb-5x46789x9x18x6x2x96x96-training-cat.mat"))
            self.info = read_binary_matrix(os.path.join(self.root, "smallnorb-5x46789x9x18x6x2x96x96-training-info.mat"))
        elif split == "test":
            self.dat = read_binary_matrix(os.path.join(self.root, "smallnorb-5x01235x9x18x6x2x96x96-testing-dat.mat"))
            self.cat = read_binary_matrix(os.path.join(self.root, "smallnorb-5x01235x9x18x6x2x96x96-testing-cat.mat"))
            self.info = read_binary_matrix(os.path.join(self.root, "smallnorb-5x01235x9x18x6x2x96x96-testing-info.mat"))
        else:
            raise RuntimeError(f"Invalid split {split}")

    def __len__(self) -> int:

        return self.dat.shape[0]

    def __getitem__(self, index):

        if torch.is_tensor(index):
            index = index.item()

        x = self.dat[index].astype(np.float32) / 255
        x = x[0] # use only the first camera
        x = cv2.resize(x, (IMAGE_WIDTH, IMAGE_HEIGHT), interpolation=cv2.INTER_LINEAR)
        x = np.tile(x[None,...], [IMAGE_CHANNELS, 1, 1])

        cat = one_hot(self.cat[index], num_classes=5)
        info = np.array([
            self.info[index,1].astype(float) / 8, # elevation
            self.info[index,2].astype(float) / 34, # azimuth
            self.info[index,3].astype(float) / 5, # lighting condition
        ])
        t = np.concatenate([cat, info], axis=0).astype(np.float32)

        return torch.tensor(x), torch.tensor(t)

    def _file_exists(self):

        files = (
            "smallnorb-5x01235x9x18x6x2x96x96-testing-cat.mat",
            "smallnorb-5x01235x9x18x6x2x96x96-testing-dat.mat",
            "smallnorb-5x01235x9x18x6x2x96x96-testing-info.mat",
            "smallnorb-5x46789x9x18x6x2x96x96-training-cat.mat",
            "smallnorb-5x46789x9x18x6x2x96x96-training-dat.mat",
            "smallnorb-5x46789x9x18x6x2x96x96-training-info.mat",
        )

        for name in files:
            if not os.path.isfile(os.path.join(self.root, name)):
                return False
        
        return True

    def _download(self):

        if os.path.exists(self.root):
            raise RuntimeError(f"Already exists: {self.root}")

        urls = (
            "https://cs.nyu.edu/~ylclab/data/norb-v1.0-small/smallnorb-5x01235x9x18x6x2x96x96-testing-cat.mat.gz",
            "https://cs.nyu.edu/~ylclab/data/norb-v1.0-small/smallnorb-5x01235x9x18x6x2x96x96-testing-dat.mat.gz",
            "https://cs.nyu.edu/~ylclab/data/norb-v1.0-small/smallnorb-5x01235x9x18x6x2x96x96-testing-info.mat.gz",
            "https://cs.nyu.edu/~ylclab/data/norb-v1.0-small/smallnorb-5x46789x9x18x6x2x96x96-training-cat.mat.gz",
            "https://cs.nyu.edu/~ylclab/data/norb-v1.0-small/smallnorb-5x46789x9x18x6x2x96x96-training-dat.mat.gz",
            "https://cs.nyu.edu/~ylclab/data/norb-v1.0-small/smallnorb-5x46789x9x18x6x2x96x96-training-info.mat.gz",
        )

        for url in urls:

            name = url.split("/")[-1]
            gz_path = os.path.join(self.root, name)
            mat_path = gz_path[:-3]
            download_file(url, gz_path, make_dirs=True)
            
            debug_print(f"decompressing {gz_path} ...")
            with gzip.open(gz_path, "rb") as fp_gz:
                with open(mat_path, "wb") as fp_mat:
                    while True:
                        chunk = fp_gz.read(4096)
                        if len(chunk) == 0: break
                        fp_mat.write(chunk)
                        fp_mat.flush()

    def _check_integrity(self) -> bool:

        if not self._file_exists():
            raise RuntimeError(f"File not found in {self.root}")

def smallnorb(download=True):

    path_to_dataset = cache_path("smallnorb")

    training_set = SmallNORB(root=path_to_dataset, download=download, split="train")
    test_set = SmallNORB(root=path_to_dataset, download=download, split="test")

    return ImageDataset(training_set, test_set)