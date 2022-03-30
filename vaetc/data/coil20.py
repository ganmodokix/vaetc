import subprocess
import os
import cv2

import torch
from torch.utils.data import Dataset
import numpy as np

from .utils import IMAGE_HEIGHT, IMAGE_WIDTH, cache_path, ImageDataset, one_hot, file_md5, download_file
from vaetc.utils import debug_print

class COIL20(Dataset):
    """ Columbia University Image Library(COIL-20) class [Nene+, 1996]
    (https://www.cs.columbia.edu/CAVE/software/softlib/coil-20.php)
    
    Returns:
        ImageDataset
    """

    def __init__(self, root, download=False) -> None:
        
        self.root_path = root
        self.data_path = os.path.join(root, "coil-20-proc")
        if not os.path.isdir(self.data_path):
            
            debug_print(f"File not downloaded yet at {self.data_path}")

            if not download:
                raise RuntimeError(f"Dataset not found in {self.data_path}")
            
            self._download()
        
        else:

            debug_print(f"File already downloaded at {self.data_path}")
        
        self.num_categories = 20
        self.num_angles = 72

    def __len__(self) -> int:

        return self.num_categories * self.num_angles

    def __getitem__(self, index):

        if torch.is_tensor(index):
            index = index.item()
            
        index_category = index // self.num_angles + 1
        index_angle = index % self.num_angles

        x: np.ndarray = cv2.imread(os.path.join(self.data_path, f"obj{index_category}__{index_angle}.png"))
        x = cv2.resize(x, [IMAGE_WIDTH, IMAGE_HEIGHT])
        x = x[...,::-1].astype(np.float32) / 255
        x = x.transpose(2, 0, 1)

        t = np.zeros(shape=[self.num_categories + 1, ])
        t[0] = index_angle / self.num_angles
        t[index_category] = 1.

        return torch.tensor(x), torch.tensor(t)

    def _download(self):

        if os.path.exists(self.data_path):
            raise RuntimeError(f"Already exists: {self.data_path}")

        url = "http://www.cs.columbia.edu/CAVE/databases/SLAM_coil-20_coil-100/coil-20/coil-20-proc.zip"
        download_file(url, self.data_path + ".zip", make_dirs=True)

        subprocess.run(["unzip", "coil-20-proc.zip"], cwd=self.root_path)

def coil20(download=True):

    path_to_dataset = cache_path("coil20")
    return ImageDataset(COIL20(root=path_to_dataset, download=download))