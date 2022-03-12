from math import hypot
import os
from multiprocessing import Pool, cpu_count

import numpy as np
from tqdm import tqdm
import cv2
import torch
from torch.utils.data import Dataset

from vaetc.utils.debug import debug_print

from .utils import IMAGE_HEIGHT, IMAGE_WIDTH, ImageDataset, cache_path

class Danbooru2019Portraits(Dataset):
    """ Danbooru2019 Portraits
    (https://www.gwern.net/Crops#danbooru2019-portraits)

    n = 302,652

    See also: https://www.kaggle.com/subinium/highresolution-anime-face-dataset-512x512
    """

    def __init__(self, root_path: str, download=False) -> None:
        super().__init__()

        self.root_path = root_path
        
        if download and self._download_required():
            self._download()

        if self._proprocessing_required():
            self._preprocess()

        debug_print("Scanning portraits dir ...")
        self.image_paths = [
            entry.path
            for entry in os.scandir(os.path.join(self.root_path, "portraits_cropped"))
            if entry.is_file()
        ]
        self.image_paths.sort()


    def _download_required(self) -> bool:

        if not os.path.isdir(os.path.join(self.root_path, "portraits")):
            return True

        if not os.path.isfile(os.path.join(self.root_path, "lbpcascade_animeface.xml")):
            return True

        return False

    def _proprocessing_required(self) -> bool:

        if not os.path.isdir(os.path.join(self.root_path, "portraits_cropped")):
            return True
        
        n_raw = len(os.listdir(os.path.join(self.root_path, "portraits")))
        n_cropped = len(os.listdir(os.path.join(self.root_path, "portraits_cropped")))
        if n_raw != n_cropped:
            return True
        
        return False

    def _download(self):

        print("This dataset (Danbooru2019 Portraits) must be downloaded and placed manually as:")
        print("1. Download the entire dataset from https://www.kaggle.com/subinium/highresolution-anime-face-dataset-512x512; connections are refused by the original source via rsync")
        print(f"2. Extract it in {self.root_path}")
        print(f"3. Download lbpcascade_animeface.xml in {self.root_path}")

        raise NotImplementedError("This dataset is currently available by manual download")

    def _proprocess_image(self, arg):

        img_name, img_path = arg
            
        img = cv2.imread(img_path)
        imgh, imgw, imgc = img.shape

        face_detector = cv2.CascadeClassifier(os.path.join(self.root_path, "lbpcascade_animeface.xml"))

        # face detecting
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        gray = cv2.equalizeHist(gray)
        faces = face_detector.detectMultiScale(gray, minSize=(64, 64))
        
        # face cropping
        centers = [(x + w // 2, y + h // 2) for x, y, w, h in faces]
        deviations = [(xc - imgw // 2, yc - imgh // 2) for xc, yc in centers]
        deviations_l2 = [hypot(xd, yd) for xd, yd in deviations]
        if len(deviations_l2) > 0:
            most_center = np.argmin(deviations_l2)
            x, y, w, h = faces[most_center]
            img = img[y:y+h,x:x+w]

        # Resizing to the specified size
        img = cv2.resize(img, [IMAGE_WIDTH, IMAGE_HEIGHT], interpolation=cv2.INTER_LANCZOS4)

        # save the face
        cv2.imwrite(os.path.join(self.root_path, "portraits_cropped", img_name), img)

    def _preprocess(self):

        with os.scandir(os.path.join(self.root_path, "portraits")) as scd:
            img_list = [(entry.name, entry.path) for entry in scd]

        os.makedirs(os.path.join(self.root_path, "portraits_cropped"), exist_ok=True)

        debug_print("Preprocessing ...")
        with Pool() as p:
            iter = p.imap(self._proprocess_image, img_list)
            results = list(tqdm(iter, total=len(img_list)))

    def __len__(self) -> int:
        
        return len(self.image_paths)

    def _load_image(self, image_path: str) -> np.ndarray:
        
        img = cv2.imread(image_path)

        # reformatting
        img = img[...,::-1] # BGR -> RGB
        img = img.transpose(2, 0, 1)
        img = img.astype(np.float32) / 255

        return img

    def __getitem__(self, index):
        
        image_path = self.image_paths[index]

        img = self._load_image(image_path)

        return torch.tensor(img), torch.empty(size=(0, ))

def danbooru(download=True):

    root_path = cache_path("danbooru")

    ds = Danbooru2019Portraits(root_path=root_path, download=download)
    return ImageDataset(training_set=ds)