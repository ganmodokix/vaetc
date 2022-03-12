import os

import torch
from torch.nn import functional as F
from torchvision import datasets, transforms

from .utils import ImageDataset, cache_path, IMAGE_HEIGHT, IMAGE_WIDTH

def cub_200_2011():
    """ Caltech-UCSD Birds-200-2011
    (http://www.vision.caltech.edu/visipedia/CUB-200-2011.html)
    
    Returns:
        ImageDataset
    """

    path_to_dataset = cache_path("cub_200_2011/cub_200_2011_cropped")

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Resize(size=(IMAGE_HEIGHT, IMAGE_WIDTH)),
    ])

    target_transform = transforms.Compose([
        transforms.Lambda(lambda x: torch.tensor(x)),
        transforms.Lambda(lambda x: F.one_hot(x, num_classes=200).float()),
    ])

    training_set = datasets.ImageFolder(
        root=os.path.join(path_to_dataset, "train"),
        transform=transform,
        target_transform=target_transform)

    test_set = datasets.ImageFolder(
        root=os.path.join(path_to_dataset, "test"),
        transform=transform,
        target_transform=target_transform)

    return ImageDataset(training_set, test_set)