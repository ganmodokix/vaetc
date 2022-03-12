import os

import torch
from torch.nn import functional as F
from torchvision import transforms, datasets

from .utils import ImageDataset, cache_path, IMAGE_HEIGHT, IMAGE_WIDTH

def image_dataset(dataset_name):
    """ Fallback Dataset with classes"""

    dir_path = cache_path(dataset_name)
    assert os.path.isdir(dir_path)
    
    class_list = set()
    def add_class(set_path, class_list):
        if os.path.isdir(set_path):
            for entry in os.scandir(set_path):
                if entry.is_dir():
                    class_list.add(entry.name)
    
    add_class(os.path.join(dir_path, "train"), class_list)
    add_class(os.path.join(dir_path, "test"), class_list)
    add_class(os.path.join(dir_path, "validation"), class_list)
    num_classes = len(class_list)

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Resize(size=(IMAGE_HEIGHT, IMAGE_WIDTH)),
    ])

    target_transform = transforms.Compose([
        transforms.Lambda(lambda x: torch.tensor(x)),
        transforms.Lambda(lambda x: F.one_hot(x, num_classes=num_classes).float()),
    ])

    training_set = datasets.ImageFolder(
        root=os.path.join(dir_path, "train"),
        transform=transform,
        target_transform=target_transform) \
        if os.path.isdir(os.path.join(dir_path, "train")) else None

    test_set = datasets.ImageFolder(
        root=os.path.join(dir_path, "test"),
        transform=transform,
        target_transform=target_transform) \
        if os.path.isdir(os.path.join(dir_path, "test")) else None

    validation_set = datasets.ImageFolder(
        root=os.path.join(dir_path, "validation"),
        transform=transform,
        target_transform=target_transform) \
        if os.path.isdir(os.path.join(dir_path, "validation")) else None

    return ImageDataset(training_set, test_set, validation_set)