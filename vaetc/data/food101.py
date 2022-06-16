import torch
from torch.nn import functional as F
from torchvision import datasets, transforms

from .utils import IMAGE_CHANNELS, IMAGE_HEIGHT, IMAGE_WIDTH, ImageDataset, cache_path

def food101():
    """ The Food-101 Dataset
    (https://data.vision.ee.ethz.ch/cvl/datasets_extra/food-101/)
    
    Returns:
        ImageDataset
    """

    path_to_dataset = cache_path("food101")

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Resize(size=(IMAGE_HEIGHT, IMAGE_WIDTH)),
    ])

    target_transform = transforms.Compose([
        transforms.Lambda(lambda x: torch.tensor(x)),
        transforms.Lambda(lambda x: F.one_hot(x, num_classes=101).float()),
    ])

    training_set = datasets.Food101(
        root=path_to_dataset,
        split="train",
        download=True,
        transform=transform,
        target_transform=target_transform)

    test_set = datasets.Food101(
        root=path_to_dataset,
        split="test",
        download=True,
        transform=transform,
        target_transform=target_transform)

    return ImageDataset(training_set, test_set)
