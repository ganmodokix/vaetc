import torch
from torch.nn import functional as F
from torchvision import datasets, transforms

from .utils import IMAGE_HEIGHT, IMAGE_WIDTH, ImageDataset, cache_path

def svhn():
    """ SVHN dataset (http://ufldl.stanford.edu/housenumbers/)
    
    Returns:
        ImageDataset
    """

    path_to_dataset = cache_path("svhn")

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Resize(size=(IMAGE_HEIGHT, IMAGE_WIDTH)),
    ])

    target_transform = transforms.Compose([
        transforms.Lambda(lambda x: torch.tensor(x)),
        transforms.Lambda(lambda x: F.one_hot(x, num_classes=10).float()),
    ])


    training_set = datasets.SVHN(
        root=path_to_dataset,
        split="train",
        download=True,
        transform=transform,
        target_transform=target_transform)

    test_set = datasets.SVHN(
        root=path_to_dataset,
        split="test",
        download=True,
        transform=transform,
        target_transform=target_transform)

    return ImageDataset(training_set, test_set)