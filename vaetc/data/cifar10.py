import torch
from torch.nn import functional as F
from torchvision import datasets, transforms

from .utils import ImageDataset, cache_path, IMAGE_HEIGHT, IMAGE_WIDTH

def cifar10():
    """ CIFAR-10 (https://www.cs.toronto.edu/~kriz/cifar.html)
    
    Returns:
        ImageDataset
    """

    path_to_dataset = cache_path("cifar10")

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Resize(size=(IMAGE_HEIGHT, IMAGE_WIDTH))
    ])

    target_transform = transforms.Compose([
        transforms.Lambda(lambda x: torch.tensor(x)),
        transforms.Lambda(lambda x: F.one_hot(x, num_classes=10).float()),
    ])

    training_set = datasets.CIFAR10(
        root=path_to_dataset,
        train=True,
        download=True,
        transform=transform,
        target_transform=target_transform)

    test_set = datasets.CIFAR10(
        root=path_to_dataset,
        train=False,
        download=True,
        transform=transform,
        target_transform=target_transform)

    return ImageDataset(training_set, test_set)