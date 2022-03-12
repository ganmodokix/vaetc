import torch
from torch.nn import functional as F
from torchvision import datasets, transforms

from .utils import IMAGE_CHANNELS, IMAGE_HEIGHT, IMAGE_WIDTH, ImageDataset, cache_path

def mnist():
    """ MNIST
    (http://yann.lecun.com/exdb/mnist/)
    
    Returns:
        ImageDataset
    """

    path_to_dataset = cache_path("mnist")

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Lambda(lambda x: x.repeat(IMAGE_CHANNELS, 1, 1)),
        transforms.Resize(size=(IMAGE_HEIGHT, IMAGE_WIDTH)),
    ])

    target_transform = transforms.Compose([
        transforms.Lambda(lambda x: torch.tensor(x)),
        transforms.Lambda(lambda x: F.one_hot(x, num_classes=10).float()),
    ])

    training_set = datasets.MNIST(
        root=path_to_dataset,
        train=True,
        download=True,
        transform=transform,
        target_transform=target_transform)

    test_set = datasets.MNIST(
        root=path_to_dataset,
        train=False,
        download=True,
        transform=transform,
        target_transform=target_transform)

    return ImageDataset(training_set, test_set)
