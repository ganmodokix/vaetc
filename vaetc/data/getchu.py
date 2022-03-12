import torch
from torch.nn import functional as F
from torchvision import datasets, transforms

from .utils import ImageDataset, cache_path, IMAGE_HEIGHT, IMAGE_WIDTH

def getchu():
    """ Getchu Animeface Dataset
    (https://github.com/bchao1/Anime-Face-Dataset)
    
    Returns:
        ImageDataset
    """

    path_to_dataset = cache_path("getchu/getchu")

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Resize(size=(IMAGE_HEIGHT, IMAGE_WIDTH)),
    ])

    target_transform = transforms.Compose([
        transforms.Lambda(lambda x: torch.tensor(x)),
        transforms.Lambda(lambda x: F.one_hot(x, num_classes=20).float()),
    ])


    training_set = datasets.ImageFolder(
        root=path_to_dataset,
        transform=transform,
        target_transform=target_transform)

    return ImageDataset(training_set)