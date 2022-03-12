import torch
from torch.nn import functional as F
from torchvision import datasets, transforms

from .utils import ImageDataset, IMAGE_SHAPE

def fake():
    """ Fake Dataset by torchvision
    
    Returns:
        ImageDataset
    """

    target_transform = transforms.Compose([
        transforms.Lambda(lambda x: torch.LongTensor([x])),
        transforms.Lambda(lambda x: F.one_hot(x, num_classes=10).float()),
    ])

    training_set = datasets.FakeData(
        size=10000,
        image_size=IMAGE_SHAPE,
        num_classes=10,
        transform=transforms.ToTensor(),
        target_transform=target_transform)

    return ImageDataset(training_set)
