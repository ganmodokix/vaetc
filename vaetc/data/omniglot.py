import torch
from torch.nn import functional as F
from torchvision import datasets, transforms

from .utils import IMAGE_CHANNELS, IMAGE_HEIGHT, IMAGE_WIDTH, ImageDataset, cache_path
from vaetc.utils import debug_print

def omniglot():
    """ Omniglot dataset
    (https://github.com/brendenlake/omniglot/)
    
    Returns:
        ImageDataset
    """

    path_to_dataset = cache_path("omniglot")

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Lambda(lambda x: x.repeat(IMAGE_CHANNELS, 1, 1)),
        transforms.Resize(size=(IMAGE_HEIGHT, IMAGE_WIDTH)),
    ])

    def test(x):
        debug_print(x)
        return x

    target_transform = transforms.Compose([
        transforms.Lambda(lambda x: torch.tensor(x)),
        transforms.Lambda(lambda x: F.one_hot(x, num_classes=1623).float()),
    ])

    background_set = datasets.Omniglot(
        root=path_to_dataset,
        background=True,
        download=True,
        transform=transform,
        target_transform=target_transform,
    )

    evaluation_set = datasets.Omniglot(
        root=path_to_dataset,
        background=False,
        download=True,
        transform=transform,
        target_transform=target_transform,
    )

    return ImageDataset(background_set, evaluation_set)
