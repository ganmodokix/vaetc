from torchvision import datasets, transforms
from torch.utils.data import Subset

from .utils import ImageDataset, cache_path, IMAGE_HEIGHT, IMAGE_WIDTH

def celeba():
    """ CelebA (https://mmlab.ie.cuhk.edu.hk/projects/CelebA.html)
    
    Returns:
        ImageDataset
    """

    path_to_dataset = cache_path("celeba")

    transform = transforms.Compose([
        transforms.CenterCrop(144),
        transforms.Resize(size=(IMAGE_HEIGHT, IMAGE_WIDTH)),
        transforms.ToTensor(),
    ])

    target_transform = transforms.Compose([
        transforms.Lambda(lambda x: x.float()),
    ])

    training_set = datasets.CelebA(
        root=path_to_dataset,
        split="train",
        download=True,
        transform=transform,
        target_transform=target_transform)

    test_set = datasets.CelebA(
        root=path_to_dataset,
        split="test",
        download=True,
        transform=transform,
        target_transform=target_transform)

    validation_set = datasets.CelebA(
        root=path_to_dataset,
        split="valid",
        download=True,
        transform=transform,
        target_transform=target_transform)

    return ImageDataset(training_set, test_set, validation_set)

def celeba_small():

    original_dataset = celeba()

    num_training   = len(original_dataset.training_set) // 10
    num_test       = len(original_dataset.test_set) // 10
    num_validation = len(original_dataset.validation_set) // 10

    training_set   = Subset(original_dataset.training_set  , list(range(num_training // 10)))
    test_set       = Subset(original_dataset.test_set      , list(range(num_test // 10)))
    validation_set = Subset(original_dataset.validation_set, list(range(num_validation // 10)))

    return ImageDataset(training_set, test_set, validation_set)
