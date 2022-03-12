from torchvision import datasets, transforms

from .utils import ImageDataset, cache_path

def widerface():
    """ Widerface dataset (http://shuoyang1213.me/WIDERFACE/)
    
    Returns:
        ImageDataset
    """

    raise NotImplementedError("WIP: Cropping")
    
    path_to_dataset = cache_path("widerface")

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Resize(size=(IMAGE_HEIGHT, IMAGE_WIDTH)),
    ])

    target_transform = transforms.Lambda(lambda _: 0)

    training_set = datasets.WIDERFace(
        root=path_to_dataset,
        split="train",
        transform=transform,
        target_transform=target_transform,
        download=False)

    validation_set = datasets.WIDERFace(
        root=path_to_dataset,
        split="val",
        transform=transform,
        target_transform=target_transform,
        download=False)

    test_set = datasets.WIDERFace(
        root=path_to_dataset,
        split="test",
        transform=transform,
        target_transform=target_transform,
        download=False)

    return ImageDataset(training_set, test_set, validation_set)
