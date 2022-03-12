from torchvision import datasets, transforms

from .utils import IMAGE_HEIGHT, IMAGE_WIDTH, ImageDataset, cache_path

def lsun_bedroom():
    """ LSUN Bedroom (https://www.yf.io/p/lsun)
    
    Returns:
        ImageDataset
    """

    path_to_dataset = cache_path("lsun")
    
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Resize(size=(IMAGE_HEIGHT, IMAGE_WIDTH)),
    ])

    training_set = datasets.LSUN(
        root=path_to_dataset,
        classes=["bedroom_train"],
        transform=transform)

    validation_set = datasets.LSUN(
        root=path_to_dataset,
        classes=["bedroom_val"],
        transform=transform)

    return ImageDataset(training_set, validation_set)
