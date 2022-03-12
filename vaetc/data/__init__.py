# x, t
# x: image with shape (IMAGE_CHANNELS, IMAGE_HEIGHT, IMAGE_WIDTH), ndim==1, float
# t: target ndim==1, float

from .cars import cars
from .celeba import celeba, celeba_small
from .cifar10 import cifar10
from .cub_200_2011 import cub_200_2011
from .fake import fake
from .getchu import getchu
from .kmnist import kmnist
from .lsun import lsun_bedroom
from .mnist import mnist
from .omniglot import omniglot
from .stl10 import stl10
from .svhn import svhn
from .widerface import widerface

from .dsprites import dsprites
from .shapes3d import shapes3d
from .mpi3d import mpi3d_real
from .smallnorb import smallnorb
from .teapot import teapot

from .afhq import afhq_v2_cat, afhq_v2_dog, afhq_v2_wild
from .ffhq import ffhq
from .danbooru import danbooru

from .images import image_dataset
from .utils import ImageDataset
from .utils import IMAGE_SHAPE, IMAGE_CHANNELS, IMAGE_HEIGHT, IMAGE_WIDTH