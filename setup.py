import os

import pkg_resources
from setuptools import setup, find_packages

requirements_path = os.path.join(os.path.dirname(__file__), "requirements.txt")

with open(requirements_path, "r") as fp:
    install_requires = []
    for line in fp.readlines():
        line = line.strip()
        if line != "":
            install_requires += [line]

setup(
    name="vaetc",
    version="0.1.0",
    description="A representation learning toolkit for PyTorch, mainly on VAE-based models",
    packages=find_packages(),
    install_requires=install_requires
)