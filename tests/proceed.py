import os, sys

import yaml

import torch

sys.path.append(os.path.dirname(__file__) + '/../')
import vaetc

if __name__ == "__main__":

    checkpoint_path = "runs.tests/current/checkpoint_last.pth"
    checkpoint = vaetc.load_checkpoint(checkpoint_path=checkpoint_path)

    vaetc.proceed(checkpoint, extend_epochs=200)
    vaetc.evaluate(checkpoint)