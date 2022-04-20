import os, sys

import yaml

import torch

sys.path.append(os.path.dirname(__file__) + '/../')
import vaetc
sys.path.pop()

if __name__ == "__main__":

    checkpoint = vaetc.Checkpoint(options={
        "model_name": "twostagevae",
        "dataset": "celeba",
        "batch_size": 256,
        "logger_path": "runs.tests/twostagevae_celeba_L64",
        "hyperparameters": yaml.safe_dump({
            "lr": 1e-4,
            "z_dim": 64,
        }),
        "cuda_sync": False,
        "very_verbose": True,
        "epochs": 256,
        # "until_convergence": True,
        # "patience": 10,
    })

    vaetc.fit(checkpoint)
    vaetc.evaluate(checkpoint)