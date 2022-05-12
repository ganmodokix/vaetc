import os, sys

import yaml

import torch

sys.path.append(os.path.dirname(__file__) + '/../')
import vaetc
sys.path.pop()

if __name__ == "__main__":

    checkpoint = vaetc.Checkpoint(options={
        "model_name": "laddervae",
        "dataset": "mnist",
        "batch_size": 256,
        "logger_path": "runs.tests/laddervae",
        "hyperparameters": yaml.safe_dump({
            "lr": 1e-4,
            "z_dim": 16,
            "num_layers": 2,
            "batchnorm_latent": True,
            "warmup": True,
        }),
        "cuda_sync": True,
        "very_verbose": True,
        "epochs": 64,
    })

    vaetc.fit(checkpoint)
    vaetc.evaluate(checkpoint)