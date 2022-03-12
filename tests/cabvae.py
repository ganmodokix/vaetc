import os, sys

import yaml

import torch

sys.path.append(os.path.dirname(__file__) + '/../')
import vaetc

if __name__ == "__main__":

    vaetc.non_deterministic()

    checkpoint = vaetc.Checkpoint(options={
        "model_name": "cabvae",
        "dataset": "mnist",
        "epochs": 100,
        "batch_size": 256,
        "logger_path": "runs.tests/cabvae",
        "hyperparameters": yaml.safe_dump({
            "lr": 1e-4,
            "z_dim": 2,
            "beta": 15,
        }),
        "cuda_sync": True,
        "very_verbose": True,
    })

    vaetc.fit(checkpoint)
    vaetc.evaluate(checkpoint)