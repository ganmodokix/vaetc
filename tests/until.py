import os, sys

import yaml

import torch

sys.path.append(os.path.dirname(__file__) + '/../')
import vaetc

if __name__ == "__main__":

    vaetc.deterministic(3407)

    checkpoint = vaetc.Checkpoint(options={
        "model_name": "vae",
        "dataset": "kmnist",
        "epochs": 10,
        "batch_size": 256,
        "logger_path": "runs.tests/until",
        "hyperparameters": yaml.safe_dump({
            "lr": 1e-4,
            "lr_disc": 1e-3,
            "z_dim": 16,
            "gamma": 100
        }),
        "cuda_sync": False,
        "very_verbose": True,
        "until_convergence": True,
        "patience": 10
    })

    vaetc.fit(checkpoint)
    vaetc.evaluate(checkpoint)