import os, sys

import yaml

import torch

sys.path.append(os.path.dirname(__file__) + '/../')
import vaetc
sys.path.pop()

if __name__ == "__main__":

    checkpoint = vaetc.Checkpoint(options={
        "model_name": "vamppriorvae",
        "dataset": "mnist",
        "batch_size": 256,
        "logger_path": "runs.tests/vamppriorvae",
        "hyperparameters": yaml.safe_dump({
            "lr": 1e-3,
            "z_dim": 16,
            "num_pseudo_inputs": 500,
        }),
        "cuda_sync": False,
        "very_verbose": True,
        "epochs": 64,
        # "until_convergence": True,
        # "patience": 10,
    })

    vaetc.fit(checkpoint)
    vaetc.evaluate(checkpoint)