import os, sys

import yaml

import torch

sys.path.append(os.path.dirname(__file__) + '/../')
import vaetc
sys.path.pop()

if __name__ == "__main__":

    checkpoint = vaetc.Checkpoint(options={
        "model_name": "wvi",
        "dataset": "mnist",
        "batch_size": 256,
        "logger_path": "runs.tests/wvi",
        "hyperparameters": yaml.safe_dump({
            "lr": 1e-3,
            "z_dim": 64,
        }),
        "cuda_sync": True,
        "very_verbose": True,
        "until_convergence": True,
        "patience": 5,
    })

    vaetc.fit(checkpoint)
    vaetc.evaluate(checkpoint)