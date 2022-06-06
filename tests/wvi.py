import os, sys

import yaml

import torch

sys.path.append(os.path.dirname(__file__) + '/../')
import vaetc
sys.path.pop()

if __name__ == "__main__":

    checkpoint = vaetc.Checkpoint(options={
        "model_name": "wvi",
        "dataset": "celeba",
        "batch_size": 64,
        "logger_path": "runs.tests/wvi_celeba",
        "hyperparameters": yaml.safe_dump({
            "lr": 1e-4,
            "z_dim": 64,
            "sinkhorn_iterations": 50,
            "eps": 0.01,
        }),
        "cuda_sync": True,
        "very_verbose": True,
        # "until_convergence": True,
        # "patience": 5,
        "epochs": 100,
    })

    vaetc.fit(checkpoint)
    vaetc.evaluate(checkpoint)