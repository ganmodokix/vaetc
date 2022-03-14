import os, sys

import yaml

import torch

sys.path.append(os.path.dirname(__file__) + '/../')
import vaetc
sys.path.pop()

if __name__ == "__main__":

    vaetc.deterministic(3407)

    checkpoint = vaetc.Checkpoint(options={
        "model_name": "avb",
        "dataset": "getchu",
        "epochs": 100,
        "batch_size": 256,
        "logger_path": "runs.tests/avb",
        "hyperparameters": yaml.safe_dump({
            "lr": 1e-3,
            "lr_disc": 1e-3,
            "beta": 1,
            "z_dim": 16,
        }),
        "cuda_sync": True,
        "very_verbose": True,
        "until_convergence": True,
        "patience": 5,
    })

    vaetc.fit(checkpoint)
    vaetc.evaluate(checkpoint)