import os, sys

import yaml

import torch

sys.path.append(os.path.dirname(__file__) + '/../')
import vaetc
sys.path.pop()

if __name__ == "__main__":

    checkpoint = vaetc.Checkpoint(options={
        "model_name": "vae",
        "dataset": "celeba",
        "batch_size": 256,
        "logger_path": "runs.tests/vae_gaussian-reconstruction",
        "hyperparameters": yaml.safe_dump({
            "lr": 1e-3,
            "z_dim": 16,
            "decoder_distribution": "gaussian",
        }),
        "cuda_sync": True,
        "very_verbose": True,
        "epochs": 100,
        # "until_convergence": True,
        # "patience": 10,
    })

    vaetc.fit(checkpoint)
    vaetc.evaluate(checkpoint)