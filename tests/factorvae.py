import os, sys

import yaml

import torch

sys.path.append(os.path.dirname(__file__) + '/../')
import vaetc
from vaetc.models import FactorVAE

def visualize_disc(model: FactorVAE):

    ls = torch.linspace(-3., 3., 256)
    x = torch.stack([ls[:,None].tile(1, 256), ls[None,:].tile(256, 1)], dim=2)
    x = torch.cat([x, torch.zeros(size=[256, 256, model.z_dim - 2])], dim=2)
    x = x.view(-1, model.z_dim)
    x = x.cuda()
    
    p = model.disc_block(x)

    p = p.cpu().detach()
    p = p.view(256, 256, 2)[:,:,1]
    p = p.numpy()


if __name__ == "__main__":

    vaetc.deterministic(3407)

    checkpoint = vaetc.Checkpoint(options={
        "model_name": "factorvae",
        "dataset": "mnist",
        "epochs": 10,
        "batch_size": 256,
        "logger_path": "runs.tests/factorvae2",
        "hyperparameters": yaml.safe_dump({
            "lr": 1e-4,
            "lr_disc": 1e-3,
            "z_dim": 16,
            "gamma": 100
        }),
        "cuda_sync": False,
        "very_verbose": True,
        "until_convergence": True,
        "patience": 5,
    })

    vaetc.fit(checkpoint)
    vaetc.evaluate(checkpoint)