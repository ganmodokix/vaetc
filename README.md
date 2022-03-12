# vaetc

(mainly-)VAE-based representation learning toolkit

## Environments
We have developed and tested this repository in the following environment:
- Python 3.9.9
- PyTorch 1.10.1+cu102
- CUDA 11.4
- Ubuntu 18.04.5

### Setup
We recommend to use `pip` within a `venv` environment.
```
sudo apt-get install python3.9 python3.9-dev python3.9-venv
python3.9 -m venv .env
source .env/bin/activate
pip install -U pip
pip install -r requirements.txt
```

## How to train
```python
import vaetc

options = {
    "model_name": "factorvae",
    "hyperparameters": r'{"lr": 1e-4, "lr_disc": 1e-4, "z_dim": 16, "gamma": 6}',
    "dataset": "mnist",
    "logger_path": "runs.test",
    "epochs": 15,
    "batch_size": 256,
    "cuda_sync": True,
    "very_verbose": True,
}
checkpoint = vaetc.Checkpoint(options)
vaetc.fit(checkpoint)
vaetc.evaluate(checkpoint)
```

## Dataset Cache
Datasets are downloaded in `$VAETC_PATH` (or `~/.vaetc` in default)

## Citation
If you use this toolkit, please cite it as below:
```bibtex
@misc{
    title = {{vaetc}: VAE-based Representation Learning Toolkit},
    author = {Ganmodokix},
    howpublished = {\url{https://github.com/ganmodokix/vaetc}}
}
```