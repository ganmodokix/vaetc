# vaetc: VAE-based representation learning toolkit

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

## Available assets

### Models
Representation learning methods below are implemented in PyTorch. They are mainly autoencoder-based models.
- AE [[Ballard, 1987]](https://www.aaai.org/Library/AAAI/1987/aaai87-050.php)
- Neocognitron [[Fukushima, 1980]](https://link.springer.com/article/10.1007/BF00344251) / CNN [[LeCun+, 1989]](https://ieeexplore.ieee.org/document/6795724)
- VAE [[Kingma+, 2013]](https://openreview.net/forum?id=33X9fd2-9FyZd)
- Conditional VAE [[Kingma+, 2014]](https://papers.nips.cc/paper/2014/hash/d523773c6b194f37b938d340d5d02232-Abstract.html)
- VAE-GAN [[Larsen+, 2015]](https://arxiv.org/abs/1512.09300)
- ALI [[Dumoulin+, 2016]](https://openreview.net/forum?id=B1ElR4cgg) / BiGAN [[Donahue+, 2016]](https://arxiv.org/abs/1605.09782v7)
- $\beta$-VAE [[Higgins+, 2016]](https://openreview.net/forum?id=Sy2fzU9gl)
- LadderVAE [[Sønderby+, 2016]](https://arxiv.org/abs/1602.02282)
- AVB [[Mescheder+, 2017]](https://proceedings.mlr.press/v70/mescheder17a.html)
- AnnealedVAE [[Burgess+, 2017]](https://arxiv.org/abs/1804.03599)
- Deep Feature Consistent VAE [[Hou+, 2017]](https://ieeexplore.ieee.org/document/7926714)
- InfoVAE (MMD divergence) [[Zhao+, 2017]](https://arxiv.org/abs/1706.02262)
- VLadderAE [[Zhao+, 2017]](https://proceedings.mlr.press/v70/zhao17c.html)
- DIP-VAE-I/-II [[Kumar+, 2018]](https://openreview.net/forum?id=H1kG7GZAW)
- FactorVAE [[Kim and Mnih, 2018]](http://proceedings.mlr.press/v80/kim18b.html)
- DAGMM [[Zong+, 2018]](https://openreview.net/forum?id=BJJLHbb0-)
- IntroVAE [[Huang+, 2018]](https://proceedings.neurips.cc/paper/2018/hash/093f65e080a295f8076b1c5722a46aa2-Abstract.html)
- $\beta$-VAE with cyclical annealing [[Fu+, 2019]](https://arxiv.org/abs/1903.10145)
- U-VITAE [[Detlefsen and Hauberg, 2019]](https://proceedings.neurips.cc/paper/2019/hash/3493894fa4ea036cfc6433c3e2ee63b0-Abstract.html)
- GuidedVAE [[Ding+, 2020]](https://openaccess.thecvf.com/content_CVPR_2020/html/Ding_Guided_Variational_Autoencoder_for_Disentanglement_Learning_CVPR_2020_paper.html)
- Soft-IntroVAE [[Daniel+, 2021]](https://openaccess.thecvf.com/content/CVPR2021/html/Daniel_Soft-IntroVAE_Analyzing_and_Improving_the_Introspective_Variational_Autoencoder_CVPR_2021_paper.html)
- $\sigma$-VAE [[Rybkin+, 2021]](http://proceedings.mlr.press/v139/rybkin21a.html)
- $\beta$-Annealed VAE [[Sankarapandian+, 2021]](https://arxiv.org/abs/2107.10667)

### Datasets
Many visual datasets are avaiable as below.
- COIL-20 [[Nene+, 1996]](https://www.cs.columbia.edu/CAVE/software/softlib/coil-20.php)
- MNIST [[LeCun+, 1998]](http://yann.lecun.com/exdb/mnist/)
- SmallNORB [[Huang and LeCun, 2005]](https://cs.nyu.edu/~ylclab/data/norb-v1.0-small/)
- CIFAR-10 [[Krizhevsky, 2009]](https://www.cs.toronto.edu/~kriz/cifar.html)
- STL-10 [[Coates+, 2011]](https://cs.stanford.edu/~acoates/stl10)
- SVHN [[Netzer+, 2011]](http://ufldl.stanford.edu/housenumbers/)
- CUB-200 2011 [[Wah+, 2011]](http://www.vision.caltech.edu/visipedia/CUB-200-2011.html)
- LSUN Bedroom [[Yu+, 2015]](https://www.yf.io/p/lsun)
- Omniglot [[Lake+, 2015]](https://github.com/brendenlake/omniglot)
- CelebA [[Liu+, 2015]](https://mmlab.ie.cuhk.edu.hk/projects/CelebA.html)
- (Work in progress) WiderFace [[Yang+, 2016]](http://shuoyang1213.me/WIDERFACE/)
- Fake dataset [[Torch Contributors, 2017]](https://pytorch.org/vision/stable/generated/torchvision.datasets.FakeData.html#torchvision.datasets.FakeData)
- dSprites [[Matthey+, 2017]](https://github.com/deepmind/dsprites-dataset)
- 3D Shapes [[Burgess+, 2018]](https://github.com/deepmind/3d-shapes)
- FFHQ [[Karras+, 2018]](https://openaccess.thecvf.com/content_CVPR_2019/html/Karras_A_Style-Based_Generator_Architecture_for_Generative_Adversarial_Networks_CVPR_2019_paper.html)
- Getchu AnimeFace [[Chao+, 2018]](https://github.com/bchao1/Anime-Face-Dataset)
- KMNIST [[Clanuwat+, 2018]](http://codh.rois.ac.jp/kmnist/index.html.en)
- MPI3D [[Gondal+, 2019]](https://proceedings.neurips.cc/paper/2019/hash/d97d404b6119214e4a7018391195240a-Abstract.html)
- Teapots [[Kim+, 2019]](https://openaccess.thecvf.com/content_ICCV_2019/html/Kim_Bayes-Factor-VAE_Hierarchical_Bayesian_Deep_Auto-Encoder_Models_for_Factor_Disentanglement_ICCV_2019_paper.html)
- AFHQ [[Choi+, 2020]](https://openaccess.thecvf.com/content_CVPR_2020/html/Choi_StarGAN_v2_Diverse_Image_Synthesis_for_Multiple_Domains_CVPR_2020_paper.html)
- Danbooru2019 Portraits [[Branwen+, 2020]](https://www.gwern.net/Crops)

These datasets are (manually) downloaded and cached in `$VAETC_PATH` (or `~/.vaetc` in default).

### Metrics
Many quantitative metrics are available, especially a variety of disentanglement metrics.
- Reconstruction MSE
- Reconstruction PSNR
- Rate & Distortion (SGVB estimator) [[Kingma+, 2013]](https://openreview.net/forum?id=33X9fd2-9FyZd)
- ELBO (SGVB estimator) [[Kingma+, 2013]](https://openreview.net/forum?id=33X9fd2-9FyZd)
- Z-min ($\beta$-VAE metric) [[Higgins+, 2016]](https://openreview.net/forum?id=Sy2fzU9gl)
- Z-diff (FactorVAE metric) [[Kim and Mnih, 2018]](http://proceedings.mlr.press/v80/kim18b.html)
- DCI Disentanglement [[Eastwood+, 2018]](https://openreview.net/forum?id=By-7dz-AZ)
- DCI Completeness [[Eastwood+, 2018]](https://openreview.net/forum?id=By-7dz-AZ)
- DCI Informativeness [[Eastwood+, 2018]](https://openreview.net/forum?id=By-7dz-AZ)
- SAP [[Kumar+, 2018]](https://openreview.net/forum?id=H1kG7GZAW)
- Modularity Score [[Ridgeway and Mozer, 2018]](https://proceedings.neurips.cc/paper/2018/hash/2b24d495052a8ce66358eb576b8912c8-Abstract.html)
- MIG / Robust MIG [[Do and Tran, 2019]](https://openreview.net/forum?id=HJgK0h4Ywr)
- WINDIN [[Do and Tran, 2019]](https://openreview.net/forum?id=HJgK0h4Ywr)
- JEMMIG [[Do and Tran, 2019]](https://openreview.net/forum?id=HJgK0h4Ywr)
- DCIMIG [[Sepliarskaia+, 2019]](https://arxiv.org/abs/1910.05587)
- IRS [[Suter+, 2019]](http://proceedings.mlr.press/v97/suter19a.html)
- MIG-sup [[Li+, 2019]](https://openreview.net/forum?id=SJxpsxrYPS)

## Citation
If you use this toolkit, please cite it as below:
```bibtex
@misc{
    title = {{vaetc}: VAE-based Representation Learning Toolkit},
    author = {Ganmodokix},
    howpublished = {\url{https://github.com/ganmodokix/vaetc}}
}
```