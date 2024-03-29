import os
from typing import Dict, List, Optional
from matplotlib.colors import NoNorm

import yaml
import numpy as np
from scipy.stats import sem
import torch
from tqdm import tqdm

from torch.utils.data import Dataset, DataLoader

from vaetc.evaluation.metrics.do2020.mutual import dcimig, modularity

from . import do2020
from . import dci
from . import transfer
from . import intervention
from . import predictor
from . import elbo

from vaetc.utils import debug_print
from vaetc.checkpoint import Checkpoint
from vaetc.models.abstract import RLModel, GaussianEncoderAutoEncoderRLModel
from vaetc.evaluation.preprocess import encode_set, EncodedData

CELEBA_PROMINENT_ATTRIBUTES = [
    (4, "Bald"),
    (5, "Bangs"),
    (20, "Male"),
    (26, "Pale Skin"),
    (31, "Smiling"),
    (39, "Young"),
]

def evaluate_set(data: EncodedData, random_state: int = 42) -> Dict[str, float]:

    results = {}

    def add_result(metric_name, arr, verbose=True):
        
        if verbose:
            debug_print(f"Added result {metric_name}: {float(np.mean(arr))}")
        
        if metric_name in results:
            results[metric_name] += [arr]
        else:
            results[metric_name] = [arr]

    def mean_result():
        means = {}
        for metric_name in results:
            means[metric_name] = np.mean(results[metric_name])
            means[metric_name] = float(means[metric_name])
        return means
        
    # Intervention-based metrics
    debug_print("Calculating Beta-VAE metric...")
    betavae_metric = intervention.betavae_metric(data.z, data.t, random_state=random_state)
    add_result("Beta-VAE Metric", betavae_metric)

    debug_print("Calculating FactorVAE metric...")
    factorvae_metric = intervention.factorvae_metric(data.z, data.t, random_state=random_state)
    add_result("FactorVAE Metric", factorvae_metric)

    debug_print("Calculating IRS metric...")
    irs = intervention.irs(data.z, data.t)
    add_result("IRS", irs)

    # Predictor-based metrics
    debug_print("Calculating SAP score...")
    sap_score = predictor.sap_score(data.z, data.t, random_state=random_state)
    add_result("SAP", sap_score)

    # MIG-based in a [Do et al., 2020]'s manner
    if data.is_gaussian():

        # # [WIP] too slow
        # debug_print("Calculating MISJED...")
        # misjed = do2020.misjed(data.mean, data.logvar)
        # add_result("MISJED", misjed)

        debug_print("Calculating Informativeness, WINDIN, (R)MIG, and JEMMIG...")
        for x_i, t_i, z_i, mean_i, logvar_i, x2_i in tqdm(data.iter_batch(), total=data.num_batch()):

            elbo_est = elbo.elbo(x_i, x2_i, mean_i, logvar_i)
            rate = elbo.rate(mean_i, logvar_i)
            distortion = elbo.distortion(x_i, x2_i)
            mse = elbo.mean_squared_error(x_i, x2_i)
            psnr = elbo.psnr(x_i, x2_i)
            add_result("ELBO", elbo_est.mean(), verbose=False)
            add_result("Rate", rate.mean(), verbose=False)
            add_result("Distortion", distortion.mean(), verbose=False)
            add_result("Reconstruction MSE", mse.mean(), verbose=False)
            add_result("Reconstruction PSNR [dB]", psnr.mean(), verbose=False)

            informativeness = do2020.informativeness(mean_i, logvar_i)
            add_result("Informativeness", np.mean(informativeness), verbose=False)
            for i in range(data.z_dim()):
                add_result(f"Informativeness (z_{i:03})", informativeness[i], verbose=False)
            
            windin = do2020.windin(mean_i, logvar_i)
            add_result("WINDIN", windin, verbose=False)

            rmig_k, jemmig_k, normalized_jemmig_k = do2020.rmig_jemmig(mean_i, logvar_i, t_i)
            add_result("RMIG", np.mean(rmig_k), verbose=False)
            add_result("JEMMIG", np.mean(jemmig_k), verbose=False)
            add_result("Normalized JEMMIG", np.mean(normalized_jemmig_k), verbose=False)
            for i in range(data.t_dim()):
                add_result(f"RMIG (t_{i:03})", rmig_k[i], verbose=False)
                add_result(f"JEMMIG (t_{i:03})", jemmig_k[i], verbose=False)
                add_result(f"Normalized JEMMIG (t_{i:03})", normalized_jemmig_k[i], verbose=False)

            mig_sup_i = do2020.mig_sup(mean_i, logvar_i, t_i)
            add_result("MIG-sup", np.mean(mig_sup_i), verbose=False)
            for i in range(data.z_dim()):
                add_result(f"MIG-sup (z_{i:03})", mig_sup_i[i], verbose=False)

            modularity = do2020.modularity(mean_i, logvar_i, t_i)
            add_result("Modularity Score", np.mean(modularity), verbose=False)
            for i in range(data.z_dim()):
                add_result(f"Modularity Score (z_{i:03})", modularity[i], verbose=False)

            dcimig = do2020.dcimig(mean_i, logvar_i, t_i)
            add_result("DCIMIG", dcimig, verbose=False)

    else:

        debug_print("Skipped the information-based metrics; q(z|x) is not Gaussian")

    # DCI scores
    debug_print("Calculating the DCI metrics...")
    disentanglement, completeness, informativeness = dci.dci_score(data.z, data.t)
    add_result("DCI Disentanglement", disentanglement)
    add_result("DCI Completeness", np.mean(completeness))
    add_result("DCI Informativeness", np.mean(informativeness))
    for i in range(data.t_dim()):
        add_result(f"DCI Completeness (t_{i:03d})", completeness[i], verbose=False)
        add_result(f"DCI Informativeness (t_{i:03d})", informativeness[i], verbose=False)

    # linear transfer learning
    debug_print("Evaluating the performance of transfer learning...")
    acc_linear, acc_dummy = transfer.linear_transferability(data.z, data.t)
    add_result("Linear Transferability Target Accuracy", acc_linear)
    add_result("Dummy Target Accuracy", acc_dummy)
    for i in range(data.t_dim()):
        add_result(f"Linear Transferability Target Accuracy (t_{i:03d})", acc_linear[i], verbose=False)
        add_result(f"Dummy Target Accuracy (t_{i:03d})", acc_dummy[i], verbose=False)

    return mean_result()

def measure(model: RLModel, dataset: Dataset, logger_path: str, suffix: str, num_measurement: int = 1):

    data = EncodedData(model, dataset)

    # ==================================================
    # for additional evaluations

    np.savez_compressed(os.path.join(logger_path, f"zt_{suffix}"), z=data.z, t=data.t)

    # ==================================================
    # (multiple) measurement(s)

    evaluations_sets = []
    for k in range(num_measurement):

        debug_print(f"Measurement {k+1}/{num_measurement} ...")

        evaluations = evaluate_set(data)
        evaluations_sets.append(evaluations)

    evaluations_mean, evaluations_se = {}, {}
    for key in evaluations_sets[0]:
        values = [evaluations_sets[i][key] for i in range(num_measurement)]
        evaluations_mean[key] = float(np.mean(values))
        if num_measurement >= 2:
            evaluations_se[key] = float(sem(values))

    with open(os.path.join(logger_path, f"metrics_{suffix}.yaml"), "w") as fp:
        yaml.safe_dump(evaluations_mean, fp)
    
    if num_measurement >= 2:
        with open(os.path.join(logger_path, f"metrics_{suffix}_se.yaml"), "w") as fp:
            yaml.safe_dump(evaluations_se, fp)
    
    for k in range(num_measurement):
        with open(os.path.join(logger_path, f"metrics_{suffix}_{k}.yaml"), "w") as fp:
            yaml.safe_dump(evaluations_sets[k], fp)

def evaluate(checkpoint: Checkpoint):

    debug_print("in the validation set")

    logger_path = checkpoint.options["logger_path"]

    measure(checkpoint.model, checkpoint.dataset.validation_set, logger_path, "valid")
    measure(checkpoint.model, checkpoint.dataset.test_set, logger_path, "test")
    