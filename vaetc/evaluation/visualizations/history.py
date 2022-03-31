import os

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from vaetc.checkpoint import Checkpoint

def visualize(checkpoint: Checkpoint):

    history_dir = os.path.join(checkpoint.options["logger_path"], "history")
    os.makedirs(history_dir, exist_ok=True)
    for key in checkpoint.history:

        if key.startswith("val_"): continue
            
        x = np.arange(len(checkpoint.history[key]))
        y_train = checkpoint.history[key]
        y_valid = checkpoint.history[f"val_{key}"]

        plt.figure()
        sns.set(style="whitegrid")
        plt.plot(x, y_train, label="Training")
        plt.plot(x, y_valid, label="Validation")
        plt.legend()
        plt.savefig(os.path.join(history_dir, f"{key}.svg"))
        plt.savefig(os.path.join(history_dir, f"{key}.pdf"))
        plt.close()