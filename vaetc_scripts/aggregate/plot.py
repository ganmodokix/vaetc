import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import yaml
import argparse

def convex(x, y):
    x2 = []
    y2 = []
    for i in range(len(x)):
        while len(y2) >= 2 and y2[-1] - y2[-2] < (y[i] - y2[-2]) / (x[i] - x2[-2]) * (x2[-1] - x2[-2]):
            x2.pop()
            y2.pop()
        x2 += [x[i]]
        y2 += [y[i]]
    return np.array(x2), np.array(y2)

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--input_path", "-i", type=str, default="aggregate/aggregated.yaml", help="yaml path")
    parser.add_argument("--output_path", "-o", type=str, default="aggregate/plot.svg", help="output path (.svg recommended)")
    args = parser.parse_args()

    with open(args.input_path, "r") as fp:
        results = yaml.safe_load(fp)

    x_name = "DCI Disentanglement"
    y_name = "DCI Informativeness"
    x, y, c = [], [], []

    for inst in results:

        if "test" in inst["metrics"]:

            x.append(inst["metrics"]["test"][x_name])
            y.append(inst["metrics"]["test"][y_name])

            hyperparameters_label = ""
            for name, value in inst["hyperparameters"].items():
                hyperparameters_label += f"{name}={value}\n"
            hyperparameters_label = hyperparameters_label.strip()
            
            ci = f"""{inst["options"]["model_name"]}\n""" \
                + f"""{hyperparameters_label}"""
            
            c.append(ci)

    model_names = [inst["options"]["model_name"] for inst in results]
    model_names = list(set(model_names))
    model_names.sort()
    if "vsevae2" in model_names:
        model_names.remove("vsevae2")
    model_names = ["vsevae2"] + model_names

    model_name_map = {
        "vsevae2": "Ours",
        "vae": "VAE",
        "bvae": "$\\beta$-VAE",
        "btcvae": "$\\beta$-TCVAE",
        "dipvaei": "DIP-VAE-I",
        "dipvaeii": "DIP-VAE-II",
        "factorvae": "FactorVAE",
        "infovae": "InfoVAE",
    }

    plt.figure()
    sns.set()
    
    for model_name in model_names:
        if model_name == "annealedvae": continue
        indices = [i for i in range(len(x)) if results[i]["options"]["model_name"] == model_name]
        xs = np.array([x[i] for i in indices])
        ys = np.array([y[i] for i in indices])
        xargsort = np.argsort(xs)
        xs = xs[xargsort]
        ys = ys[xargsort]
        label = model_name_map[model_name] if model_name in model_name_map else model_name
        plt.scatter(xs, ys, label=label, marker="*" if model_name == "vsevae2" else "o")
        xs, ys = convex(xs, ys)
        plt.plot(xs, ys)
        # for i in indices:
        #     plt.annotate(c[i], (x[i], y[i]))
    plt.xlabel(x_name)
    plt.ylabel(y_name)

    xmin, xmax = min(*x), max(*x)
    ymin, ymax = min(*y), max(*y)
    xmin, xmax = xmin - (xmax - xmin) * 0.05, xmax + (xmax - xmin) * 0.2
    ymin, ymax = ymin - (ymax - ymin) * 0.05, ymax + (ymax - ymin) * 0.2

    plt.xlim([xmin, xmax])
    plt.ylim([ymin, ymax])
    plt.legend()

    plt.savefig(args.output_path)