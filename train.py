"""
Train a small neural network on the Indian Liver Patient Dataset (ILPD),
once normally and once with differential privacy (DP-SGD via Opacus),
and compare how much accuracy privacy costs.

Usage:
    python train.py               # full experiment (30 seeds x all epsilons)
    python train.py --seeds 2     # quick run
"""

import argparse
import json
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, balanced_accuracy_score, roc_auc_score
from opacus import PrivacyEngine

warnings.filterwarnings("ignore")  # Opacus warns about secure RNG; fine for an experiment

ROOT = Path(__file__).parent
DATA = ROOT / "data" / "raw" / "Indian Liver Patient Dataset (ILPD).csv"
RESULTS = ROOT / "results"

COLUMNS = [
    "age", "gender", "total_bilirubin", "direct_bilirubin", "alk_phosphotase",
    "alt", "ast", "total_proteins", "albumin", "ag_ratio", "label",
]
SKEWED = ["total_bilirubin", "direct_bilirubin", "alk_phosphotase", "alt", "ast"]


# ---------------------------------------------------------------- data

def load_data():
    df = pd.read_csv(DATA, header=None, names=COLUMNS)
    df["gender"] = (df["gender"] == "Male").astype(float)
    df["label"] = (df["label"] == 1).astype(float)  # 1 = liver patient, 0 = not
    for col in SKEWED:  # lab values have long tails; log makes them easier to learn
        df[col] = np.log1p(df[col])
    X = df.drop(columns="label").to_numpy(dtype=np.float32)
    y = df["label"].to_numpy(dtype=np.float32)
    return X, y


def split(X, y, seed):
    X_tr, X_te, y_tr, y_te = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=seed
    )
    # Fill the 4 missing A/G ratios and standardise using training rows only.
    med = np.nanmedian(X_tr, axis=0)
    X_tr = np.where(np.isnan(X_tr), med, X_tr)
    X_te = np.where(np.isnan(X_te), med, X_te)
    mu, sd = X_tr.mean(0), X_tr.std(0) + 1e-8
    return (X_tr - mu) / sd, (X_te - mu) / sd, y_tr, y_te


# ---------------------------------------------------------------- model

def make_model(n_features):
    return nn.Sequential(
        nn.Linear(n_features, 32), nn.ReLU(),
        nn.Linear(32, 16), nn.ReLU(),
        nn.Linear(16, 1),
    )


def evaluate(model, X_te, y_te):
    model.eval()
    with torch.no_grad():
        prob = torch.sigmoid(model(torch.from_numpy(X_te)).squeeze(1)).numpy()
    pred = (prob >= 0.5).astype(float)
    return {
        "accuracy": accuracy_score(y_te, pred),
        "balanced_accuracy": balanced_accuracy_score(y_te, pred),
        "roc_auc": roc_auc_score(y_te, prob),
    }


def train(X_tr, y_tr, X_te, y_te, seed, epsilon=None, args=None):
    """epsilon=None trains normally; otherwise trains with DP-SGD to that budget."""
    torch.manual_seed(seed)
    np.random.seed(seed)

    model = make_model(X_tr.shape[1])
    lr = args.lr if epsilon is None else args.dp_lr
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    loader = DataLoader(
        TensorDataset(torch.from_numpy(X_tr), torch.from_numpy(y_tr)),
        batch_size=args.batch_size, shuffle=True,
    )
    # Weight the minority class so the model can't just predict "patient" every time.
    pos_weight = torch.tensor((y_tr == 0).sum() / (y_tr == 1).sum())
    loss_fn = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

    noise = None
    if epsilon is not None:
        engine = PrivacyEngine(accountant="rdp")
        model, optimizer, loader = engine.make_private_with_epsilon(
            module=model,
            optimizer=optimizer,
            data_loader=loader,
            target_epsilon=epsilon,
            target_delta=args.delta,
            epochs=args.epochs,
            max_grad_norm=args.max_grad_norm,  # clip each patient's gradient
        )
        noise = optimizer.noise_multiplier

    for _ in range(args.epochs):
        model.train()
        for xb, yb in loader:
            if len(xb) == 0:  # Poisson sampling can produce an empty batch
                continue
            optimizer.zero_grad()
            loss_fn(model(xb).squeeze(1), yb).backward()
            optimizer.step()

    metrics = evaluate(model, X_te, y_te)
    metrics["noise_multiplier"] = noise
    if epsilon is not None:
        metrics["epsilon_spent"] = engine.get_epsilon(args.delta)
    return metrics


# ---------------------------------------------------------------- plot

def plot(summary, args):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    eps = args.epsilons
    dp = summary.loc[[f"DP eps={e:g}" for e in eps]]
    base = summary.loc["no privacy"]
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))
    for ax, metric, label in [
        (axes[0], "roc_auc", "ROC-AUC"),
        (axes[1], "balanced_accuracy", "Balanced accuracy"),
    ]:
        ax.errorbar(eps, dp[metric], yerr=dp[f"{metric}_sd"], marker="o",
                    capsize=3, color="#2a6fdb", label="With privacy (DP-SGD)")
        ax.axhline(base[metric], color="#1b9e5a", ls="--", label="No privacy")
        ax.axhline(0.5, color="#999999", ls=":", label="Random / majority guess")
        ax.set_xscale("log")
        ax.set_xticks(eps)
        ax.set_xticklabels([f"{e:g}" for e in eps])
        ax.set_xlabel("Privacy budget ε  (smaller = more private)")
        ax.set_ylabel(label)
        ax.set_ylim(0.4, 0.8)
        ax.grid(alpha=0.3)
    axes[0].legend(loc="lower right")
    fig.suptitle(f"Indian Liver Patient Dataset: privacy vs. usefulness "
                 f"(mean ± sd over {args.seeds} splits, δ={args.delta:g})")
    fig.tight_layout()
    fig.savefig(RESULTS / "privacy_tradeoff.png", dpi=150)


# ---------------------------------------------------------------- main

def main():
    p = argparse.ArgumentParser()
    # Settings below were tuned on seeds 10-29, which the final run (seeds 0..N-1) does not use.
    p.add_argument("--seeds", type=int, default=30)
    p.add_argument("--epochs", type=int, default=20)
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--lr", type=float, default=0.001)     # no-privacy model
    p.add_argument("--dp-lr", type=float, default=0.005)  # DP model
    p.add_argument("--max-grad-norm", type=float, default=1.0)
    p.add_argument("--delta", type=float, default=1e-3)  # below 1/466 training rows
    p.add_argument("--epsilons", type=float, nargs="+", default=[0.1, 0.25, 0.5, 1, 2, 4, 8, 16])
    args = p.parse_args()

    X, y = load_data()
    print(f"ILPD: {len(y)} patients, {X.shape[1]} features, "
          f"{y.mean():.1%} are liver patients\n")

    rows = []
    for seed in range(args.seeds):
        X_tr, X_te, y_tr, y_te = split(X, y, seed)

        majority = np.ones_like(y_te)
        rows.append({"setting": "majority-class guess", "epsilon": None, "seed": seed,
                     "accuracy": accuracy_score(y_te, majority),
                     "balanced_accuracy": 0.5, "roc_auc": 0.5})

        rows.append({"setting": "no privacy", "epsilon": None, "seed": seed,
                     **train(X_tr, y_tr, X_te, y_te, seed, None, args)})

        for eps in args.epsilons:
            rows.append({"setting": f"DP eps={eps:g}", "epsilon": eps, "seed": seed,
                         **train(X_tr, y_tr, X_te, y_te, seed, eps, args)})
        print(f"seed {seed} done")

    df = pd.DataFrame(rows)
    RESULTS.mkdir(exist_ok=True)
    df.to_csv(RESULTS / "all_runs.csv", index=False)

    order = ["majority-class guess", "no privacy"] + [f"DP eps={e:g}" for e in args.epsilons]
    summary = (
        df.groupby("setting", sort=False)
        .agg(accuracy=("accuracy", "mean"), accuracy_sd=("accuracy", "std"),
             balanced_accuracy=("balanced_accuracy", "mean"),
             balanced_accuracy_sd=("balanced_accuracy", "std"),
             roc_auc=("roc_auc", "mean"), roc_auc_sd=("roc_auc", "std"),
             noise_multiplier=("noise_multiplier", "mean"))
        .reindex(order)
    )
    summary.to_csv(RESULTS / "summary.csv")
    with open(RESULTS / "config.json", "w") as f:
        json.dump(vars(args), f, indent=2)

    print("\nMean over", args.seeds, "random train/test splits:\n")
    print(summary.round(3).to_string())
    plot(summary, args)


if __name__ == "__main__":
    main()
