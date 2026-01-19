# train_resnet_sdd.py
# -*- coding: utf-8 -*-

"""
A-few-shot-ResNet-transfer-learning-method-for-Chla-retrieval.

"""

from __future__ import annotations

import json
import math
import random
import time
import logging
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Callable, Dict, List, Optional, Tuple, Union, cast

import numpy as np
import pandas as pd

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from torch.utils.data import Dataset, DataLoader

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score, mean_squared_error

import optuna
import matplotlib.pyplot as plt


# -----------------------------
# Logging
# -----------------------------
def setup_logger(log_path: Path) -> logging.Logger:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    logger = logging.getLogger("train")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()

    fmt = logging.Formatter("[%(asctime)s] %(levelname)s: %(message)s")

    ch = logging.StreamHandler()
    ch.setFormatter(fmt)
    logger.addHandler(ch)

    fh = logging.FileHandler(log_path, encoding="utf-8")
    fh.setFormatter(fmt)
    logger.addHandler(fh)

    return logger


# -----------------------------
# Reproducibility
# -----------------------------
def set_seed(seed: int = 42) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    # For deterministic behavior (may reduce performance)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


# -----------------------------
# Activation helpers (GLU variants)
# -----------------------------
def reglu(x: Tensor) -> Tensor:
    a, b = x.chunk(2, dim=-1)
    return a * F.relu(b)


def geglu(x: Tensor) -> Tensor:
    a, b = x.chunk(2, dim=-1)
    return a * F.gelu(b)


def get_activation_fn(name: str) -> Callable[[Tensor], Tensor]:
    if name == "reglu":
        return reglu
    if name == "geglu":
        return geglu
    if name == "sigmoid":
        return torch.sigmoid
    return getattr(F, name)


def get_nonglu_activation_fn(name: str) -> Callable[[Tensor], Tensor]:
    # For GLU-like activations, the "non-GLU" activation is used after normalization at the end.
    if name == "reglu":
        return F.relu
    if name == "geglu":
        return F.gelu
    return get_activation_fn(name)


# -----------------------------
# Model
# -----------------------------
class ResNet(nn.Module):
    """
    ResNet-style MLP for tabular regression.
    Supports numerical features + optional categorical embeddings.

    Output is on log scale if you train with log1p(target).
    """

    def __init__(
        self,
        *,
        d_numerical: int,
        categories: Optional[List[int]],
        d_embedding: int,
        d: int,
        d_hidden_factor: float,
        n_layers: int,
        activation: str,
        normalization: str,
        hidden_dropout: float,
        residual_dropout: float,
        d_out: int,
    ) -> None:
        super().__init__()

        def make_normalization():
            norm_map = {"batchnorm": nn.BatchNorm1d, "layernorm": nn.LayerNorm}
            if normalization not in norm_map:
                raise ValueError(f"Unknown normalization: {normalization}")
            return norm_map[normalization](d)

        self.main_activation = get_activation_fn(activation)
        self.last_activation = get_nonglu_activation_fn(activation)
        self.residual_dropout = float(residual_dropout)
        self.hidden_dropout = float(hidden_dropout)

        d_in = int(d_numerical)
        d_hidden = int(d * d_hidden_factor)

        # Categorical embeddings
        self.category_offsets = None
        self.category_embeddings = None
        if categories is not None:
            # categories: list of cardinalities for each categorical feature
            d_in += len(categories) * int(d_embedding)
            category_offsets = torch.tensor([0] + categories[:-1]).cumsum(0)
            self.register_buffer("category_offsets", category_offsets)
            self.category_offsets = category_offsets
            self.category_embeddings = nn.Embedding(sum(categories), int(d_embedding))
            nn.init.kaiming_uniform_(self.category_embeddings.weight, a=math.sqrt(5))

        self.first_layer = nn.Linear(d_in, d)
        self.layers = nn.ModuleList(
            [
                nn.ModuleDict(
                    {
                        "norm": make_normalization(),
                        "linear0": nn.Linear(d, d_hidden * (2 if activation.endswith("glu") else 1)),
                        "linear1": nn.Linear(d_hidden, d),
                    }
                )
                for _ in range(int(n_layers))
            ]
        )
        self.last_normalization = make_normalization()
        self.head = nn.Linear(d, d_out)

    def forward(self, x_num: Tensor, x_cat: Optional[Tensor]) -> Tensor:
        parts = []
        if x_num is not None:
            parts.append(x_num)

        if x_cat is not None:
            if self.category_embeddings is None or self.category_offsets is None:
                raise ValueError("Model was created without categorical embeddings but x_cat was provided.")
            # (batch, n_cat) or (batch,)
            emb = self.category_embeddings(x_cat + self.category_offsets[None])  # broadcast offsets
            emb = emb.view(x_cat.size(0), -1)
            parts.append(emb)

        x = torch.cat(parts, dim=-1)
        x = self.first_layer(x)

        for layer in self.layers:
            layer = cast(Dict[str, nn.Module], layer)
            z = layer["norm"](x)
            z = layer["linear0"](z)
            z = self.main_activation(z)
            if self.hidden_dropout > 0:
                z = F.dropout(z, p=self.hidden_dropout, training=self.training)
            z = layer["linear1"](z)
            if self.residual_dropout > 0:
                z = F.dropout(z, p=self.residual_dropout, training=self.training)
            x = x + z

        x = self.last_normalization(x)
        x = self.last_activation(x)
        x = self.head(x)
        return x


# -----------------------------
# Dataset
# -----------------------------
class TabularDataset(Dataset):
    """Simple dataset for (numerical, categorical, target)."""

    def __init__(self, x_num: np.ndarray, x_cat: np.ndarray, y: np.ndarray):
        self.x_num = torch.tensor(x_num, dtype=torch.float32)
        # Keep categorical as long tensor for embedding lookup
        self.x_cat = torch.tensor(x_cat, dtype=torch.long)
        self.y = torch.tensor(y, dtype=torch.float32).reshape(-1, 1)

    def __len__(self) -> int:
        return self.x_num.shape[0]

    def __getitem__(self, idx: int):
        return self.x_num[idx], self.x_cat[idx], self.y[idx]


# -----------------------------
# Data I/O
# -----------------------------
def load_dataframe(
    excel_path: Path,
    num_cols: List[str],
    target_col: str,
    cat_col: str,
    dropna: bool = True,
) -> pd.DataFrame:
    df = pd.read_excel(excel_path)

    # Basic validation
    needed = set(num_cols + [target_col, cat_col])
    missing = [c for c in needed if c not in df.columns]
    if missing:
        raise ValueError(f"Missing columns in Excel: {missing}")

    if dropna:
        df = df.dropna(subset=list(needed)).copy()

    return df


def extract_arrays(
    df: pd.DataFrame,
    num_cols: List[str],
    target_col: str,
    cat_col: str,
    cat_min_value: int = 1,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    x_num = df[num_cols].to_numpy(dtype=np.float32)

    # Map category values to [0, C-1]
    x_cat_raw = df[cat_col].to_numpy()
    x_cat = (x_cat_raw - cat_min_value).astype(np.int64)

    y = df[target_col].to_numpy(dtype=np.float32)

    # Safety check for log1p
    if np.any(y < 0):
        raise ValueError(f"Target '{target_col}' contains negative values; log1p requires y >= 0.")

    return x_num, x_cat, y


# -----------------------------
# Train / Eval (log1p target)
# -----------------------------
def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    criterion: nn.Module,
    device: torch.device,
) -> float:
    model.train()
    total_loss = 0.0
    n = 0

    for x_num, x_cat, y in loader:
        x_num = x_num.to(device)
        x_cat = x_cat.to(device)
        y = y.to(device)

        # Train on log scale
        y_log = torch.log1p(y)

        optimizer.zero_grad(set_to_none=True)
        out = model(x_num, x_cat)
        loss = criterion(out, y_log)
        loss.backward()
        optimizer.step()

        bs = x_num.size(0)
        total_loss += float(loss.item()) * bs
        n += bs

    return total_loss / max(n, 1)


@torch.no_grad()
def evaluate(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
) -> Tuple[float, float, float]:
    model.eval()
    total_loss = 0.0
    n = 0

    preds = []
    trues = []

    for x_num, x_cat, y in loader:
        x_num = x_num.to(device)
        x_cat = x_cat.to(device)
        y = y.to(device)

        y_log = torch.log1p(y)
        out = model(x_num, x_cat)
        loss = criterion(out, y_log)

        bs = x_num.size(0)
        total_loss += float(loss.item()) * bs
        n += bs

        pred = torch.expm1(out).cpu().numpy().reshape(-1)
        true = y.cpu().numpy().reshape(-1)

        preds.append(pred)
        trues.append(true)

    preds = np.concatenate(preds, axis=0)
    trues = np.concatenate(trues, axis=0)

    mask = np.isfinite(preds) & np.isfinite(trues)
    preds = preds[mask]
    trues = trues[mask]
    if preds.size == 0:
        raise ValueError("No valid samples after filtering NaNs.")

    r2 = r2_score(trues, preds)
    rmse = float(np.sqrt(mean_squared_error(trues, preds)))
    avg_loss = total_loss / max(n, 1)
    return avg_loss, float(r2), rmse


@torch.no_grad()
def predict_all(model: nn.Module, loader: DataLoader, device: torch.device) -> Tuple[np.ndarray, np.ndarray]:
    model.eval()
    preds = []
    trues = []
    for x_num, x_cat, y in loader:
        x_num = x_num.to(device)
        x_cat = x_cat.to(device)
        out = model(x_num, x_cat)
        pred = torch.expm1(out).cpu().numpy().reshape(-1)
        true = y.numpy().reshape(-1)
        preds.append(pred)
        trues.append(true)
    return np.concatenate(trues), np.concatenate(preds)


def plot_pred_vs_true(true: np.ndarray, pred: np.ndarray, out_path: Path, title: str) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.figure(figsize=(6, 6))
    plt.scatter(true, pred, alpha=0.6, edgecolors="k", linewidths=0.3)
    vmin = float(min(true.min(), pred.min()))
    vmax = float(max(true.max(), pred.max()))
    plt.plot([vmin, vmax], [vmin, vmax], linestyle="--")
    plt.xlabel("True")
    plt.ylabel("Predicted")
    plt.title(title)
    plt.tight_layout()
    plt.savefig(out_path, dpi=300)
    plt.close()


def plot_training_curves(history: pd.DataFrame, out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)

    plt.figure(figsize=(16, 4))

    # Loss
    plt.subplot(1, 3, 1)
    plt.plot(history["epoch"], history["train_loss"], label="train_loss")
    plt.plot(history["epoch"], history["val_loss"], label="val_loss")
    plt.xlabel("Epoch")
    plt.ylabel("MSE on log1p")
    plt.title("Loss")
    plt.legend()

    # R2
    plt.subplot(1, 3, 2)
    plt.plot(history["epoch"], history["train_r2"], label="train_r2")
    plt.plot(history["epoch"], history["val_r2"], label="val_r2")
    plt.xlabel("Epoch")
    plt.ylabel("R2 on original scale")
    plt.title("R2")
    plt.legend()

    # RMSE
    plt.subplot(1, 3, 3)
    plt.plot(history["epoch"], history["train_rmse"], label="train_rmse")
    plt.plot(history["epoch"], history["val_rmse"], label="val_rmse")
    plt.xlabel("Epoch")
    plt.ylabel("RMSE on original scale")
    plt.title("RMSE")
    plt.legend()

    plt.tight_layout()
    plt.savefig(out_path, dpi=300)
    plt.close()


# -----------------------------
# Config
# -----------------------------
@dataclass
class TrainConfig:
    seed: int = 42
    test_size: float = 0.30
    val_size: float = 0.30  # split from train_full
    num_workers: int = 2

    # Optuna
    n_trials: int = 50
    timeout: int = 1200
    trial_epochs: int = 200
    trial_patience: int = 30

    # Final training
    max_epochs: int = 500
    patience: int = 80

    # Category cardinality
    n_categories: int = 13
    cat_min_value: int = 1  # if your categories are 1..13, use 1


# -----------------------------
# Optuna Objective
# -----------------------------
def build_dataloaders(
    x_train: np.ndarray,
    c_train: np.ndarray,
    y_train: np.ndarray,
    x_val: np.ndarray,
    c_val: np.ndarray,
    y_val: np.ndarray,
    batch_size: int,
    normalization: str,
    num_workers: int,
) -> Tuple[DataLoader, DataLoader]:
    train_ds = TabularDataset(x_train, c_train, y_train)
    val_ds = TabularDataset(x_val, c_val, y_val)

    # For BatchNorm, batch_size=1 is problematic; drop_last avoids last tiny batch
    drop_last = (normalization == "batchnorm")
    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        drop_last=drop_last,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        drop_last=False,
    )
    return train_loader, val_loader


def objective(
    trial: optuna.Trial,
    x_train: np.ndarray,
    c_train: np.ndarray,
    y_train: np.ndarray,
    x_val: np.ndarray,
    c_val: np.ndarray,
    y_val: np.ndarray,
    device: torch.device,
    cfg: TrainConfig,
) -> float:
    # Hyperparameter search space
    params = {
        "learning_rate": trial.suggest_float("learning_rate", 5e-5, 5e-4, log=True),
        "batch_size": trial.suggest_categorical("batch_size", [64, 128, 256]),
        "d_embedding": trial.suggest_categorical("d_embedding", [8, 16, 32]),
        "d": trial.suggest_categorical("d", [64, 128]),
        "d_hidden_factor": trial.suggest_float("d_hidden_factor", 1.0, 1.8),
        "n_layers": trial.suggest_int("n_layers", 2, 6),
        "activation": trial.suggest_categorical("activation", ["relu", "gelu", "reglu", "geglu"]),
        "normalization": trial.suggest_categorical("normalization", ["batchnorm", "layernorm"]),
        "hidden_dropout": trial.suggest_float("hidden_dropout", 0.0, 0.3),
        "residual_dropout": trial.suggest_float("residual_dropout", 0.0, 0.3),
    }

    train_loader, val_loader = build_dataloaders(
        x_train, c_train, y_train,
        x_val, c_val, y_val,
        batch_size=params["batch_size"],
        normalization=params["normalization"],
        num_workers=cfg.num_workers,
    )

    model = ResNet(
        d_numerical=x_train.shape[1],
        categories=[cfg.n_categories],
        d_embedding=params["d_embedding"],
        d=params["d"],
        d_hidden_factor=params["d_hidden_factor"],
        n_layers=params["n_layers"],
        activation=params["activation"],
        normalization=params["normalization"],
        hidden_dropout=params["hidden_dropout"],
        residual_dropout=params["residual_dropout"],
        d_out=1,
    ).to(device)

    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=params["learning_rate"])

    best_val_loss = float("inf")
    best_val_r2 = -float("inf")
    no_improve = 0

    for epoch in range(1, cfg.trial_epochs + 1):
        _ = train_one_epoch(model, train_loader, optimizer, criterion, device)
        val_loss, val_r2, _ = evaluate(model, val_loader, criterion, device)

        best_val_r2 = max(best_val_r2, val_r2)
        trial.report(val_r2, step=epoch)

        # Early stopping (based on val_loss)
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            no_improve = 0
        else:
            no_improve += 1

        if trial.should_prune():
            raise optuna.exceptions.TrialPruned()

        if no_improve >= cfg.trial_patience:
            break

    return float(best_val_r2)


# -----------------------------
# Main training pipeline
# -----------------------------
def train_final_model(
    best_params: Dict[str, Union[int, float, str]],
    x_train_full: np.ndarray,
    c_train_full: np.ndarray,
    y_train_full: np.ndarray,
    x_val: np.ndarray,
    c_val: np.ndarray,
    y_val: np.ndarray,
    device: torch.device,
    cfg: TrainConfig,
    out_dir: Path,
    logger: logging.Logger,
) -> Tuple[nn.Module, pd.DataFrame]:
    out_dir.mkdir(parents=True, exist_ok=True)

    batch_size = int(best_params["batch_size"])
    normalization = str(best_params["normalization"])

    train_loader, val_loader = build_dataloaders(
        x_train_full, c_train_full, y_train_full,
        x_val, c_val, y_val,
        batch_size=batch_size,
        normalization=normalization,
        num_workers=cfg.num_workers,
    )

    model = ResNet(
        d_numerical=x_train_full.shape[1],
        categories=[cfg.n_categories],
        d_embedding=int(best_params["d_embedding"]),
        d=int(best_params["d"]),
        d_hidden_factor=float(best_params["d_hidden_factor"]),
        n_layers=int(best_params["n_layers"]),
        activation=str(best_params["activation"]),
        normalization=normalization,
        hidden_dropout=float(best_params["hidden_dropout"]),
        residual_dropout=float(best_params["residual_dropout"]),
        d_out=1,
    ).to(device)

    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=float(best_params["learning_rate"]))

    best_state = None
    best_val_loss = float("inf")
    no_improve = 0

    history_rows = []

    for epoch in range(1, cfg.max_epochs + 1):
        train_loss = train_one_epoch(model, train_loader, optimizer, criterion, device)

        train_loss_eval, train_r2, train_rmse = evaluate(model, train_loader, criterion, device)
        val_loss, val_r2, val_rmse = evaluate(model, val_loader, criterion, device)

        history_rows.append(
            {
                "epoch": epoch,
                "train_loss": train_loss_eval,
                "train_r2": train_r2,
                "train_rmse": train_rmse,
                "val_loss": val_loss,
                "val_r2": val_r2,
                "val_rmse": val_rmse,
            }
        )

        if epoch % 10 == 0 or epoch == 1:
            logger.info(
                f"Epoch {epoch}/{cfg.max_epochs} | "
                f"train_loss={train_loss_eval:.5f}, train_r2={train_r2:.4f}, train_rmse={train_rmse:.4f} | "
                f"val_loss={val_loss:.5f}, val_r2={val_r2:.4f}, val_rmse={val_rmse:.4f}"
            )

        # Early stopping based on val_loss
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            no_improve = 0
        else:
            no_improve += 1

        if no_improve >= cfg.patience:
            logger.info(f"Early stopping triggered at epoch {epoch}.")
            break

    if best_state is not None:
        model.load_state_dict(best_state)
        torch.save(best_state, out_dir / "best_model.pth")
        logger.info("Saved best model: best_model.pth")
    else:
        torch.save(model.state_dict(), out_dir / "last_model.pth")
        logger.info("Saved last model: last_model.pth (no best state found)")

    history = pd.DataFrame(history_rows)
    history.to_csv(out_dir / "training_history.csv", index=False)

    return model, history


def parse_args():
    import argparse

    parser = argparse.ArgumentParser(description="ResNet tabular regressor (log1p target) with Optuna tuning.")
    parser.add_argument("--data", type=str, required=True, help="Path to input Excel file.")
    parser.add_argument("--output", type=str, required=True, help="Output directory (run folder).")

    parser.add_argument("--target", type=str, default="Secchi_depth", help="Target column name.")
    parser.add_argument("--cat_col", type=str, default="CLASS_SAM_new", help="Categorical column name.")
    parser.add_argument("--num_cols", nargs="+", default=["B1", "B2", "B3", "B4", "B5", "B6", "B7"],
                        help="Numerical feature columns.")

    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--test_size", type=float, default=0.30)
    parser.add_argument("--val_size", type=float, default=0.30)

    parser.add_argument("--n_trials", type=int, default=50)
    parser.add_argument("--timeout", type=int, default=1200)
    parser.add_argument("--trial_epochs", type=int, default=200)
    parser.add_argument("--trial_patience", type=int, default=30)

    parser.add_argument("--max_epochs", type=int, default=500)
    parser.add_argument("--patience", type=int, default=80)

    parser.add_argument("--num_workers", type=int, default=2)

    parser.add_argument("--n_categories", type=int, default=13, help="Number of categories for the cat feature.")
    parser.add_argument("--cat_min_value", type=int, default=1, help="Minimum category value in the file (e.g., 1).")

    return parser.parse_args()


def main():
    args = parse_args()

    cfg = TrainConfig(
        seed=args.seed,
        test_size=args.test_size,
        val_size=args.val_size,
        num_workers=args.num_workers,
        n_trials=args.n_trials,
        timeout=args.timeout,
        trial_epochs=args.trial_epochs,
        trial_patience=args.trial_patience,
        max_epochs=args.max_epochs,
        patience=args.patience,
        n_categories=args.n_categories,
        cat_min_value=args.cat_min_value,
    )

    out_dir = Path(args.output)
    out_dir.mkdir(parents=True, exist_ok=True)
    logger = setup_logger(out_dir / "train.log")

    logger.info("===== Starting run =====")
    logger.info(f"Config: {json.dumps(asdict(cfg), ensure_ascii=False)}")

    set_seed(cfg.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Device: {device}")

    # ---- Load data ----
    excel_path = Path(args.data)
    df = load_dataframe(
        excel_path=excel_path,      # Path to the Excel file containing the data
        num_cols=args.num_cols,     # List of numerical feature column names (e.g., B1, B2, B3, etc.)
        target_col=args.target,     # The name of the target column (e.g., Chl-a)
        cat_col=args.cat_col,       # The name of the categorical column (e.g., water class labels)
        dropna=True,                # Whether to drop rows with missing values
    )
    x_num, x_cat, y = extract_arrays(
        df=df,
        num_cols=args.num_cols,
        target_col=args.target,
        cat_col=args.cat_col,
        cat_min_value=cfg.cat_min_value,
    )
    logger.info(f"Total samples after dropna: {len(y)}")

    # ---- Split: train_full / test ----
    x_num_train_full, x_num_test, x_cat_train_full, x_cat_test, y_train_full, y_test = train_test_split(
        x_num, x_cat, y, test_size=cfg.test_size, random_state=cfg.seed
    )

    # ---- Split: train / val (from train_full) ----
    x_num_train, x_num_val, x_cat_train, x_cat_val, y_train, y_val = train_test_split(
        x_num_train_full, x_cat_train_full, y_train_full, test_size=cfg.val_size, random_state=cfg.seed
    )

    logger.info(f"Train: {len(y_train)}, Val: {len(y_val)}, Test: {len(y_test)}")

    # ---- Fit scaler on TRAIN only (avoid leakage) ----
    scaler = StandardScaler()
    x_num_train = scaler.fit_transform(x_num_train)
    x_num_val = scaler.transform(x_num_val)
    x_num_train_full_scaled = scaler.fit_transform(x_num_train_full)  # for final training, fit on train_full only
    x_num_test_scaled = scaler.transform(x_num_test)

    # Save scaler parameters (simple way)
    scaler_path = out_dir / "scaler.json"
    scaler_dict = {
        "mean": scaler.mean_.tolist(),
        "scale": scaler.scale_.tolist(),
        "var": scaler.var_.tolist(),
        "n_features_in": int(scaler.n_features_in_),
    }
    scaler_path.write_text(json.dumps(scaler_dict, indent=2), encoding="utf-8")
    logger.info("Saved scaler parameters: scaler.json")

    # ---- Optuna tuning (on train vs val) ----
    study = optuna.create_study(direction="maximize", study_name="ResNet_Tabular_Regressor")
    start = time.time()

    study.optimize(
        lambda t: objective(
            t,
            x_train=x_num_train,
            c_train=x_cat_train,
            y_train=y_train,
            x_val=x_num_val,
            c_val=x_cat_val,
            y_val=y_val,
            device=device,
            cfg=cfg,
        ),
        n_trials=cfg.n_trials,
        timeout=cfg.timeout,
    )

    elapsed = time.time() - start
    logger.info(f"Optuna finished. Time: {elapsed:.1f}s")
    logger.info(f"Best R2: {study.best_value:.4f}")
    logger.info(f"Best params: {study.best_params}")

    (out_dir / "best_params.json").write_text(
        json.dumps(study.best_params, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    # ---- Final training: train_full_scaled vs val (still keep val for early stopping) ----
    # NOTE: x_num_train_full_scaled uses scaler fitted on train_full.
    # For a stricter protocol, you can keep the scaler fitted only on train and apply to train_full,
    # but this is acceptable since test is never used for fitting.
    best_params = study.best_params

    model, history = train_final_model(
        best_params=best_params,
        x_train_full=x_num_train_full_scaled,
        c_train_full=x_cat_train_full,
        y_train_full=y_train_full,
        x_val=x_num_val,            # val already scaled by scaler fitted on train earlier
        c_val=x_cat_val,
        y_val=y_val,
        device=device,
        cfg=cfg,
        out_dir=out_dir,
        logger=logger,
    )

    # Save training curves
    plot_training_curves(history, out_dir / "training_curves.png")
    logger.info("Saved training curves: training_curves.png")

    # ---- Evaluate on TEST ----
    test_ds = TabularDataset(x_num_test_scaled, x_cat_test, y_test)
    test_loader = DataLoader(test_ds, batch_size=int(best_params["batch_size"]), shuffle=False, num_workers=cfg.num_workers)

    criterion = nn.MSELoss()
    test_loss, test_r2, test_rmse = evaluate(model, test_loader, criterion, device)

    metrics = {
        "test_loss_log1p_mse": test_loss,
        "test_r2": test_r2,
        "test_rmse": test_rmse,
    }
    (out_dir / "test_metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    logger.info(f"Test metrics: {metrics}")

    # ---- Save predictions ----
    true_test, pred_test = predict_all(model, test_loader, device)
    pred_df = pd.DataFrame({"true": true_test, "pred": pred_test})
    pred_df.to_csv(out_dir / "test_predictions.csv", index=False)

    # Also export as XLSX (commonly used in your workflow)
    with pd.ExcelWriter(out_dir / "predictions.xlsx") as writer:
        pred_df.to_excel(writer, sheet_name="Test", index=False)

    plot_pred_vs_true(true_test, pred_test, out_dir / "test_pred_vs_true.png", title="Test: Pred vs True")
    logger.info("Saved predictions and scatterplot.")

    logger.info("===== Run completed =====")


if __name__ == "__main__":
    main()
