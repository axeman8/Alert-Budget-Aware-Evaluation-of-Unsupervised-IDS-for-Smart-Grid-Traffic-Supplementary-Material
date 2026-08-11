#!/usr/bin/env python3
"""Camera-ready multi-seed extension for ANUBIS paper 1316.

Purpose
-------
Re-run the five accepted-paper unsupervised IDS detectors across pre-specified
random seeds while preserving the accepted file-disjoint split and the rule
that thresholds are derived only from normal validation scores.

Outputs include per-seed metrics, across-seed summaries with 95% t intervals,
and raw window-level false-trigger rates/day for 100 ms windows.

This script intentionally does NOT implement an SCD/specification baseline:
the supplied artifact contains only the pre-extracted 100 ms feature table and
not the SCD file or raw packet-level fields required for a faithful baseline.
"""

from __future__ import annotations

import argparse
import gc
import json
import os
import platform
import random
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import scipy
from scipy.stats import t as student_t
import sklearn
from sklearn.covariance import EmpiricalCovariance
from sklearn.ensemble import IsolationForest
from sklearn.kernel_approximation import Nystroem
from sklearn.linear_model import SGDOneClassSVM
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.preprocessing import MinMaxScaler, RobustScaler

import tensorflow as tf
from tensorflow.keras import callbacks, layers, optimizers, regularizers
from tensorflow.keras.models import Model


NORMAL_LABEL = 0
ATTACK_LABEL = 1
WINDOW_SECONDS = 0.1
WINDOWS_PER_DAY = 86400.0 / WINDOW_SECONDS
REFERENCE_PERCENTILE = 95.0
LSTM_SEQ_LEN = 20
MAX_ATTACK_SAMPLES_PER_FILE = 100000

METADATA_COLS = ["pcap_file", "window_id", "win_start_epoch"]
TRAIN_FILES = (
    [f"Baseline-{str(i).zfill(3)}.pcapng" for i in range(1, 17)]
    + [f"Baseline-30min-{str(i).zfill(3)}.pcapng" for i in range(1, 7)]
)
VAL_FILES = [f"Baseline-{str(i).zfill(3)}.pcapng" for i in range(17, 21)]
TEST_FILES = [
    "Baseline-30min-007.pcapng",
    "Baseline-60min-001.pcapng",
    "Baseline-60min-002.pcapng",
]

# Pre-specified consecutive seeds retain the accepted-paper seed (67) while
# avoiding post-hoc seed selection.
DEFAULT_SEEDS = list(range(67, 77))

# Fractions, not percentages. These include the accepted-paper budgets plus a
# low-FPR stress-test range motivated by the reviewer comment on raw alert load.
DEFAULT_BUDGETS = [
    0.0001,  # 0.01%
    0.0002,  # 0.02%
    0.0005,  # 0.05%
    0.0010,  # 0.10%
    0.0020,  # 0.20%
    0.0050,  # 0.50%
    0.0100,  # 1%
    0.0200,  # 2%
    0.0500,  # 5%
    0.1000,  # 10%
]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--data", default="all.parquet", help="Path to all.parquet")
    p.add_argument("--out", default="camera_ready_multiseed_results", help="Output directory")
    p.add_argument(
        "--seeds",
        default=",".join(map(str, DEFAULT_SEEDS)),
        help="Comma-separated integer seeds (default: 67..76)",
    )
    p.add_argument(
        "--budgets",
        default=",".join(map(str, DEFAULT_BUDGETS)),
        help="Comma-separated FPR-budget fractions, e.g. 0.0001,0.001,0.01",
    )
    p.add_argument("--skip-lstm", action="store_true")
    p.add_argument("--skip-hybrid", action="store_true")
    p.add_argument("--epochs-dense", type=int, default=100)
    p.add_argument("--epochs-lstm", type=int, default=100)
    p.add_argument("--epochs-hybrid", type=int, default=150)
    return p.parse_args()


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    try:
        tf.keras.utils.set_random_seed(seed)
    except Exception:
        tf.random.set_seed(seed)
    try:
        tf.config.experimental.enable_op_determinism()
    except Exception:
        pass


def reconstruction_mae(model: Model, data: np.ndarray, batch_size: int = 2048) -> np.ndarray:
    recon = model.predict(data, batch_size=batch_size, verbose=0)
    return np.mean(np.abs(recon - data), axis=1)


def robust_stats(x: np.ndarray) -> tuple[float, float]:
    med = float(np.median(x))
    mad = float(np.median(np.abs(x - med)))
    return med, max(mad, 1e-6)


def robust_zscore(x: np.ndarray, med: float, mad: float) -> np.ndarray:
    return 0.6745 * (x - med) / mad


def operational_metrics(
    normal_scores: np.ndarray,
    dos_scores: np.ndarray,
    fdi_scores: np.ndarray,
    threshold: float,
) -> dict:
    attack_scores = np.concatenate([dos_scores, fdi_scores])
    scores = np.concatenate([normal_scores, attack_scores])
    labels = np.concatenate(
        [
            np.full(len(normal_scores), NORMAL_LABEL, dtype=int),
            np.full(len(attack_scores), ATTACK_LABEL, dtype=int),
        ]
    )
    preds = (scores > threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(labels, preds, labels=[NORMAL_LABEL, ATTACK_LABEL]).ravel()
    fpr = fp / (tn + fp) if (tn + fp) else np.nan
    false_triggers_per_day = fpr * WINDOWS_PER_DAY if np.isfinite(fpr) else np.nan
    seconds_between = (WINDOW_SECONDS / fpr) if fpr and fpr > 0 else np.inf
    return {
        "threshold": float(threshold),
        "accuracy": accuracy_score(labels, preds),
        "attack_precision": precision_score(labels, preds, pos_label=ATTACK_LABEL, zero_division=0),
        "attack_recall": recall_score(labels, preds, pos_label=ATTACK_LABEL, zero_division=0),
        "attack_f1": f1_score(labels, preds, pos_label=ATTACK_LABEL, zero_division=0),
        "roc_auc": roc_auc_score(labels, scores),
        "pr_auc": average_precision_score(labels, scores),
        "fpr": fpr,
        "fp": int(fp),
        "tn": int(tn),
        "fn": int(fn),
        "tp": int(tp),
        "normal_windows": int(tn + fp),
        "attack_windows": int(fn + tp),
        "dos_recall": float(np.mean(dos_scores > threshold)),
        "fdi_recall": float(np.mean(fdi_scores > threshold)),
        "alerts_per_1k_normals": fpr * 1000.0,
        "raw_false_triggers_per_day": false_triggers_per_day,
        "mean_seconds_between_raw_false_triggers": seconds_between,
    }


def evaluate_budgets(
    model_name: str,
    seed: int,
    val_scores: np.ndarray,
    test_scores: np.ndarray,
    dos_scores: np.ndarray,
    fdi_scores: np.ndarray,
    budgets: list[float],
) -> pd.DataFrame:
    rows = []
    for budget in budgets:
        pct = 100.0 * (1.0 - budget)
        threshold = float(np.percentile(val_scores, pct))
        m = operational_metrics(test_scores, dos_scores, fdi_scores, threshold)
        m.update(
            {
                "model": model_name,
                "seed": seed,
                "target_fpr_budget": budget,
                "target_fpr_budget_pct": budget * 100.0,
                "threshold_percentile": pct,
                "validation_windows": len(val_scores),
                "expected_validation_exceedances": budget * len(val_scores),
            }
        )
        rows.append(m)
    return pd.DataFrame(rows)


def reference_row(
    model_name: str,
    seed: int,
    val_scores: np.ndarray,
    test_scores: np.ndarray,
    dos_scores: np.ndarray,
    fdi_scores: np.ndarray,
) -> dict:
    threshold = float(np.percentile(val_scores, REFERENCE_PERCENTILE))
    m = operational_metrics(test_scores, dos_scores, fdi_scores, threshold)
    m.update(
        {
            "model": model_name,
            "seed": seed,
            "reference_percentile": REFERENCE_PERCENTILE,
            "validation_windows": len(val_scores),
        }
    )
    return m


def make_sequences_from_array(arr: np.ndarray, seq_len: int) -> tuple[np.ndarray, int]:
    n = (len(arr) // seq_len) * seq_len
    if n == 0:
        return np.empty((0, seq_len, arr.shape[1]), dtype=arr.dtype), 0
    dropped = len(arr) - n
    return arr[:n].reshape(-1, seq_len, arr.shape[1]), dropped


def prepare_file_sequences(
    all_df: pd.DataFrame,
    feat_cols: list[str],
    file_list: list[str],
    scaler,
    seq_len: int,
) -> np.ndarray:
    seqs = []
    for fname in file_list:
        grp = all_df[all_df["pcap_file"] == fname].sort_values("window_id")
        raw = np.nan_to_num(grp[feat_cols].values, nan=0.0, posinf=0.0, neginf=0.0)
        scaled = scaler.transform(raw).astype(np.float32)
        file_seqs, _ = make_sequences_from_array(scaled, seq_len)
        if len(file_seqs):
            seqs.append(file_seqs)
    if not seqs:
        return np.empty((0, seq_len, len(feat_cols)), dtype=np.float32)
    return np.concatenate(seqs, axis=0)


def prepare_attack_arrays(
    all_df: pd.DataFrame,
    feat_cols: list[str],
    prefix: str,
    scaler,
) -> list[np.ndarray]:
    arrays = []
    attack_df = all_df[all_df["pcap_file"].str.startswith(prefix)]
    for _, grp in sorted(attack_df.groupby("pcap_file"), key=lambda x: x[0]):
        raw = np.nan_to_num(grp[feat_cols].values, nan=0.0, posinf=0.0, neginf=0.0)
        # Preserve accepted-artifact cap semantics. Current attack files are
        # expected to be below this cap; if not, take a deterministic prefix
        # rather than introduce seed-dependent sampling into the test set.
        raw = raw[:MAX_ATTACK_SAMPLES_PER_FILE]
        arrays.append(scaler.transform(raw).astype(np.float32))
    return arrays


def concat_scores(arrays: list[np.ndarray], score_fn) -> np.ndarray:
    return np.concatenate([np.asarray(score_fn(x), dtype=float).reshape(-1) for x in arrays])


class DenseReferenceAE(Model):
    def __init__(self, n_input: int):
        super().__init__()
        self.encoder = tf.keras.Sequential(
            [
                layers.Dense(32, activation="relu"),
                layers.Dense(16, activation="relu"),
                layers.Dense(8, activation="relu"),
            ]
        )
        self.decoder = tf.keras.Sequential(
            [
                layers.Dense(16, activation="relu"),
                layers.Dense(32, activation="relu"),
                layers.Dense(n_input, activation="sigmoid"),
            ]
        )

    def call(self, x):
        return self.decoder(self.encoder(x))


class HybridAblationAE(Model):
    LATENT_DIM = 8
    NOISE_STD = 0.02
    DROPOUT_RATE = 0.10
    L2_REG = 1e-5
    L1_ACTIVITY = 1e-5

    def __init__(self, n_input: int):
        super().__init__()
        self.noise = layers.GaussianNoise(self.NOISE_STD)
        self.encoder = tf.keras.Sequential(
            [
                layers.Dense(64, kernel_regularizer=regularizers.l2(self.L2_REG)),
                layers.BatchNormalization(),
                layers.Activation("relu"),
                layers.Dropout(self.DROPOUT_RATE),
                layers.Dense(32, kernel_regularizer=regularizers.l2(self.L2_REG)),
                layers.BatchNormalization(),
                layers.Activation("relu"),
                layers.Dense(
                    self.LATENT_DIM,
                    activation=None,
                    activity_regularizer=regularizers.l1(self.L1_ACTIVITY),
                    name="latent",
                ),
            ]
        )
        self.decoder = tf.keras.Sequential(
            [
                layers.Dense(32, kernel_regularizer=regularizers.l2(self.L2_REG)),
                layers.BatchNormalization(),
                layers.Activation("relu"),
                layers.Dense(64, kernel_regularizer=regularizers.l2(self.L2_REG)),
                layers.BatchNormalization(),
                layers.Activation("relu"),
                layers.Dense(n_input, activation="linear"),
            ]
        )

    def call(self, x, training=False):
        z = self.encoder(self.noise(x, training=training), training=training)
        return self.decoder(z, training=training)

    def encode(self, x, training=False):
        return self.encoder(x, training=training)


def make_lstm_ae(n_input: int) -> Model:
    inp = tf.keras.Input(shape=(LSTM_SEQ_LEN, n_input))
    x = layers.LSTM(32, return_sequences=True)(inp)
    enc = layers.LSTM(16)(x)
    x = layers.RepeatVector(LSTM_SEQ_LEN)(enc)
    x = layers.LSTM(16, return_sequences=True)(x)
    x = layers.LSTM(32, return_sequences=True)(x)
    out = layers.TimeDistributed(layers.Dense(n_input, activation="sigmoid"))(x)
    model = Model(inp, out, name="lstm_autoencoder")
    model.compile(optimizer="adam", loss="mae")
    return model


def summarise_across_seeds(df: pd.DataFrame, group_cols: list[str]) -> pd.DataFrame:
    metric_cols = [
        "attack_precision",
        "attack_recall",
        "attack_f1",
        "fpr",
        "alerts_per_1k_normals",
        "raw_false_triggers_per_day",
        "mean_seconds_between_raw_false_triggers",
        "dos_recall",
        "fdi_recall",
        "roc_auc",
        "pr_auc",
    ]
    rows = []
    for keys, grp in df.groupby(group_cols, dropna=False):
        if not isinstance(keys, tuple):
            keys = (keys,)
        row = dict(zip(group_cols, keys))
        n = grp["seed"].nunique()
        row["n_seeds"] = n
        for col in metric_cols:
            x = grp[col].replace([np.inf, -np.inf], np.nan).dropna().to_numpy(dtype=float)
            if len(x) == 0:
                mean = sd = lo = hi = np.nan
            else:
                mean = float(np.mean(x))
                sd = float(np.std(x, ddof=1)) if len(x) > 1 else 0.0
                if len(x) > 1:
                    half = float(student_t.ppf(0.975, len(x) - 1) * sd / np.sqrt(len(x)))
                    lo, hi = mean - half, mean + half
                else:
                    lo = hi = mean
            row[f"{col}_mean"] = mean
            row[f"{col}_sd"] = sd
            row[f"{col}_ci95_low"] = lo
            row[f"{col}_ci95_high"] = hi
        rows.append(row)
    return pd.DataFrame(rows)


def run() -> None:
    args = parse_args()
    seeds = [int(x.strip()) for x in args.seeds.split(",") if x.strip()]
    budgets = [float(x.strip()) for x in args.budgets.split(",") if x.strip()]
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    data_path = Path(args.data)
    print(f"Loading {data_path} ...")
    all_df = pd.read_parquet(data_path)
    feat_cols = [c for c in all_df.columns if c not in METADATA_COLS]

    def split_raw(files: list[str]) -> np.ndarray:
        grp = all_df[all_df["pcap_file"].isin(files)].sort_values(["pcap_file", "window_id"])
        return np.nan_to_num(grp[feat_cols].values, nan=0.0, posinf=0.0, neginf=0.0)

    train_raw = split_raw(TRAIN_FILES)
    val_raw = split_raw(VAL_FILES)
    test_raw = split_raw(TEST_FILES)

    scaler_mm = MinMaxScaler().fit(train_raw)
    scaler_rb = RobustScaler(quantile_range=(5, 95)).fit(train_raw)
    train_mm = scaler_mm.transform(train_raw).astype(np.float32)
    val_mm = scaler_mm.transform(val_raw).astype(np.float32)
    test_mm = scaler_mm.transform(test_raw).astype(np.float32)
    train_rb = scaler_rb.transform(train_raw).astype(np.float32)
    val_rb = scaler_rb.transform(val_raw).astype(np.float32)
    test_rb = scaler_rb.transform(test_raw).astype(np.float32)

    dos_mm = prepare_attack_arrays(all_df, feat_cols, "DOS", scaler_mm)
    fdi_mm = prepare_attack_arrays(all_df, feat_cols, "FDI", scaler_mm)
    dos_rb = prepare_attack_arrays(all_df, feat_cols, "DOS", scaler_rb)
    fdi_rb = prepare_attack_arrays(all_df, feat_cols, "FDI", scaler_rb)

    # Seed-independent temporal partitions.
    if not args.skip_lstm:
        train_seq = prepare_file_sequences(all_df, feat_cols, TRAIN_FILES, scaler_mm, LSTM_SEQ_LEN)
        val_seq = prepare_file_sequences(all_df, feat_cols, VAL_FILES, scaler_mm, LSTM_SEQ_LEN)
        test_seq = prepare_file_sequences(all_df, feat_cols, TEST_FILES, scaler_mm, LSTM_SEQ_LEN)
        dos_seq_arrays = []
        for arr in dos_mm:
            seqs, _ = make_sequences_from_array(arr, LSTM_SEQ_LEN)
            if len(seqs):
                dos_seq_arrays.append(seqs)
        fdi_seq_arrays = []
        for arr in fdi_mm:
            seqs, _ = make_sequences_from_array(arr, LSTM_SEQ_LEN)
            if len(seqs):
                fdi_seq_arrays.append(seqs)
    else:
        train_seq = val_seq = test_seq = None
        dos_seq_arrays = fdi_seq_arrays = []

    per_seed_ref = []
    per_seed_budget = []
    timing_rows = []
    n_input = len(feat_cols)

    for seed in seeds:
        print(f"\n===== SEED {seed} =====")
        set_seed(seed)

        # Dense AE
        tf.keras.backend.clear_session()
        set_seed(seed)
        t0 = time.time()
        dense = DenseReferenceAE(n_input)
        dense.compile(optimizer="adam", loss="mae")
        dense.fit(
            train_mm,
            train_mm,
            epochs=args.epochs_dense,
            batch_size=512,
            validation_data=(val_mm, val_mm),
            callbacks=[callbacks.EarlyStopping(monitor="val_loss", patience=5, restore_best_weights=True)],
            verbose=0,
        )
        dense_val = reconstruction_mae(dense, val_mm)
        dense_test = reconstruction_mae(dense, test_mm)
        dense_dos = concat_scores(dos_mm, lambda x: reconstruction_mae(dense, x))
        dense_fdi = concat_scores(fdi_mm, lambda x: reconstruction_mae(dense, x))
        per_seed_ref.append(reference_row("Dense AE", seed, dense_val, dense_test, dense_dos, dense_fdi))
        per_seed_budget.append(evaluate_budgets("Dense AE", seed, dense_val, dense_test, dense_dos, dense_fdi, budgets))
        timing_rows.append({"seed": seed, "model": "Dense AE", "seconds": time.time() - t0})
        del dense
        gc.collect()

        # Isolation Forest
        t0 = time.time()
        iso = IsolationForest(n_estimators=100, contamination=0.05, random_state=seed, n_jobs=-1)
        iso.fit(train_mm)
        iso_val = -iso.decision_function(val_mm)
        iso_test = -iso.decision_function(test_mm)
        iso_dos = concat_scores(dos_mm, lambda x: -iso.decision_function(x))
        iso_fdi = concat_scores(fdi_mm, lambda x: -iso.decision_function(x))
        per_seed_ref.append(reference_row("Isolation Forest", seed, iso_val, iso_test, iso_dos, iso_fdi))
        per_seed_budget.append(evaluate_budgets("Isolation Forest", seed, iso_val, iso_test, iso_dos, iso_fdi, budgets))
        timing_rows.append({"seed": seed, "model": "Isolation Forest", "seconds": time.time() - t0})
        del iso
        gc.collect()

        # Nystroem + SGD One-Class SVM
        t0 = time.time()
        gamma_scale = 1.0 / (train_mm.shape[1] * train_mm.var())
        fmap = Nystroem(gamma=gamma_scale, n_components=100, random_state=seed)
        train_mapped = fmap.fit_transform(train_mm)
        svm = SGDOneClassSVM(nu=0.05, random_state=seed)
        svm.fit(train_mapped)

        def ocscore(x: np.ndarray) -> np.ndarray:
            return -svm.decision_function(fmap.transform(x))

        svm_val = ocscore(val_mm)
        svm_test = ocscore(test_mm)
        svm_dos = concat_scores(dos_mm, ocscore)
        svm_fdi = concat_scores(fdi_mm, ocscore)
        per_seed_ref.append(reference_row("One-Class SVM", seed, svm_val, svm_test, svm_dos, svm_fdi))
        per_seed_budget.append(evaluate_budgets("One-Class SVM", seed, svm_val, svm_test, svm_dos, svm_fdi, budgets))
        timing_rows.append({"seed": seed, "model": "One-Class SVM", "seconds": time.time() - t0})
        del fmap, train_mapped, svm
        gc.collect()

        # LSTM AE
        if not args.skip_lstm:
            tf.keras.backend.clear_session()
            set_seed(seed)
            t0 = time.time()
            lstm = make_lstm_ae(n_input)
            lstm.fit(
                train_seq,
                train_seq,
                epochs=args.epochs_lstm,
                batch_size=512,
                validation_data=(val_seq, val_seq),
                callbacks=[callbacks.EarlyStopping(monitor="val_loss", patience=5, restore_best_weights=True)],
                verbose=0,
            )

            def lstm_scores(seqs: np.ndarray) -> np.ndarray:
                recon = lstm.predict(seqs, verbose=0)
                return np.mean(np.abs(recon - seqs), axis=-1).flatten()

            l_val = lstm_scores(val_seq)
            l_test = lstm_scores(test_seq)
            l_dos = np.concatenate([lstm_scores(x) for x in dos_seq_arrays])
            l_fdi = np.concatenate([lstm_scores(x) for x in fdi_seq_arrays])
            per_seed_ref.append(reference_row("LSTM AE", seed, l_val, l_test, l_dos, l_fdi))
            per_seed_budget.append(evaluate_budgets("LSTM AE", seed, l_val, l_test, l_dos, l_fdi, budgets))
            timing_rows.append({"seed": seed, "model": "LSTM AE", "seconds": time.time() - t0})
            del lstm
            gc.collect()

        # Hybrid AE
        if not args.skip_hybrid:
            tf.keras.backend.clear_session()
            set_seed(seed)
            t0 = time.time()
            hybrid = HybridAblationAE(n_input)
            hybrid.compile(
                optimizer=optimizers.AdamW(learning_rate=1e-3, weight_decay=1e-5),
                loss=tf.keras.losses.Huber(delta=1.0),
            )
            hybrid.fit(
                train_rb,
                train_rb,
                epochs=args.epochs_hybrid,
                batch_size=1024,
                validation_data=(val_rb, val_rb),
                callbacks=[
                    callbacks.EarlyStopping(
                        monitor="val_loss", patience=8, restore_best_weights=True, min_delta=1e-4
                    ),
                    callbacks.ReduceLROnPlateau(
                        monitor="val_loss", factor=0.5, patience=3, min_lr=1e-5, verbose=0
                    ),
                ],
                verbose=0,
                shuffle=True,
            )
            h_train_recon = reconstruction_mae(hybrid, train_rb)
            h_val_recon = reconstruction_mae(hybrid, val_rb)
            h_test_recon = reconstruction_mae(hybrid, test_rb)
            h_train_lat = hybrid.encode(train_rb, training=False).numpy()
            h_val_lat = hybrid.encode(val_rb, training=False).numpy()
            h_test_lat = hybrid.encode(test_rb, training=False).numpy()
            cov = EmpiricalCovariance().fit(h_train_lat)
            h_train_md = cov.mahalanobis(h_train_lat)
            h_val_md = cov.mahalanobis(h_val_lat)
            h_test_md = cov.mahalanobis(h_test_lat)
            rmed, rmad = robust_stats(h_train_recon)
            mmed, mmad = robust_stats(h_train_md)
            h_val = robust_zscore(h_val_recon, rmed, rmad) + robust_zscore(h_val_md, mmed, mmad)
            h_test = robust_zscore(h_test_recon, rmed, rmad) + robust_zscore(h_test_md, mmed, mmad)

            def hscore(x: np.ndarray) -> np.ndarray:
                er = reconstruction_mae(hybrid, x)
                z = hybrid.encode(x, training=False).numpy()
                md = cov.mahalanobis(z)
                return robust_zscore(er, rmed, rmad) + robust_zscore(md, mmed, mmad)

            h_dos = concat_scores(dos_rb, hscore)
            h_fdi = concat_scores(fdi_rb, hscore)
            per_seed_ref.append(reference_row("Hybrid AE", seed, h_val, h_test, h_dos, h_fdi))
            per_seed_budget.append(evaluate_budgets("Hybrid AE", seed, h_val, h_test, h_dos, h_fdi, budgets))
            timing_rows.append({"seed": seed, "model": "Hybrid AE", "seconds": time.time() - t0})
            del hybrid, cov, h_train_lat, h_val_lat, h_test_lat
            gc.collect()

        # Incremental checkpoint after every seed.
        pd.DataFrame(per_seed_ref).to_csv(out_dir / "per_seed_reference.csv", index=False)
        pd.concat(per_seed_budget, ignore_index=True).to_csv(out_dir / "per_seed_budgets.csv", index=False)
        pd.DataFrame(timing_rows).to_csv(out_dir / "timings.csv", index=False)

    ref_df = pd.DataFrame(per_seed_ref)
    budget_df = pd.concat(per_seed_budget, ignore_index=True)
    ref_summary = summarise_across_seeds(ref_df, ["model"])
    budget_summary = summarise_across_seeds(budget_df, ["model", "target_fpr_budget_pct"])
    ref_summary.to_csv(out_dir / "summary_reference.csv", index=False)
    budget_summary.to_csv(out_dir / "summary_budgets.csv", index=False)

    val_n = len(val_mm)
    empirical_fpr_step = 1.0 / val_n
    manifest = {
        "purpose": "ANUBIS camera-ready multi-seed and low-FPR extension",
        "data": str(data_path.resolve()),
        "rows": int(len(all_df)),
        "features": int(len(feat_cols)),
        "train_windows": int(len(train_mm)),
        "validation_windows": int(val_n),
        "normal_test_windows": int(len(test_mm)),
        "seeds": seeds,
        "budgets_fraction": budgets,
        "budgets_percent": [x * 100.0 for x in budgets],
        "reference_percentile": REFERENCE_PERCENTILE,
        "window_seconds": WINDOW_SECONDS,
        "windows_per_day_assumption": WINDOWS_PER_DAY,
        "empirical_validation_fpr_resolution_fraction": empirical_fpr_step,
        "empirical_validation_fpr_resolution_percent": empirical_fpr_step * 100.0,
        "raw_false_triggers_per_day_at_one_validation_exceedance": empirical_fpr_step * WINDOWS_PER_DAY,
        "python": sys.version,
        "platform": platform.platform(),
        "numpy": np.__version__,
        "pandas": pd.__version__,
        "scipy": scipy.__version__,
        "sklearn": sklearn.__version__,
        "tensorflow": tf.__version__,
        "tensorflow_devices": [d.name for d in tf.config.list_physical_devices()],
        "notes": [
            "95% intervals are t intervals across stochastic seeds, not population-level generalisation intervals.",
            "Raw false triggers/day assumes each positive 100 ms window is surfaced before aggregation or suppression.",
            "No SCD/specification baseline is implemented because the supplied artifact lacks SCD and raw packet-level data.",
        ],
    }
    (out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    print("\nCompleted.")
    print(f"Results: {out_dir.resolve()}")
    print(
        f"Validation empirical FPR resolution: {empirical_fpr_step*100:.6f}% "
        f"(~{empirical_fpr_step*WINDOWS_PER_DAY:.1f} raw false triggers/day at 10 Hz)."
    )


if __name__ == "__main__":
    run()
