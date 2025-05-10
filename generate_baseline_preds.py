#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
File: generate_baseline_preds.py
Author: Dimitrios Kafetzis (kafetzis@aueb.gr)
Created: May 2025
License: MIT

Description:
    Generates predictions using baseline detector models (Threshold Detector and Linear Model).
    Loads raw or one-hot NSL-KDD data, aligns features with the training feature space,
    and saves model predictions for evaluation. Handles proper normalization and
    feature alignment to ensure consistent evaluation across all models.

Usage:
    $ python generate_baseline_preds.py --test-file PATH_TO_TEST_FILE
    
    Examples:
    $ python generate_baseline_preds.py --test-file data/nsl_kdd_dataset/NSL-KDD-Hard.csv
"""

import os
import argparse
from pathlib import Path

import json
import numpy as np
import pandas as pd
import tensorflow as tf

from data.loader import load_nslkdd_dataset, process_nslkdd_dataset, normalize_features
from models.threshold_detector import ThresholdDetector
from models.linear_regressor    import create_linear_model

def build_feature_space(data_dir: str):
    """
    Load TRAIN split (raw), process & one-hot to derive canonical features.
    Returns raw_train_columns, train_enc_df, feature_cols.
    """
    raw_train_df, _ = load_nslkdd_dataset(data_dir=data_dir, split="both")
    proc_train = process_nslkdd_dataset(raw_train_df.copy())
    train_enc = pd.get_dummies(proc_train, columns=["protocol_type", "service", "flag"])
    # drop anything the models don’t ingest
    non_feats = {"label", "attack_type", "binary_label", "difficulty"}
    feature_cols = [c for c in train_enc.columns if c not in non_feats]
    return raw_train_df.columns, train_enc, feature_cols

def load_hard_set(data_dir: str,
                  raw_train_columns,
                  train_enc: pd.DataFrame,
                  feature_cols,
                  hard_csv: str):
    """
    Read NSL-KDD-Hard.csv (raw or one-hot) and align to train_enc's columns.
    Returns X_test (float32), y_test (int32).
    """
    # peek header
    header = pd.read_csv(hard_csv, nrows=0).columns.tolist()

    if "protocol_type" in header:
        # raw CSV → process + one-hot + align
        raw_hard = pd.read_csv(hard_csv, header=None, names=raw_train_columns)
        proc_hard = process_nslkdd_dataset(raw_hard)
        hard_enc = pd.get_dummies(proc_hard, columns=["protocol_type", "service", "flag"])
        
        # CRITICAL: Make sure all training features exist in hard set
        for col in feature_cols:
            if col not in hard_enc.columns:
                hard_enc[col] = 0
        
        # Now align columns to match training order
        X = hard_enc[feature_cols].to_numpy(dtype=np.float32)
        y = hard_enc["binary_label"].to_numpy(dtype=np.int32)
    else:
        # one-hot CSV → just reindex to train_enc
        hard_df = pd.read_csv(hard_csv)
        
        # CRITICAL: Ensure all columns from training exist
        for col in feature_cols:
            if col not in hard_df.columns:
                hard_df[col] = 0
                
        aligned = hard_df.reindex(columns=train_enc.columns, fill_value=0)
        X_df = aligned[feature_cols].copy()
        # cast any "True"/"False" → numeric
        for col in X_df.columns:
            if X_df[col].dtype == object:
                X_df[col] = pd.to_numeric(X_df[col], errors="coerce").fillna(0.0)
        X = X_df.to_numpy(dtype=np.float32)

        # binary_label may already exist
        if "binary_label" in aligned:
            y = aligned["binary_label"].to_numpy(dtype=np.int32)
        else:
            # fallback: label != "normal"
            y = (aligned["label"] != "normal").astype(np.int32).to_numpy()

    return X, y

def find_saved_linear_dir(saved_models_dir: str):
    for d in os.listdir(saved_models_dir):
        full = os.path.join(saved_models_dir, d)
        if os.path.isdir(full) and d.startswith("linear_model"):
            return full
    raise FileNotFoundError(f"No linear_model in {saved_models_dir}")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--test-file", required=True,
                   help="Path to NSL-KDD-Hard.csv (raw or one-hot)")
    args = p.parse_args()

    PROJECT_ROOT = Path(__file__).parent.parent.resolve()
    data_dir     = str(PROJECT_ROOT / "data" / "nsl_kdd_dataset")
    out_dir      = PROJECT_ROOT / "evaluation" / "predictions"
    os.makedirs(out_dir, exist_ok=True)

    # 1) build feature space from raw TRAIN
    raw_cols, train_enc, feat_cols = build_feature_space(data_dir)

    # 2) load & align Hard set
    X_test, y_test = load_hard_set(
        data_dir, raw_cols, train_enc, feat_cols, args.test_file
    )

    # 3) Normalize train and test data
    X_train = train_enc[feat_cols].to_numpy(dtype=np.float32)
    y_train = train_enc["binary_label"].to_numpy(dtype=np.int32)
    
    # Normalize with robust handling
    X_train_norm, mean, std = normalize_features(X_train)
    
    # Ensure X_test has the same number of features as X_train
    if X_test.shape[1] != X_train.shape[1]:
        print(f"Feature dimension mismatch: X_train={X_train.shape[1]}, X_test={X_test.shape[1]}")
        
        if X_test.shape[1] < X_train.shape[1]:
            # Pad X_test with zeros
            pad_width = X_train.shape[1] - X_test.shape[1]
            X_test_padded = np.pad(X_test, ((0, 0), (0, pad_width)), 'constant')
            X_test = X_test_padded
            print(f"X_test padded to {X_test.shape}")
        else:
            # Truncate X_test
            X_test = X_test[:, :X_train.shape[1]]
            print(f"X_test truncated to {X_test.shape}")
    
    X_test_norm, _, _ = normalize_features(X_test, mean, std)

    # 4) ThresholdDetector baseline
    thr = ThresholdDetector(name="threshold_detector", percentile=99.5)
    # Calibrate on normalized data
    thr.calibrate(X_train_norm, y_train, feature_indices=list(range(len(feat_cols))))
    # Predict on normalized data
    y_thr = thr.predict(X_test_norm, feature_indices=list(range(len(feat_cols))))
    np.save(out_dir / "threshold_detector_y_pred.npy", y_thr)
    np.save(out_dir / "threshold_detector_y_true.npy", y_test)
    
    # Save normalization parameters with threshold detector
    thr_dir = PROJECT_ROOT / "saved_models" / "threshold_detector"
    os.makedirs(thr_dir, exist_ok=True)
    np.save(thr_dir / "X_mean.npy", mean)
    np.save(thr_dir / "X_std.npy", std)
    # Save thresholds
    with open(thr_dir / "thresholds.json", "w") as f:
        json.dump(thr.thresholds, f)

    # 5) LinearRegressor baseline
    try:
        lin_dir = find_saved_linear_dir(str(PROJECT_ROOT / "saved_models"))
        lin_model = tf.keras.models.load_model(lin_dir, compile=False)
        
        # Try to load normalization parameters, otherwise use the ones we just calculated
        try:
            lin_mean = np.load(os.path.join(lin_dir, "X_mean.npy"))
            lin_std = np.load(os.path.join(lin_dir, "X_std.npy"))
            
            # Verify dimensions
            if lin_mean.shape[0] != X_test.shape[1]:
                print(f"Linear model dimension mismatch: model={lin_mean.shape[0]}, X_test={X_test.shape[1]}")
                
                if X_test.shape[1] < lin_mean.shape[0]:
                    # Pad X_test
                    pad_width = lin_mean.shape[0] - X_test.shape[1]
                    X_test_padded = np.pad(X_test, ((0, 0), (0, pad_width)), 'constant')
                    X_test = X_test_padded
                else:
                    # Truncate X_test
                    X_test = X_test[:, :lin_mean.shape[0]]
            
            X_test_lin_norm = (X_test - lin_mean) / lin_std
        except Exception as e:
            print(f"Using calculated normalization parameters: {e}")
            X_test_lin_norm = X_test_norm
        
        # Handle NaNs and infs
        X_test_lin_norm = np.nan_to_num(X_test_lin_norm, nan=0.0, posinf=0.0, neginf=0.0)
        
        # Predict
        y_lin_prob = lin_model.predict(X_test_lin_norm).flatten()
        y_lin = (y_lin_prob >= 0.5).astype(np.int32)
        np.save(out_dir / "linear_regressor_y_pred.npy", y_lin)
        np.save(out_dir / "linear_regressor_y_true.npy", y_test)
        
        print("✅ Baseline predictions saved to", out_dir)
    except Exception as e:
        print(f"❌ Error with linear model: {e}")

if __name__ == "__main__":
    main()
