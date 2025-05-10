#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
File: generate_advanced_models_preds.py
Author: Dimitrios Kafetzis (kafetzis@aueb.gr)
Created: May 2025
License: MIT

Description:
    Loads TensorFlow models from the model directory, evaluates them on the
    NSL-KDD-Hard dataset (raw or one-hot encoded), and saves prediction results.
    Handles proper data alignment, normalization, and sequence reshaping
    to ensure valid predictions for all model architectures.

Usage:
    $ python generate_advanced_models_preds.py --test-file PATH_TO_TEST_FILE --model-dir MODEL_DIR --output-dir OUTPUT_DIR [options]
    
    Examples:
    $ python generate_advanced_models_preds.py --test-file data/nsl_kdd_dataset/NSL-KDD-Hard.csv --model-dir saved_models --output-dir evaluation_results/predictions
    $ python generate_advanced_models_preds.py --test-file data/nsl_kdd_dataset/NSL-KDD-Hard.csv --model-dir saved_models --batch-size 128 --threshold 0.3
"""

import os
import glob
import argparse
import logging
from pathlib import Path

import numpy as np
import pandas as pd
import tensorflow as tf

from data.loader import load_nslkdd_dataset, process_nslkdd_dataset, create_sequence_dataset

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("evaluator")


def load_and_align_hard(data_dir: str, hard_csv: str):
    """
    1) Build one-hot feature space from raw TRAIN.
    2) Read Hard CSV raw vs one-hot, then align.
    Returns X_test, y_test, feature_cols.
    """
    # Load training data to get canonical feature space
    raw_train_df, _ = load_nslkdd_dataset(data_dir=data_dir, split="both")
    proc_train = process_nslkdd_dataset(raw_train_df.copy())
    train_enc = pd.get_dummies(proc_train, columns=["protocol_type", "service", "flag"])
    non_feats = {"label", "binary_label", "attack_type", "difficulty"}
    feature_cols = [c for c in train_enc.columns if c not in non_feats]

    # Try loading hard dataset as raw first
    try:
        raw_hard = pd.read_csv(hard_csv, header=None, names=raw_train_df.columns)
        proc_hard = process_nslkdd_dataset(raw_hard)
        hard_enc = pd.get_dummies(proc_hard, columns=["protocol_type", "service", "flag"])
        logger.info("RAW Hard CSV detected; processing + one-hot + align…")
    except:
        # Try as already one-hot
        hard_df = pd.read_csv(hard_csv)
        hard_enc = hard_df
        logger.info("One-hot Hard CSV detected; reindexing to canonical space…")
    
    # Ensure all training features exist in hard set
    hard_aligned = pd.DataFrame()
    
    # Add all feature columns
    for col in feature_cols:
        if col in hard_enc.columns:
            hard_aligned[col] = hard_enc[col]
        else:
            hard_aligned[col] = 0
    
    # Convert object columns to numeric
    for col in hard_aligned.columns:
        if hard_aligned[col].dtype == object:
            hard_aligned[col] = pd.to_numeric(hard_aligned[col], errors="coerce").fillna(0.0)
    
    # Prepare features and labels
    X = hard_aligned[feature_cols].to_numpy(dtype=np.float32)
    
    # Get binary labels
    if "binary_label" in hard_enc.columns:
        y = hard_enc["binary_label"].to_numpy(dtype=np.int32)
    elif "label" in hard_enc.columns:
        y = (hard_enc["label"] != "normal").astype(np.int32).to_numpy()
    else:
        # Default to all attacks if no label
        y = np.ones(len(hard_aligned), dtype=np.int32)
    
    logger.info(f"Hard set: {X.shape[0]} samples × {X.shape[1]} features")
    return X, y, feature_cols


def make_test_dataset(X, y, model, batch_size, model_dir):
    """
    Wrap (X,y) into a tf.data.Dataset, using sliding windows for sequence models.
    Apply normalization parameters from training.
    """
    # Apply normalization parameters if available
    try:
        X_mean = np.load(os.path.join(model_dir, "X_mean.npy"))
        X_std = np.load(os.path.join(model_dir, "X_std.npy"))
        
        # Verify dimensions
        if X_mean.shape[0] != X.shape[1]:
            logger.warning(f"Dimension mismatch: features={X.shape[1]}, mean={X_mean.shape[0]}")
            
            # Try to reconcile dimensions
            if X_mean.shape[0] > X.shape[1]:
                logger.warning("Model trained with more features than test data!")
                # Need to pad test data to match model's expected dimensions
                pad_width = X_mean.shape[0] - X.shape[1]
                X_padded = np.pad(X, ((0, 0), (0, pad_width)), 'constant')
                X = X_padded
            else:
                logger.warning("Model trained with fewer features than test data!")
                # Truncate test data to match model's expected dimensions
                X = X[:, :X_mean.shape[0]]
                
        # Now normalize
        X = (X - X_mean) / X_std
        X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)  # Handle any NaNs or infs
        logger.info(f"Applied normalization from {model_dir}")
    except Exception as e:
        logger.warning(f"No normalization params in {model_dir}: {e}")
        # Just make sure test data has no NaNs/infs
        X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
    
    # Check for input dimension compatibility with model
    inp_shape = model.input_shape
    if len(inp_shape) == 3:  # Sequence model
        expected_features = inp_shape[2]
        if X.shape[1] != expected_features:
            logger.warning(f"Feature mismatch for sequence model: have {X.shape[1]}, need {expected_features}")
            if X.shape[1] < expected_features:
                # Pad with zeros
                pad_width = expected_features - X.shape[1]
                X_padded = np.pad(X, ((0, 0), (0, pad_width)), 'constant')
                X = X_padded
            else:
                # Truncate
                X = X[:, :expected_features]
        
        W = inp_shape[1]  # Window size
        S = max(1, W // 2)  # Step size
        return create_sequence_dataset(
            features=X,
            labels=y,
            window_size=W,
            step_size=S,
            batch_size=batch_size,
            shuffle=False
        )
    else:  # Regular model
        expected_features = inp_shape[1]
        if X.shape[1] != expected_features:
            logger.warning(f"Feature mismatch: have {X.shape[1]}, need {expected_features}")
            if X.shape[1] < expected_features:
                # Pad with zeros
                pad_width = expected_features - X.shape[1]
                X_padded = np.pad(X, ((0, 0), (0, pad_width)), 'constant')
                X = X_padded
            else:
                # Truncate
                X = X[:, :expected_features]
        
        return tf.data.Dataset.from_tensor_slices((X, y)) \
                              .batch(batch_size) \
                              .prefetch(tf.data.AUTOTUNE)

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--test-file",  required=True,
                   help="Path to NSL-KDD-Hard.csv (raw or one-hot)")
    p.add_argument("--model-dir",  required=True,
                   help="Folder containing saved_models/<model>_<ts>/")
    p.add_argument("--output-dir", required=True,
                   help="Where to write <model>_y_pred.npy and _y_true.npy")
    p.add_argument("--batch-size", type=int, default=256)
    p.add_argument("--threshold",  type=float, default=0.5)
    args = p.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    PROJECT_ROOT = Path(__file__).parent.resolve()
    nsl_data_dir = str(PROJECT_ROOT / "data" / "nsl_kdd_dataset")

    # Load & align Hard Set once
    X_test, y_test, _ = load_and_align_hard(nsl_data_dir, args.test_file)

    # Iterate over each saved model
    for sub in sorted(glob.glob(os.path.join(args.model_dir, "*"))):
        if not os.path.isdir(sub):
            continue
        model_name = os.path.basename(sub)
        logger.info(f"\n=== Generating preds for {model_name} ===")

        try:
            model = tf.keras.models.load_model(sub, compile=False)
            ds = make_test_dataset(X_test, y_test, model, args.batch_size, sub)

            # predict proba → threshold
            y_prob = model.predict(ds, batch_size=args.batch_size).flatten()
            y_pred = (y_prob >= args.threshold).astype(np.int32)

            # recover y_true for sequence models
            if len(model.input_shape) == 3:
                W = model.input_shape[1]
                S = max(1, W // 2)
                
                # Apply normalization
                try:
                    X_mean = np.load(os.path.join(sub, "X_mean.npy"))
                    X_std = np.load(os.path.join(sub, "X_std.npy"))
                    X_test_norm = (X_test - X_mean) / X_std
                except Exception:
                    X_test_norm = X_test
                    
                seq_ds = create_sequence_dataset(
                    features=X_test_norm,
                    labels=y_test,
                    window_size=W,
                    step_size=S,
                    batch_size=args.batch_size,
                    shuffle=False
                )
                y_true = np.concatenate([labels.numpy() for _, labels in seq_ds])
            else:
                y_true = y_test

            np.save(os.path.join(args.output_dir, f"{model_name}_y_pred.npy"), y_pred)
            np.save(os.path.join(args.output_dir, f"{model_name}_y_true.npy"), y_true)
            logger.info(f"Saved → {model_name}_y_pred.npy  &  {model_name}_y_true.npy")
        except Exception as e:
            logger.error(f"Error processing {model_name}: {e}")

if __name__ == "__main__":
    main()
