#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
File: threshold_detector_evaluation.py
Author: Dimitrios Kafetzis (kafetzis@aueb.gr)
Created: May 2025
License: MIT

Description:
    Evaluates threshold detector models and saves comprehensive metrics.
    Loads thresholds from JSON, applies them to test data, and calculates
    precision, recall, F1 score, accuracy, and ROC AUC. Generates both binary
    predictions and pseudoprobabilities at various thresholds for visualization.

Usage:
    $ python threshold_detector_evaluation.py --detector-dir DETECTOR_DIR --test-file TEST_FILE --output-dir OUTPUT_DIR
    
    Examples:
    $ python threshold_detector_evaluation.py --detector-dir saved_models/threshold_detector_1746776936 --test-file data/nsl_kdd_dataset/NSL-KDD-Hard.csv --output-dir evaluation_results/threshold_detector
    $ python threshold_detector_evaluation.py --model-dir saved_models --test-file data/nsl_kdd_dataset/NSL-KDD-Hard.csv
"""

import os
import sys
import json
import numpy as np
import pandas as pd
import logging
from pathlib import Path
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score, roc_auc_score
from data.loader import load_nslkdd_dataset, create_aligned_feature_space, normalize_features
from models.threshold_detector import ThresholdDetector
from evaluation_config import DECISION_THRESHOLDS, PREDICTIONS_DIR

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def evaluate_threshold_detector(detector_path, test_file, output_dir):
    """Evaluate a threshold detector model and save comprehensive metrics."""
    logger.info(f"Evaluating threshold detector: {detector_path}")
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Ensure the predictions directory exists
    os.makedirs(PREDICTIONS_DIR, exist_ok=True)
    
    # Load thresholds
    thresholds_file = os.path.join(detector_path, "thresholds.json")
    if not os.path.isfile(thresholds_file):
        logger.error(f"No thresholds.json found in {detector_path}")
        return False
    
    with open(thresholds_file, 'r') as f:
        thresholds = json.load(f)
    
    # Load normalization parameters
    try:
        X_mean = np.load(os.path.join(detector_path, "X_mean.npy"))
        X_std = np.load(os.path.join(detector_path, "X_std.npy"))
        norm_params_loaded = True
        logger.info("Loaded normalization parameters")
    except Exception as e:
        logger.warning(f"Could not load normalization parameters: {e}")
        norm_params_loaded = False
    
    # Load dataset
    PROJECT_ROOT = Path(__file__).parent.resolve()
    nsl_data_dir = str(PROJECT_ROOT / "data" / "nsl_kdd_dataset")
    
    # Try to determine if the test file is raw or processed
    try:
        # First, try loading it as a CSV with header
        test_df = pd.read_csv(test_file)
        is_raw = False
        logger.info(f"Loaded processed test file: {test_file}")
    except:
        try:
            # Try loading it as raw NSL-KDD data
            # Get the train set to determine column names
            train_df, _ = load_nslkdd_dataset(nsl_data_dir, split="both")
            test_df = pd.read_csv(test_file, header=None, names=train_df.columns)
            is_raw = True
            logger.info(f"Loaded raw test file: {test_file}")
        except Exception as e:
            logger.error(f"Failed to load test file {test_file}: {e}")
            return False
    
    # Process the data to align with the feature space
    if is_raw:
        # If raw, need to align with training feature space
        train_df, _ = load_nslkdd_dataset(nsl_data_dir, split="both")
        # Use copy() to avoid DataFrame fragmentation warnings
        train_enc, aligned_test, feature_cols = create_aligned_feature_space(train_df.copy(), test_df.copy())
    else:
        # If already processed, ensure it has the needed columns
        train_df, _ = load_nslkdd_dataset(nsl_data_dir, split="both")
        train_enc, _, feature_cols = create_aligned_feature_space(train_df.copy())
        
        # Create aligned test dataset (using DataFrame constructor to avoid fragmentation)
        test_data = {}
        for col in feature_cols:
            if col in test_df.columns:
                test_data[col] = test_df[col].values
            else:
                test_data[col] = np.zeros(len(test_df))
        
        # Add binary label if needed
        if "binary_label" in test_df.columns:
            test_data["binary_label"] = test_df["binary_label"].values
        elif "label" in test_df.columns:
            test_data["binary_label"] = (test_df["label"] != "normal").astype(int).values
        else:
            # Default to assuming all are attacks if no label
            test_data["binary_label"] = np.ones(len(test_df))
        
        aligned_test = pd.DataFrame(test_data)
    
    # Extract features and labels
    X_test = aligned_test[feature_cols].to_numpy(dtype=np.float32)
    y_test = aligned_test["binary_label"].to_numpy(dtype=np.int32)
    
    # Apply normalization
    if norm_params_loaded:
        X_test_norm, _, _ = normalize_features(X_test, X_mean, X_std)
    else:
        X_test_norm, _, _ = normalize_features(X_test)
    
    # Create and configure detector
    detector = ThresholdDetector()
    detector.thresholds = {int(k): float(v) for k, v in thresholds.items()}
    detector.is_calibrated = True
    
    # Get feature indices
    feature_indices = list(range(min(X_test_norm.shape[1], len(detector.thresholds))))
    logger.info(f"Using {len(feature_indices)} features")
    
    # Make predictions
    y_pred = detector.predict(X_test_norm, feature_indices)
    
    # Calculate metrics
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    accuracy = accuracy_score(y_test, y_pred)
    
    # Calculate pseudo-probabilities
    scores = np.zeros(len(X_test_norm))
    for idx in feature_indices:
        if idx < X_test_norm.shape[1] and idx < len(detector.thresholds):
            threshold = detector.thresholds[idx]
            excess = X_test_norm[:, idx] - threshold
            scores = np.maximum(scores, excess)
    
    # Convert to probability-like values using sigmoid
    scores = 1 / (1 + np.exp(-scores))
    
    # Calculate AUC if possible
    try:
        roc_auc = roc_auc_score(y_test, scores)
    except:
        roc_auc = 0.0
    
    # Save all metrics
    model_name = os.path.basename(detector_path).split("_")[0]
    timestamp = os.path.basename(detector_path).split("_")[1] if "_" in os.path.basename(detector_path) else "0"
    
    # Save metrics to JSON
    metrics = {
        "precision": float(precision),
        "recall": float(recall),
        "f1_score": float(f1),
        "accuracy": float(accuracy),
        "roc_auc": float(roc_auc),
        "num_samples": int(len(y_test))
    }
    
    metrics_file = os.path.join(output_dir, f"{model_name}_{timestamp}_metrics.json")
    with open(metrics_file, 'w') as f:
        json.dump(metrics, f, indent=2)
    logger.info(f"Saved metrics to {metrics_file}")
    
    # Save predictions for later visualization
    predictions_dir = os.path.join(output_dir, "predictions")
    os.makedirs(predictions_dir, exist_ok=True)
    
    # Save to both the threshold detector output directory and the main predictions directory
    for out_dir in [predictions_dir, PREDICTIONS_DIR]:
        np.save(os.path.join(out_dir, f"{model_name}_{timestamp}_y_pred.npy"), y_pred)
        np.save(os.path.join(out_dir, f"{model_name}_{timestamp}_y_true.npy"), y_test)
        np.save(os.path.join(out_dir, f"{model_name}_{timestamp}_raw_probs.npy"), scores)
        
        # Save a common y_true.npy to fix the "file not found" error
        np.save(os.path.join(out_dir, "y_true.npy"), y_test)
        
        # Generate binary predictions at different thresholds for ROC/PR curves
        for threshold in DECISION_THRESHOLDS:
            y_pred_t = (scores >= threshold).astype(np.int32)
            np.save(os.path.join(out_dir, f"{model_name}_{timestamp}_pred_t{threshold:.2f}.npy"), y_pred_t)
    
    logger.info(f"Successfully evaluated {model_name}")
    return True

def find_threshold_detector_models(model_dir):
    """Find all threshold detector models in the model directory."""
    threshold_models = []
    for d in os.listdir(model_dir):
        if d.startswith("threshold_detector_") and os.path.isdir(os.path.join(model_dir, d)):
            threshold_models.append(os.path.join(model_dir, d))
    return threshold_models

def main():
    import argparse
    
    parser = argparse.ArgumentParser(description="Evaluate threshold detector models")
    parser.add_argument("--detector-dir", help="Path to specific threshold detector directory (optional)")
    parser.add_argument("--model-dir", default="saved_models", help="Directory containing all models")
    parser.add_argument("--test-file", required=True, help="Path to test data file")
    parser.add_argument("--output-dir", default="evaluation_results/threshold_detector", help="Directory to save evaluation results")
    
    args = parser.parse_args()
    
    # If a specific detector directory is provided, use it
    if args.detector_dir:
        if not os.path.isdir(args.detector_dir):
            logger.error(f"Detector directory not found: {args.detector_dir}")
            sys.exit(1)
        
        success = evaluate_threshold_detector(args.detector_dir, args.test_file, args.output_dir)
        if success:
            logger.info("Threshold detector evaluation completed successfully")
        else:
            logger.error("Failed to evaluate threshold detector")
            sys.exit(1)
    else:
        # Otherwise, find all threshold detector models in the model directory
        threshold_models = find_threshold_detector_models(args.model_dir)
        
        if not threshold_models:
            logger.error(f"No threshold detector models found in {args.model_dir}")
            sys.exit(1)
        
        # Evaluate each model
        success_count = 0
        for model in threshold_models:
            success = evaluate_threshold_detector(model, args.test_file, args.output_dir)
            if success:
                success_count += 1
        
        if success_count > 0:
            logger.info(f"Successfully evaluated {success_count}/{len(threshold_models)} threshold detector models")
        else:
            logger.error("Failed to evaluate any threshold detector models")
            sys.exit(1)

if __name__ == "__main__":
    main()