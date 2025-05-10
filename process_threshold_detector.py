#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
File: process_threshold_detector.py
Author: Dimitrios Kafetzis (kafetzis@aueb.gr)
Created: May 2025
License: MIT

Description:
    Processes a threshold detector model and generates predictions on test data.
    Loads thresholds from a JSON file, applies them to normalized test data,
    and produces binary predictions and pseudoprobabilities for evaluation.
    Handles both raw and preprocessed input formats.

Usage:
    $ python process_threshold_detector.py --detector-dir DETECTOR_DIR --test-file TEST_FILE --output-dir OUTPUT_DIR
    
    Examples:
    $ python process_threshold_detector.py --detector-dir saved_models/threshold_detector_1746776936 --test-file data/nsl_kdd_dataset/NSL-KDD-Hard.csv --output-dir evaluation_results/predictions
"""

import os
import sys
import json
import numpy as np
import pandas as pd
import logging
from pathlib import Path
from data.loader import load_nslkdd_dataset, create_aligned_feature_space, normalize_features
from models.threshold_detector import ThresholdDetector
from evaluation_config import DECISION_THRESHOLDS

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def process_threshold_detector(detector_path, test_file, output_dir):
    """Process a threshold detector model and save predictions."""
    logger.info(f"Processing threshold detector: {detector_path}")
    
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
        _, aligned_test, feature_cols = create_aligned_feature_space(train_df, test_df)
    else:
        # If already processed, ensure it has the needed columns
        train_df, _ = load_nslkdd_dataset(nsl_data_dir, split="both")
        train_enc, _, feature_cols = create_aligned_feature_space(train_df)
        
        # Create aligned test dataset
        aligned_test = pd.DataFrame()
        for col in feature_cols:
            if col in test_df.columns:
                aligned_test[col] = test_df[col]
            else:
                aligned_test[col] = 0
        
        # Add binary label if needed
        if "binary_label" in test_df.columns:
            aligned_test["binary_label"] = test_df["binary_label"]
        elif "label" in test_df.columns:
            aligned_test["binary_label"] = (test_df["label"] != "normal").astype(int)
        else:
            # Default to assuming all are attacks if no label
            aligned_test["binary_label"] = 1
    
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
    
    # Ensure output directory exists
    os.makedirs(output_dir, exist_ok=True)
    
    # Extract model name and timestamp
    model_name = os.path.basename(detector_path).split("_")[0]
    timestamp = os.path.basename(detector_path).split("_")[1] if "_" in os.path.basename(detector_path) else "0"
    
    # Save binary predictions
    np.save(os.path.join(output_dir, f"{model_name}_{timestamp}_y_pred.npy"), y_pred)
    np.save(os.path.join(output_dir, f"{model_name}_{timestamp}_y_true.npy"), y_test)
    
    # Generate pseudoprobabilities
    scores = np.zeros(len(X_test_norm))
    for idx in feature_indices:
        if idx < X_test_norm.shape[1] and idx < len(detector.thresholds):
            threshold = detector.thresholds[idx]
            excess = X_test_norm[:, idx] - threshold
            scores = np.maximum(scores, excess)
    
    # Convert to probability-like values
    scores = 1 / (1 + np.exp(-scores))
    
    # Save raw scores
    np.save(os.path.join(output_dir, f"{model_name}_{timestamp}_raw_probs.npy"), scores)
    
    # Generate binary predictions at different thresholds
    for threshold in DECISION_THRESHOLDS:
        y_pred_t = (scores >= threshold).astype(np.int32)
        np.save(os.path.join(output_dir, f"{model_name}_{timestamp}_pred_t{threshold:.2f}.npy"), y_pred_t)
    
    logger.info(f"Successfully saved predictions for {model_name}")
    return True

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Process threshold detector model")
    parser.add_argument("--detector-dir", required=True, help="Path to threshold detector directory")
    parser.add_argument("--test-file", required=True, help="Path to test data file")
    parser.add_argument("--output-dir", required=True, help="Directory to save predictions")
    
    args = parser.parse_args()
    
    if not os.path.isdir(args.detector_dir):
        logger.error(f"Detector directory not found: {args.detector_dir}")
        sys.exit(1)
    
    if not os.path.isfile(args.test_file):
        logger.error(f"Test file not found: {args.test_file}")
        sys.exit(1)
    
    success = process_threshold_detector(args.detector_dir, args.test_file, args.output_dir)
    if success:
        logger.info("Threshold detector processing completed successfully")
    else:
        logger.error("Failed to process threshold detector")
        sys.exit(1)