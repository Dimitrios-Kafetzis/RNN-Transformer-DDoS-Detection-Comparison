#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
File: test_significance.py
Author: Dimitrios Kafetzis (kafetzis@aueb.gr)
Created: May 2025
License: MIT

Description:
    Performs statistical significance testing to determine if performance differences
    between models are statistically significant. Uses cross-validation to generate
    multiple F1 scores for each model, then conducts pairwise t-tests to assess
    if observed differences are meaningful or simply due to chance.

Usage:
    $ python test_significance.py --test-file PATH_TO_TEST_FILE [options]
    
    Examples:
    $ python test_significance.py --test-file data/nsl_kdd_dataset/NSL-KDD-Hard.csv
    $ python test_significance.py --test-file data/nsl_kdd_dataset/NSL-KDD-Hard.csv --model-dir custom_models
"""

import os
import numpy as np
import tensorflow as tf
from sklearn.model_selection import KFold
from sklearn.metrics import f1_score
import itertools
from scipy.stats import ttest_ind
import json
import argparse
import logging
from pathlib import Path
from data.loader import load_nslkdd_dataset, process_nslkdd_dataset
from evaluation_config import NUM_CROSS_VAL_RUNS
from generate_advanced_models_preds import load_and_align_hard

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Custom JSON encoder to handle NumPy types
class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, np.bool_):
            return bool(obj)
        return super(NumpyEncoder, self).default(obj)

def test_significance(model_dir, test_file, output_dir="evaluation_results/significance"):
    """Perform cross-validation and statistical significance testing."""
    os.makedirs(output_dir, exist_ok=True)
    
    # Load dataset
    PROJECT_ROOT = Path(__file__).parent.resolve()
    nsl_data_dir = str(PROJECT_ROOT / "data" / "nsl_kdd_dataset")
    X, y, _ = load_and_align_hard(nsl_data_dir, test_file)
    
    # Create a dictionary to store F1 scores for each model
    f1_scores = {}
    
    # Perform K-fold cross validation
    kf = KFold(n_splits=NUM_CROSS_VAL_RUNS, shuffle=True, random_state=42)
    
    for model_name in os.listdir(model_dir):
        model_path = os.path.join(model_dir, model_name)
        if not os.path.isdir(model_path):
            continue
            
        base_model_name = model_name.split('_')[0]  # Extract base model name
        logger.info(f"Testing {base_model_name} ({model_name})")
        
        model_f1_scores = []
        
        try:
            # Special case for threshold detector
            if base_model_name == "threshold_detector":
                # Check if thresholds.json exists
                thresholds_path = os.path.join(model_path, "thresholds.json")
                if not os.path.isfile(thresholds_path):
                    logger.warning(f"No thresholds.json found for {model_name}")
                    continue
                    
                # Load detector
                from models.threshold_detector import ThresholdDetector
                with open(thresholds_path, 'r') as f:
                    thresholds = json.load(f)
                
                detector = ThresholdDetector()
                detector.thresholds = {int(k): float(v) for k, v in thresholds.items()}
                detector.is_calibrated = True
                
                # Normalize data if normalization params exist
                try:
                    X_mean = np.load(os.path.join(model_path, "X_mean.npy"))
                    X_std = np.load(os.path.join(model_path, "X_std.npy"))
                except Exception as e:
                    logger.warning(f"No normalization params for threshold detector: {e}")
                    # Continue with unnormalized data
                    X_mean = None
                    X_std = None
                    
                # Perform cross-validation
                model_f1_scores = []
                for i, (train_idx, test_idx) in enumerate(kf.split(X)):
                    X_train, X_test = X[train_idx], X[test_idx]
                    y_train, y_test = y[train_idx], y[test_idx]
                    
                    # Normalize if params available
                    if X_mean is not None and X_std is not None:
                        X_test = (X_test - X_mean) / X_std
                        
                    # Predict using detector
                    feature_indices = list(range(min(X_test.shape[1], len(detector.thresholds))))
                    y_pred = detector.predict(X_test, feature_indices)
                    
                    # Calculate F1 score
                    f1 = f1_score(y_test, y_pred)
                    model_f1_scores.append(f1)
                    
                    logger.info(f"  Fold {i+1}/{NUM_CROSS_VAL_RUNS}: F1={f1:.4f}")
                    
                # Store F1 scores for this model
                f1_scores[base_model_name] = model_f1_scores
                continue

        except Exception as e:
            logger.error(f"Error testing {model_name}: {e}")
    
    # Perform pairwise t-tests
    model_pairs = list(itertools.combinations(f1_scores.keys(), 2))
    t_test_results = []
    
    for model1, model2 in model_pairs:
        scores1 = f1_scores[model1]
        scores2 = f1_scores[model2]
        
        try:
            t_stat, p_value = ttest_ind(scores1, scores2)
            
            t_test_results.append({
                "model1": model1,
                "model2": model2,
                "mean_diff": float(np.mean(scores1) - np.mean(scores2)),
                "t_statistic": float(t_stat),
                "p_value": float(p_value),
                "significant": bool(p_value < 0.05)  # Convert numpy.bool_ to Python bool
            })
        except Exception as e:
            logger.error(f"Error performing t-test for {model1} vs {model2}: {e}")
    
    # Save raw F1 scores
    with open(os.path.join(output_dir, "cross_validation_f1_scores.json"), 'w') as f:
        # Convert all numpy values to Python native types
        cleaned_scores = {k: [float(score) for score in v] for k, v in f1_scores.items()}
        json.dump(cleaned_scores, f, indent=4)
    
    # Save t-test results using the custom encoder
    with open(os.path.join(output_dir, "t_test_results.json"), 'w') as f:
        json.dump(t_test_results, f, indent=4, cls=NumpyEncoder)
    
    logger.info(f"Statistical significance test results saved to {output_dir}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Perform statistical significance testing")
    parser.add_argument("--test-file", required=True, help="Path to NSL-KDD-Hard.csv")
    parser.add_argument("--model-dir", default="saved_models", help="Directory containing saved models")
    args = parser.parse_args()
    
    test_significance(args.model_dir, args.test_file)