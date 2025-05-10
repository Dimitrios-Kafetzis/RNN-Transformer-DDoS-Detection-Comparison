#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
File: analyze_attack_types.py
Author: Dimitrios Kafetzis (kafetzis@aueb.gr)
Created: May 2025
License: MIT

Description:
    Analyzes model performance across different attack types.
    Breaks down detection effectiveness by attack categories such as:
    - TCP SYN Flood
    - UDP Flood
    - HTTP Flood
    - ICMP Flood
    - Low-and-Slow attacks
    Calculates precision, recall, and F1 scores for each attack type and model.

Usage:
    $ python analyze_attack_types.py --test-file PATH_TO_TEST_FILE [options]
    
    Examples:
    $ python analyze_attack_types.py --test-file data/nsl_kdd_dataset/NSL-KDD-Hard.csv
    $ python analyze_attack_types.py --test-file data/nsl_kdd_dataset/NSL-KDD-Hard.csv --model-dir custom_models
"""

import os
import numpy as np
import pandas as pd
import tensorflow as tf
import argparse
import json
import logging
from pathlib import Path
from sklearn.metrics import precision_score, recall_score, f1_score
from data.loader import load_nslkdd_dataset, process_nslkdd_dataset
from evaluation_config import ATTACK_TYPES, PREDICTIONS_DIR
from generate_advanced_models_preds import load_and_align_hard, make_test_dataset

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def analyze_attack_types(model_dir, test_file, output_dir="evaluation_results/attack_types"):
    """Analyze model performance by attack type."""
    os.makedirs(output_dir, exist_ok=True)
    
    # Load NSL-KDD dataset with attack labels
    PROJECT_ROOT = Path(__file__).parent.resolve()
    nsl_data_dir = str(PROJECT_ROOT / "data" / "nsl_kdd_dataset")
    
    # Load ground truth predictions
    y_true = np.load(os.path.join(PREDICTIONS_DIR, "y_true.npy"))
    
    # Load raw dataset to get attack type labels
    try:
        # Try loading as raw format first
        raw_df = pd.read_csv(test_file, header=None)
        # Check if it has the right column count for NSL-KDD
        if raw_df.shape[1] != len(raw_df.columns):
            # Load with proper header
            raw_train_df, _ = load_nslkdd_dataset(nsl_data_dir, split="train")
            raw_df = pd.read_csv(test_file, header=None, names=raw_train_df.columns)
        
        # Extract attack types
        attack_labels = raw_df['label'].str.strip().str.lower()
    except:
        # If not raw format, try as processed format
        try:
            processed_df = pd.read_csv(test_file)
            attack_labels = processed_df['label'].str.strip().str.lower()
        except:
            logger.error("Could not extract attack types from test file")
            return
    
    # Make sure attack_labels has the same length as y_true
    if len(attack_labels) != len(y_true):
        logger.warning(f"Length mismatch: attack_labels ({len(attack_labels)}) != y_true ({len(y_true)})")
        # Truncate the longer one to match the shorter one
        min_length = min(len(attack_labels), len(y_true))
        attack_labels = attack_labels[:min_length]
        y_true = y_true[:min_length]
    
    # Map to attack categories
    attack_categories = attack_labels.map(lambda x: ATTACK_TYPES.get(x, "Other"))
    
    # Get predictions for each model
    results = {}
    
    # For each model's predictions
    for f in os.listdir(PREDICTIONS_DIR):
        if f.endswith("_raw_probs.npy"):
            model_name = f.split('_')[0]
            
            # Load raw probabilities
            y_probs = np.load(os.path.join(PREDICTIONS_DIR, f))
            
            # Ensure y_probs has the same length as y_true and attack_categories
            if len(y_probs) != len(y_true):
                logger.warning(f"Length mismatch: y_probs ({len(y_probs)}) != y_true ({len(y_true)})")
                # Truncate to the common length
                min_length = min(len(y_probs), len(y_true))
                y_probs = y_probs[:min_length]
                y_true = y_true[:min_length]
                attack_categories = attack_categories[:min_length]
            
            # Default threshold
            threshold = 0.5
            
            # Use 0.01 for Transformer
            if model_name == "transformer":
                threshold = 0.01
            
            # Get binary predictions
            y_pred = (y_probs >= threshold).astype(int)
            
            # Calculate metrics by attack type
            model_results = {}
            
            for attack_type in attack_categories.unique():
                # Get indices for this attack type
                attack_idx = attack_categories == attack_type
                
                if sum(attack_idx) == 0:
                    continue
                
                # Extract predictions and true labels for this attack type
                attack_y_true = y_true[attack_idx]
                attack_y_pred = y_pred[attack_idx]
                
                if len(np.unique(attack_y_true)) <= 1:
                    # Skip if all samples are the same class
                    continue
                
                # Calculate metrics
                precision = precision_score(attack_y_true, attack_y_pred, zero_division=0)
                recall = recall_score(attack_y_true, attack_y_pred, zero_division=0)
                f1 = f1_score(attack_y_true, attack_y_pred, zero_division=0)
                
                model_results[attack_type] = {
                    "precision": float(precision),
                    "recall": float(recall),
                    "f1": float(f1),
                    "num_samples": int(sum(attack_idx))
                }
            
            results[model_name] = model_results
    
    # Save results
    with open(os.path.join(output_dir, "attack_type_metrics.json"), 'w') as f:
        json.dump(results, f, indent=4)
    
    logger.info(f"Attack type analysis saved to {output_dir}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Analyze model performance by attack type")
    parser.add_argument("--test-file", required=True, help="Path to NSL-KDD-Hard.csv with attack labels")
    parser.add_argument("--model-dir", default="saved_models", help="Directory containing saved models")
    args = parser.parse_args()
    
    analyze_attack_types(args.model_dir, args.test_file)