#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
File: improved_run_evaluation.py
Author: Dimitrios Kafetzis (kafetzis@aueb.gr)
Created: May 2025
License: MIT

Description:
    An enhanced evaluation pipeline that separates threshold detector evaluation
    from neural network model evaluation, then combines results for final reporting.
    Orchestrates the complete evaluation process including prediction collection,
    performance measurement, attack type analysis, scalability testing,
    statistical significance testing, and visualization generation.

Usage:
    $ python improved_run_evaluation.py --test-file PATH_TO_TEST_FILE [options]
    
    Examples:
    $ python improved_run_evaluation.py --test-file data/nsl_kdd_dataset/NSL-KDD-Hard.csv
    $ python improved_run_evaluation.py --test-file data/nsl_kdd_dataset/NSL-KDD-Hard.csv --skip predictions performance
    $ python improved_run_evaluation.py --test-file data/nsl_kdd_dataset/NSL-KDD-Hard.csv --model-dir custom_models
"""

import os
import argparse
import logging
import subprocess
import json
import glob
import numpy as np
import shutil
from pathlib import Path
import datetime

# Setup logging to both console and file
log_filename = f"evaluation_log_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(log_filename),
        logging.StreamHandler()
    ]
)
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
        return super(NumpyEncoder, self).default(obj)

def run_threshold_detector_evaluation(test_file, model_dir="saved_models", output_dir="evaluation_results/threshold_detector"):
    """
    Run evaluation specifically for the threshold detector model using the dedicated script.
    """
    logger.info("Starting threshold detector evaluation...")
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Find threshold detector model
    threshold_model_dirs = []
    for d in os.listdir(model_dir):
        if d.startswith("threshold_detector_"):
            threshold_model_dirs.append(os.path.join(model_dir, d))
    
    if not threshold_model_dirs:
        logger.error("No threshold detector model found")
        return False
    
    # Use the most recent one
    threshold_model_dir = sorted(threshold_model_dirs)[-1]
    logger.info(f"Using threshold detector model: {threshold_model_dir}")
    
    # Run the dedicated threshold detector evaluation script
    try:
        # Use subprocess to run the evaluation script
        subprocess.run([
            "python3", "threshold_detector_evaluation.py",
            "--detector-dir", threshold_model_dir,
            "--test-file", test_file,
            "--output-dir", output_dir
        ], check=True)
        
        logger.info("Threshold detector evaluation completed successfully")
        return True
    except subprocess.CalledProcessError as e:
        logger.error(f"Error running threshold detector evaluation: {e}")
        return False

def run_neural_models_evaluation(test_file, model_dir="saved_models", skip_steps=None):
    """Run evaluation for all neural network models (excluding threshold detector)."""
    if skip_steps is None:
        skip_steps = []
    
    logger.info("Starting neural network models evaluation...")
    
    # Create a list of neural model directories (excluding threshold detector)
    neural_models = []
    for d in os.listdir(model_dir):
        full_path = os.path.join(model_dir, d)
        if (not d.startswith("threshold_detector_") and 
            os.path.isdir(full_path)):
            neural_models.append(full_path)
    
    if not neural_models:
        logger.warning("No neural network models found in the model directory")
        return False
    
    # Ensure the predictions directory exists to avoid "file not found" errors
    os.makedirs("evaluation_results/predictions", exist_ok=True)
    
    # 1. Collect raw predictions
    if "predictions" not in skip_steps:
        logger.info("Step 1: Collecting raw predictions for neural models...")
        try:
            subprocess.run([
                "python3", "collect_predictions.py", 
                "--test-file", test_file,
                "--model-dir", model_dir
            ], check=True)
        except subprocess.CalledProcessError as e:
            logger.error(f"Error collecting predictions: {e}")
    else:
        logger.info("Skipping step 1: Collecting raw predictions for neural models")
    
    # 2. Measure execution time and memory usage
    if "performance" not in skip_steps:
        logger.info("Step 2: Measuring performance for neural models...")
        try:
            subprocess.run([
                "python3", "measure_performance.py", 
                "--test-file", test_file,
                "--model-dir", model_dir
            ], check=True)
        except subprocess.CalledProcessError as e:
            logger.error(f"Error measuring performance: {e}")
    else:
        logger.info("Skipping step 2: Measuring performance for neural models")
    
    # Copy y_true.npy if needed to fix the "file not found" error
    pred_files = glob.glob(os.path.join("evaluation_results/predictions", "*_y_true.npy"))
    if pred_files and not os.path.exists(os.path.join("evaluation_results/predictions", "y_true.npy")):
        # Copy the first one to be the common ground truth
        shutil.copy(pred_files[0], os.path.join("evaluation_results/predictions", "y_true.npy"))
    
    # 3. Analyze performance by attack type
    if "attack_types" not in skip_steps:
        logger.info("Step 3: Analyzing attack types for neural models...")
        try:
            subprocess.run([
                "python3", "analyze_attack_types.py", 
                "--test-file", test_file,
                "--model-dir", model_dir
            ], check=True)
        except subprocess.CalledProcessError as e:
            logger.error(f"Error analyzing attack types: {e}")
    else:
        logger.info("Skipping step 3: Analyzing attack types for neural models")
    
    # 4. Analyze scalability
    if "scalability" not in skip_steps:
        logger.info("Step 4: Analyzing scalability for neural models...")
        try:
            subprocess.run([
                "python3", "analyze_scalability.py", 
                "--test-file", test_file,
                "--model-dir", model_dir
            ], check=True)
        except subprocess.CalledProcessError as e:
            logger.error(f"Error analyzing scalability: {e}")
    else:
        logger.info("Skipping step 4: Analyzing scalability for neural models")
    
    # 5. Perform statistical significance testing
    if "significance" not in skip_steps:
        logger.info("Step 5: Testing statistical significance for neural models...")
        try:
            subprocess.run([
                "python3", "test_significance.py", 
                "--test-file", test_file,
                "--model-dir", model_dir
            ], check=True)
        except subprocess.CalledProcessError as e:
            logger.error(f"Error testing statistical significance: {e}")
    else:
        logger.info("Skipping step 5: Testing statistical significance for neural models")
    
    # 6. Analyze model interpretability
    if "interpretability" not in skip_steps:
        logger.info("Step 6: Analyzing model interpretability for neural models...")
        try:
            subprocess.run([
                "python3", "analyze_model_interpretability.py", 
                "--test-file", test_file,
                "--model-dir", model_dir
            ], check=True)
        except subprocess.CalledProcessError as e:
            logger.error(f"Error analyzing model interpretability: {e}")
    else:
        logger.info("Skipping step 6: Analyzing model interpretability for neural models")
    
    logger.info("Neural network models evaluation completed")
    return True

def generate_combined_results(output_json="evaluation_results/complete_results.json"):
    """
    Consolidate all evaluation results into a single comprehensive JSON file.
    This includes both neural models and threshold detector results.
    """
    logger.info("Generating combined evaluation results...")
    
    # Ensure the output directory exists
    os.makedirs(os.path.dirname(output_json), exist_ok=True)
    
    # Initialize the results dictionary
    consolidated_results = {}
    
    # Load training histories
    consolidated_results["training_history"] = {}
    for f in glob.glob(os.path.join("evaluation_results/training_histories", "*.pkl")):
        model_name = os.path.basename(f).split('_')[0]
        try:
            import pickle
            with open(f, 'rb') as file:
                history = pickle.load(file)
                # Convert numpy arrays to lists for JSON serialization
                history_dict = {}
                for key, value in history.items():
                    if hasattr(value, 'tolist'):
                        history_dict[key] = value.tolist()
                    else:
                        history_dict[key] = list(value)
                consolidated_results["training_history"][model_name] = history_dict
        except Exception as e:
            logger.warning(f"Could not load training history for {model_name}: {e}")
    
    # Load performance metrics
    try:
        with open(os.path.join("evaluation_results/timing", "inference_times.json"), 'r') as f:
            consolidated_results["timing"] = json.load(f)
    except Exception as e:
        logger.warning(f"Could not load timing information: {e}")
        consolidated_results["timing"] = {}
    
    try:
        with open(os.path.join("evaluation_results/memory", "memory_usage.json"), 'r') as f:
            consolidated_results["memory"] = json.load(f)
    except Exception as e:
        logger.warning(f"Could not load memory usage information: {e}")
        consolidated_results["memory"] = {}
    
    # Load feature extraction time
    try:
        with open(os.path.join("evaluation_results/timing", "feature_extraction.json"), 'r') as f:
            consolidated_results["feature_extraction"] = json.load(f)
    except Exception as e:
        logger.warning(f"Could not load feature extraction timing: {e}")
        consolidated_results["feature_extraction"] = {}
    
    # Load attack type metrics
    try:
        with open(os.path.join("evaluation_results/attack_types", "attack_type_metrics.json"), 'r') as f:
            consolidated_results["attack_types"] = json.load(f)
    except Exception as e:
        logger.warning(f"Could not load attack type metrics: {e}")
        consolidated_results["attack_types"] = {}
    
    # Load scalability results
    try:
        with open(os.path.join("evaluation_results/scalability", "scalability_results.json"), 'r') as f:
            consolidated_results["scalability"] = json.load(f)
    except Exception as e:
        logger.warning(f"Could not load scalability results: {e}")
        consolidated_results["scalability"] = {}
    
    # Load statistical significance results
    try:
        with open(os.path.join("evaluation_results/significance", "cross_validation_f1_scores.json"), 'r') as f:
            consolidated_results["cross_validation"] = json.load(f)
    except Exception as e:
        logger.warning(f"Could not load cross-validation results: {e}")
        consolidated_results["cross_validation"] = {}
    
    try:
        with open(os.path.join("evaluation_results/significance", "t_test_results.json"), 'r') as f:
            consolidated_results["significance_tests"] = json.load(f)
    except Exception as e:
        logger.warning(f"Could not load significance test results: {e}")
        consolidated_results["significance_tests"] = []
    
    # Load model interpretability results
    try:
        with open(os.path.join("evaluation_results/interpretability", "feature_importance.json"), 'r') as f:
            consolidated_results["feature_importance"] = json.load(f)
    except Exception as e:
        logger.warning(f"Could not load feature importance data: {e}")
        consolidated_results["feature_importance"] = {}
    
    try:
        with open(os.path.join("evaluation_results/interpretability", "attention_weights.json"), 'r') as f:
            consolidated_results["attention_weights"] = json.load(f)
    except Exception as e:
        logger.warning(f"Could not load attention weights: {e}")
        consolidated_results["attention_weights"] = {}
    
    # Load threshold detector metrics if available
    threshold_metrics_files = glob.glob(os.path.join("evaluation_results/threshold_detector", "*_metrics.json"))
    if threshold_metrics_files:
        consolidated_results["threshold_detector_metrics"] = {}
        for metrics_file in threshold_metrics_files:
            model_name = os.path.basename(metrics_file).split('_')[0]
            try:
                with open(metrics_file, 'r') as f:
                    consolidated_results["threshold_detector_metrics"][model_name] = json.load(f)
            except Exception as e:
                logger.warning(f"Could not load threshold detector metrics from {metrics_file}: {e}")
    
    # Calculate overall model performance metrics
    consolidated_results["model_metrics"] = {}
    
    # Find all available prediction files
    pred_files = []
    for root, dirs, files in os.walk("evaluation_results"):
        for file in files:
            if file.endswith("_y_pred.npy"):
                pred_files.append(os.path.join(root, file))
    
    # Process each prediction file
    for pred_file in pred_files:
        # Get corresponding true labels
        true_file = pred_file.replace("_y_pred.npy", "_y_true.npy")
        if not os.path.exists(true_file):
            logger.warning(f"No true labels found for {pred_file}")
            continue
        
        # Extract model name
        model_name = os.path.basename(pred_file).split('_')[0]
        
        try:
            from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score
            
            # Load predictions and true labels
            y_pred = np.load(pred_file)
            y_true = np.load(true_file)
            
            # Calculate metrics
            precision = precision_score(y_true, y_pred)
            recall = recall_score(y_true, y_pred)
            f1 = f1_score(y_true, y_pred)
            accuracy = accuracy_score(y_true, y_pred)
            
            # Save metrics
            consolidated_results["model_metrics"][model_name] = {
                "optimal_threshold": 0.5,  # Default for binary predictions
                "f1_score": float(f1),
                "precision": float(precision),
                "recall": float(recall),
                "accuracy": float(accuracy),
                "num_samples": int(len(y_true))
            }
            
            logger.info(f"Calculated metrics for {model_name}")
        except Exception as e:
            logger.warning(f"Error calculating metrics for {model_name}: {e}")
    
    # Save consolidated results using the NumpyEncoder
    try:
        with open(output_json, 'w') as f:
            json.dump(consolidated_results, f, indent=4, cls=NumpyEncoder)
        logger.info(f"Consolidated results saved to {output_json}")
        return consolidated_results
    except Exception as e:
        logger.error(f"Error saving consolidated results: {e}")
        return None

def ensure_consistent_true_labels():
    """
    Ensure that all evaluation scripts use the same ground truth labels by
    copying one y_true.npy file to all required locations.
    """
    # Find all _y_true.npy files
    true_files = []
    for root, dirs, files in os.walk("evaluation_results"):
        for file in files:
            if file.endswith("_y_true.npy"):
                true_files.append(os.path.join(root, file))
    
    if not true_files:
        logger.warning("No ground truth files found for consistency check")
        return
    
    # Use the first one as the reference
    reference_file = true_files[0]
    reference_labels = np.load(reference_file)
    
    # Create a common y_true.npy in each relevant directory
    for root in ["evaluation_results/predictions", "evaluation_results/threshold_detector/predictions"]:
        os.makedirs(root, exist_ok=True)
        common_file = os.path.join(root, "y_true.npy")
        np.save(common_file, reference_labels)
        logger.info(f"Created consistent ground truth file at {common_file}")

def generate_visualizations():
    """
    Generate all visualizations using the combined results.
    """
    logger.info("Generating visualizations...")
    
    # Ensure consistent ground truth before generating visualizations
    ensure_consistent_true_labels()
    
    try:
        # Call the existing visualization script
        subprocess.run(["python3", "fixed_generate_visualizations.py"], check=True)
        logger.info("Visualizations generation completed")
        return True
    except subprocess.CalledProcessError as e:
        logger.error(f"Error generating visualizations: {e}")
        return False

def run_evaluation(test_file, model_dir="saved_models", skip_steps=None, output_json="evaluation_results/complete_results.json"):
    """
    Run the complete evaluation pipeline with separate handling for threshold detector and neural models.
    """
    if skip_steps is None:
        skip_steps = []
    
    logger.info(f"Starting improved evaluation with test file: {test_file}")
    logger.info(f"Using model directory: {model_dir}")
    logger.info(f"Results will be saved to: {output_json}")
    
    # Create evaluation directories
    os.makedirs("evaluation_results", exist_ok=True)
    os.makedirs("plots/model_profiles", exist_ok=True)
    os.makedirs("evaluation_results/predictions", exist_ok=True)
    
    # Step 1: Evaluate threshold detector separately
    threshold_success = run_threshold_detector_evaluation(test_file, model_dir)
    if not threshold_success:
        logger.warning("Threshold detector evaluation failed or skipped")
    
    # Step 2: Evaluate neural models
    neural_success = run_neural_models_evaluation(test_file, model_dir, skip_steps)
    if not neural_success:
        logger.warning("Neural models evaluation failed or skipped")
    
    # Ensure consistent ground truth labels across all evaluation results
    ensure_consistent_true_labels()
    
    # Step 3: Generate combined results
    consolidated_results = generate_combined_results(output_json)
    
    # Step 4: Generate visualizations
    if "visualizations" not in skip_steps:
        logger.info("Generating visualizations...")
        generate_visualizations()
    else:
        logger.info("Skipping visualization generation")
    
    logger.info(f"Evaluation pipeline completed!")
    logger.info(f"Log file saved to: {log_filename}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run the improved evaluation pipeline")
    parser.add_argument("--test-file", required=True, help="Path to NSL-KDD-Hard.csv")
    parser.add_argument("--model-dir", default="saved_models", help="Directory containing saved models")
    parser.add_argument("--skip", nargs="*", choices=["predictions", "performance", "attack_types", 
                                                     "scalability", "significance", "interpretability", 
                                                     "visualizations"],
                       default=[], help="Steps to skip")
    parser.add_argument("--output-json", default="evaluation_results/complete_results.json",
                       help="Path to output consolidated JSON file")
    
    args = parser.parse_args()
    
    run_evaluation(args.test_file, args.model_dir, args.skip, args.output_json)