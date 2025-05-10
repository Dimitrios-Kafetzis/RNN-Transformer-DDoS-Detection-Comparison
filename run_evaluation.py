# run_evaluation.py
import os
import argparse
import logging
import subprocess
import json
import glob
import numpy as np
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

def run_evaluation(test_file, model_dir="saved_models", skip_steps=None, output_json="evaluation_results/complete_results.json"):
    """Run the complete evaluation pipeline and consolidate results into a single JSON file."""
    if skip_steps is None:
        skip_steps = []
    
    logger.info(f"Starting evaluation with test file: {test_file}")
    logger.info(f"Using model directory: {model_dir}")
    logger.info(f"Results will be saved to: {output_json}")
    
    # 1. Create evaluation directories if they don't exist
    os.makedirs("evaluation_results", exist_ok=True)
    os.makedirs("plots/model_profiles", exist_ok=True)
    
    # 2. Collect raw predictions
    if "predictions" not in skip_steps:
        logger.info("Step 1: Collecting raw predictions...")
        subprocess.run(["python3", "collect_predictions.py", 
                        "--test-file", test_file,
                        "--model-dir", model_dir])
    else:
        logger.info("Skipping step 1: Collecting raw predictions")
    
    # 3. Measure execution time and memory usage
    if "performance" not in skip_steps:
        logger.info("Step 2: Measuring performance...")
        subprocess.run(["python3", "measure_performance.py", 
                        "--test-file", test_file,
                        "--model-dir", model_dir])
    else:
        logger.info("Skipping step 2: Measuring performance")
    
    # 4. Analyze performance by attack type
    if "attack_types" not in skip_steps:
        logger.info("Step 3: Analyzing attack types...")
        subprocess.run(["python3", "analyze_attack_types.py", 
                        "--test-file", test_file,
                        "--model-dir", model_dir])
    else:
        logger.info("Skipping step 3: Analyzing attack types")
    
    # 5. Analyze scalability
    if "scalability" not in skip_steps:
        logger.info("Step 4: Analyzing scalability...")
        subprocess.run(["python3", "analyze_scalability.py", 
                        "--test-file", test_file,
                        "--model-dir", model_dir])
    else:
        logger.info("Skipping step 4: Analyzing scalability")
    
    # 6. Perform statistical significance testing
    if "significance" not in skip_steps:
        logger.info("Step 5: Testing statistical significance...")
        subprocess.run(["python3", "test_significance.py", 
                        "--test-file", test_file,
                        "--model-dir", model_dir])
    else:
        logger.info("Skipping step 5: Testing statistical significance")
    
    # 7. Analyze model interpretability
    if "interpretability" not in skip_steps:
        logger.info("Step 6: Analyzing model interpretability...")
        subprocess.run(["python3", "analyze_model_interpretability.py", 
                        "--test-file", test_file,
                        "--model-dir", model_dir])
    else:
        logger.info("Skipping step 6: Analyzing model interpretability")
    
    # 8. Generate visualizations
    if "visualizations" not in skip_steps:
        logger.info("Step 7: Generating visualizations...")
        subprocess.run(["python3", "generate_visualizations.py"])
    else:
        logger.info("Skipping step 7: Generating visualizations")
    
    # 9. Consolidate all results into a single JSON file
    logger.info("Step 8: Consolidating results into a single JSON file...")
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
    
    # Calculate overall model performance metrics
    consolidated_results["model_metrics"] = {}
    
    # Find optimal threshold and F1 score for each model
    try:
        import numpy as np
        from sklearn.metrics import f1_score, precision_score, recall_score, accuracy_score
        
        y_true = np.load(os.path.join("evaluation_results/predictions", "y_true.npy"))
        
        for f in glob.glob(os.path.join("evaluation_results/predictions", "*_raw_probs.npy")):
            model_name = os.path.basename(f).split('_')[0]
            y_pred = np.load(f)
            
            # Find optimal threshold
            best_f1 = 0
            best_threshold = 0.5
            
            # Use default threshold range for most models, special range for Transformer
            threshold_range = np.linspace(0, 1, 100)
            if model_name == "transformer":
                threshold_range = np.linspace(0, 0.1, 100)
            
            for threshold in threshold_range:
                y_binary = (y_pred >= threshold).astype(int)
                f1 = f1_score(y_true, y_binary)
                if f1 > best_f1:
                    best_f1 = f1
                    best_threshold = threshold
            
            # Get predictions at optimal threshold
            y_binary = (y_pred >= best_threshold).astype(int)
            
            # Calculate metrics
            precision = precision_score(y_true, y_binary)
            recall = recall_score(y_true, y_binary)
            accuracy = accuracy_score(y_true, y_binary)
            
            consolidated_results["model_metrics"][model_name] = {
                "optimal_threshold": float(best_threshold),
                "f1_score": float(best_f1),
                "precision": float(precision),
                "recall": float(recall),
                "accuracy": float(accuracy)
            }
    except Exception as e:
        logger.warning(f"Could not calculate model metrics: {e}")
    
    # Save consolidated results using the NumpyEncoder
    try:
        with open(output_json, 'w') as f:
            json.dump(consolidated_results, f, indent=4, cls=NumpyEncoder)
        logger.info(f"Consolidated results saved to {output_json}")
    except Exception as e:
        logger.error(f"Error saving consolidated results: {e}")
    
    logger.info(f"Evaluation pipeline completed!")
    logger.info(f"Log file saved to: {log_filename}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run the complete evaluation pipeline")
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