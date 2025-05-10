# collect_predictions.py
import os
import json
import numpy as np
import tensorflow as tf
import argparse
from pathlib import Path
import logging
from data.loader import load_nslkdd_dataset, process_nslkdd_dataset
from evaluation_config import MODELS, PREDICTIONS_DIR, DECISION_THRESHOLDS
from generate_advanced_models_preds import load_and_align_hard, make_test_dataset

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def collect_raw_predictions(model_dir, test_file, output_dir=PREDICTIONS_DIR):
    """Collect raw prediction probabilities from all models."""
    os.makedirs(output_dir, exist_ok=True)
    
    # Load Hard dataset
    PROJECT_ROOT = Path(__file__).parent.resolve()
    nsl_data_dir = str(PROJECT_ROOT / "data" / "nsl_kdd_dataset")
    X_test, y_test, _ = load_and_align_hard(nsl_data_dir, test_file)
    
    # Save ground truth
    np.save(os.path.join(output_dir, "y_true.npy"), y_test)
    
    # Load each model and get predictions
    for sub in sorted(os.listdir(model_dir)):
        sub_path = os.path.join(model_dir, sub)
        if not os.path.isdir(sub_path):
            continue
            
        model_name = sub.split('_')[0]  # Extract base model name
        logger.info(f"Collecting predictions for {model_name} ({sub})")
        
        # First check if this is a threshold detector
        if model_name == "threshold_detector":
            try:
                # Load thresholds
                thresholds_path = os.path.join(sub_path, "thresholds.json")
                if not os.path.isfile(thresholds_path):
                    logger.warning(f"No thresholds.json found for {sub}")
                    continue
                
                # Normalize data if normalization params exist
                try:
                    X_mean = np.load(os.path.join(sub_path, "X_mean.npy"))
                    X_std = np.load(os.path.join(sub_path, "X_std.npy"))
                    X_test_norm = (X_test - X_mean) / X_std
                except Exception as e:
                    logger.warning(f"No normalization params in {sub_path}: {e}")
                    X_test_norm = X_test

                # Load detector and predict
                from models.threshold_detector import ThresholdDetector
                with open(thresholds_path, 'r') as f:
                    thresholds = json.load(f)

                detector = ThresholdDetector()
                detector.thresholds = {int(k): float(v) for k, v in thresholds.items()}
                detector.is_calibrated = True

                # Define feature indices to use
                feature_indices = list(range(min(X_test_norm.shape[1], len(detector.thresholds))))

                # Generate raw predictions
                y_pred = detector.predict(X_test_norm, feature_indices)

                # Save binary predictions
                np.save(os.path.join(output_dir, f"{model_name}_{sub.split('_')[1]}_y_pred.npy"), y_pred)
                np.save(os.path.join(output_dir, f"{model_name}_{sub.split('_')[1]}_y_true.npy"), y_test)

                # Generate pseudoprobabilities for compatibility with other evaluations
                # Create a scoring function: distance from threshold
                scores = np.zeros(len(X_test_norm))
                for idx in feature_indices:
                    if idx < X_test_norm.shape[1] and idx < len(detector.thresholds):
                        threshold = detector.thresholds[idx]
                        # How much the feature exceeds its threshold
                        excess = X_test_norm[:, idx] - threshold
                        # Update scores for samples that exceed threshold
                        scores = np.maximum(scores, excess)

                # Scale to 0-1 range
                scores = 1 / (1 + np.exp(-scores))  # Sigmoid transform

                # Save raw scores
                np.save(os.path.join(output_dir, f"{model_name}_{sub.split('_')[1]}_raw_probs.npy"), scores)

                # Generate binary predictions for each threshold
                for threshold in DECISION_THRESHOLDS:
                    y_pred_t = (scores >= threshold).astype(np.int32)
                    np.save(os.path.join(output_dir, f"{model_name}_{sub.split('_')[1]}_pred_t{threshold:.2f}.npy"), y_pred_t)

                logger.info(f"Successfully processed threshold detector {sub}")
            except Exception as e:
                logger.error(f"Error processing threshold detector {sub}: {e}")
            continue  # Skip the neural network model loading
            
        # For neural network models
        try:
            model = tf.keras.models.load_model(sub_path, compile=False)
            
            # Create appropriate dataset
            ds = make_test_dataset(X_test, y_test, model, 256, sub_path)
            
            # Get raw probabilities
            y_probs = model.predict(ds).flatten()
            np.save(os.path.join(output_dir, f"{model_name}_{sub.split('_')[1]}_raw_probs.npy"), y_probs)
            
            # Generate binary predictions for each threshold
            for threshold in DECISION_THRESHOLDS:
                y_pred = (y_probs >= threshold).astype(np.int32)
                np.save(os.path.join(output_dir, f"{model_name}_{sub.split('_')[1]}_pred_t{threshold:.2f}.npy"), y_pred)
                
        except Exception as e:
            logger.error(f"Error collecting predictions for {sub}: {e}")
    
    logger.info(f"All predictions saved to {output_dir}")
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Collect raw prediction probabilities from all models")
    parser.add_argument("--test-file", required=True, help="Path to NSL-KDD-Hard.csv")
    parser.add_argument("--model-dir", default="saved_models", help="Directory containing saved models")
    args = parser.parse_args()
    
    collect_raw_predictions(args.model_dir, args.test_file)