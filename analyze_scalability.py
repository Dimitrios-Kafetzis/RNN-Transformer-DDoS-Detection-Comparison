# analyze_scalability.py
import os
import time
import numpy as np
import tensorflow as tf
import psutil
import argparse
import json
import logging
from pathlib import Path
from sklearn.metrics import f1_score
from data.loader import load_nslkdd_dataset, process_nslkdd_dataset
from evaluation_config import TRAFFIC_RATES
from generate_advanced_models_preds import load_and_align_hard, make_test_dataset

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def analyze_scalability(model_dir, test_file, output_dir="evaluation_results/scalability"):
    """Analyze model performance under different traffic rates."""
    os.makedirs(output_dir, exist_ok=True)
    
    # Load Hard dataset
    PROJECT_ROOT = Path(__file__).parent.resolve()
    nsl_data_dir = str(PROJECT_ROOT / "data" / "nsl_kdd_dataset")
    X_test, y_test, _ = load_and_align_hard(nsl_data_dir, test_file)
    
    process = psutil.Process(os.getpid())
    baseline_memory = process.memory_info().rss / (1024 * 1024)  # MB
    
    results = {}
    
    # Load each model and measure performance at different traffic rates
    for sub in sorted(os.listdir(model_dir)):
        sub_path = os.path.join(model_dir, sub)
        if not os.path.isdir(sub_path):
            continue
            
        model_name = sub.split('_')[0]  # Extract base model name
        
        try:
            # Special case for threshold detector
            if model_name == "threshold_detector":
                # Load thresholds
                thresholds_path = os.path.join(sub_path, "thresholds.json")
                if not os.path.isfile(thresholds_path):
                    logger.warning(f"No thresholds.json found for {sub}")
                    continue
                
                # Load detector
                from models.threshold_detector import ThresholdDetector
                with open(thresholds_path, 'r') as f:
                    thresholds = json.load(f)
                
                detector = ThresholdDetector()
                detector.thresholds = {int(k): float(v) for k, v in thresholds.items()}
                detector.is_calibrated = True
                
                # Apply normalization if available
                try:
                    X_mean = np.load(os.path.join(sub_path, "X_mean.npy"))
                    X_std = np.load(os.path.join(sub_path, "X_std.npy"))
                    X_test_norm = (X_test - X_mean) / X_std
                except Exception as e:
                    logger.warning(f"No normalization params for {sub}: {e}")
                    X_test_norm = X_test.copy()
                
                model_results = {}
                feature_indices = list(range(min(X_test_norm.shape[1], len(detector.thresholds))))
                
                # Test at different traffic rates
                for rate in TRAFFIC_RATES:
                    # Simulate packets per second by calculating batch size
                    batch_size = min(256, max(16, rate // 10))  # Adjust batch size based on rate
                    
                    # Prepare input sample
                    X_sample = X_test_norm[:batch_size].copy()
                    y_sample = y_test[:batch_size].copy()
                    
                    # Warm-up run
                    _ = detector.predict(X_sample, feature_indices)
                    
                    # Measure inference time
                    start_time = time.perf_counter()
                    y_pred = detector.predict(X_sample, feature_indices)
                    inference_time = time.perf_counter() - start_time
                    
                    # Calculate throughput
                    throughput = len(X_sample) / inference_time  # samples per second
                    
                    # Measure memory
                    peak_memory = process.memory_info().rss / (1024 * 1024)
                    
                    # Calculate F1 score
                    f1 = f1_score(y_sample, y_pred)
                    
                    model_results[rate] = {
                        "latency_ms": inference_time * 1000,
                        "throughput_pps": throughput,
                        "memory_mb": peak_memory,
                        "f1_score": float(f1)
                    }
                    
                    logger.info(f"{model_name} at {rate} pps: F1={f1:.3f}, Latency={inference_time*1000:.2f}ms")
                
                results[model_name] = model_results
                continue
                
        except Exception as e:
            logger.error(f"Error analyzing scalability for {sub}: {e}")
    
    # Save results
    with open(os.path.join(output_dir, "scalability_results.json"), 'w') as f:
        json.dump(results, f, indent=4)
    
    logger.info(f"Scalability analysis saved to {output_dir}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Analyze model performance under different traffic rates")
    parser.add_argument("--test-file", required=True, help="Path to NSL-KDD-Hard.csv")
    parser.add_argument("--model-dir", default="saved_models", help="Directory containing saved models")
    args = parser.parse_args()
    
    analyze_scalability(args.model_dir, args.test_file)