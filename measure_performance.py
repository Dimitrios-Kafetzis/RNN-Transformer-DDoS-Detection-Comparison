# measure_performance.py
import os
import time
import numpy as np
import tensorflow as tf
import psutil
import argparse
import json
import logging
from pathlib import Path
from data.loader import load_nslkdd_dataset, process_nslkdd_dataset
from evaluation_config import MODELS, TIMING_DIR, MEMORY_DIR
from generate_advanced_models_preds import load_and_align_hard, make_test_dataset

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

def measure_model_performance(model_dir, test_file, num_runs=30, batch_size=32):
    """Measure execution time and memory usage for all models."""
    os.makedirs(TIMING_DIR, exist_ok=True)
    os.makedirs(MEMORY_DIR, exist_ok=True)
    
    process = psutil.Process(os.getpid())
    baseline_memory = process.memory_info().rss / (1024 * 1024)  # MB
    
    # Load Hard dataset
    PROJECT_ROOT = Path(__file__).parent.resolve()
    data_dir = PROJECT_ROOT / "data" / "nsl_kdd_dataset"
    
    # Measure feature extraction time
    start_time = time.perf_counter()
    X_test, y_test, _ = load_and_align_hard(str(data_dir), str(test_file))
    feature_extraction_time = time.perf_counter() - start_time
    
    # Record feature extraction time
    with open(os.path.join(TIMING_DIR, "feature_extraction.json"), 'w') as f:
        json.dump({"time_seconds": feature_extraction_time}, f)
    
    logger.info(f"Feature extraction time: {feature_extraction_time*1000:.2f} ms")
    
    results = {}
    
    # Load each model and measure performance
    for sub in sorted(os.listdir(model_dir)):
        sub_path = os.path.join(model_dir, sub)
        if not os.path.isdir(sub_path):
            continue
            
        model_name = sub.split('_')[0]  # Extract base model name
        logger.info(f"Measuring performance for {model_name} ({sub})")
        
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
                
                # Load detector
                from models.threshold_detector import ThresholdDetector
                with open(thresholds_path, 'r') as f:
                    thresholds = json.load(f)
                
                detector = ThresholdDetector()
                detector.thresholds = {int(k): float(v) for k, v in thresholds.items()}
                detector.is_calibrated = True
                
                # Measure inference time
                inference_times = []
                feature_indices = list(range(min(X_test_norm.shape[1], len(detector.thresholds))))
                
                for _ in range(num_runs):
                    start_time = time.perf_counter()
                    _ = detector.predict(X_test_norm, feature_indices)
                    inference_times.append(time.perf_counter() - start_time)
                
                # Measure peak memory during inference
                process.memory_info().rss  # Force memory info update
                start_memory = process.memory_info().rss / (1024 * 1024)
                
                _ = detector.predict(X_test_norm, feature_indices)
                
                peak_memory = process.memory_info().rss / (1024 * 1024)
                
                results[model_name] = {
                    "mean_latency_ms": np.mean(inference_times) * 1000,
                    "p95_latency_ms": np.percentile(inference_times, 95) * 1000,
                    "min_latency_ms": np.min(inference_times) * 1000,
                    "max_latency_ms": np.max(inference_times) * 1000,
                    "peak_memory_mb": peak_memory,
                    "memory_increase_mb": peak_memory - baseline_memory
                }

                logger.info(f"Successfully processed threshold detector {sub}")
            except Exception as e:
                logger.error(f"Error processing threshold detector {sub}: {e}")
            continue
        
        try:    
            # For neural network models
            model = tf.keras.models.load_model(sub_path, compile=False)
            
            # Get the expected input shape
            input_shape = model.input_shape
            
            # Prepare appropriate dataset for evaluation
            X_sample = X_test[:batch_size].copy()
            
            # Apply normalization if available
            try:
                X_mean = np.load(os.path.join(sub_path, "X_mean.npy"))
                X_std = np.load(os.path.join(sub_path, "X_std.npy"))
                X_sample = (X_sample - X_mean) / X_std
            except Exception as e:
                logger.warning(f"No normalization params: {e}")
            
            # Reshape input for sequence models
            if len(input_shape) > 2:  # This is a sequence model
                timesteps = input_shape[1]  # Usually 1 for our models
                features = input_shape[2]
                
                # Reshape to (batch_size, timesteps, features)
                if X_sample.shape[1] != features:
                    logger.warning(f"Feature dimension mismatch: model expects {features}, got {X_sample.shape[1]}")
                    # Truncate or pad features if needed
                    if X_sample.shape[1] > features:
                        X_sample = X_sample[:, :features]
                    else:
                        pad_width = features - X_sample.shape[1]
                        X_sample = np.pad(X_sample, ((0, 0), (0, pad_width)), mode='constant')
                
                X_sample = X_sample.reshape(-1, timesteps, X_sample.shape[1])
            
            # Warm-up run
            _ = model.predict(X_sample)
            
            # Measure inference time
            inference_times = []
            for _ in range(num_runs):
                start_time = time.perf_counter()
                _ = model.predict(X_sample)
                inference_times.append(time.perf_counter() - start_time)
            
            # Measure peak memory during inference
            process.memory_info().rss  # Force memory info update
            start_memory = process.memory_info().rss / (1024 * 1024)
            
            _ = model.predict(X_sample)
            
            peak_memory = process.memory_info().rss / (1024 * 1024)
            
            # Measure post-processing time (threshold application)
            post_times = []
            for _ in range(num_runs):
                y_prob = model.predict(X_sample)
                start_time = time.perf_counter()
                _ = (y_prob >= 0.5).astype(int)
                post_times.append(time.perf_counter() - start_time)
            
            results[model_name] = {
                "mean_latency_ms": np.mean(inference_times) * 1000,
                "p95_latency_ms": np.percentile(inference_times, 95) * 1000,
                "min_latency_ms": np.min(inference_times) * 1000,
                "max_latency_ms": np.max(inference_times) * 1000,
                "post_processing_ms": np.mean(post_times) * 1000,
                "peak_memory_mb": peak_memory,
                "memory_increase_mb": peak_memory - baseline_memory
            }
                
        except Exception as e:
            logger.error(f"Error measuring performance for {sub}: {e}")
    
    # Save results
    with open(os.path.join(TIMING_DIR, "inference_times.json"), 'w') as f:
        json.dump(results, f, indent=4)
    
    with open(os.path.join(MEMORY_DIR, "memory_usage.json"), 'w') as f:
        json.dump(results, f, indent=4)
    
    logger.info(f"Performance measurements saved to {TIMING_DIR} and {MEMORY_DIR}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Measure execution time and memory usage for all models")
    parser.add_argument("--test-file", required=True, help="Path to NSL-KDD-Hard.csv")
    parser.add_argument("--model-dir", default="saved_models", help="Directory containing saved models")
    parser.add_argument("--num-runs", type=int, default=30, help="Number of runs for timing measurements")
    args = parser.parse_args()
    
    measure_model_performance(args.model_dir, args.test_file, args.num_runs)