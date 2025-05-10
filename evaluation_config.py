# evaluation_config.py
import os

# Output directories
RESULTS_DIR = "evaluation_results"
PLOTS_DIR = "plots/model_profiles"
HISTORY_DIR = os.path.join(RESULTS_DIR, "training_histories")
PREDICTIONS_DIR = os.path.join(RESULTS_DIR, "predictions")
TIMING_DIR = os.path.join(RESULTS_DIR, "timing")
MEMORY_DIR = os.path.join(RESULTS_DIR, "memory")

# Ensure directories exist
for dir_path in [RESULTS_DIR, PLOTS_DIR, HISTORY_DIR, PREDICTIONS_DIR, TIMING_DIR, MEMORY_DIR]:
    os.makedirs(dir_path, exist_ok=True)

# Model names
MODELS = ["threshold_detector", "linear_model", "shallow_dnn", "dnn", "lstm", "gru", "transformer"]

# Attack types (for NSL-KDD)
ATTACK_TYPES = {
    "normal": "Normal",
    "neptune": "TCP SYN Flood",
    "smurf": "UDP Flood",
    "teardrop": "UDP Flood",
    "pod": "ICMP Flood",
    "land": "Low-and-Slow",
    "back": "HTTP Flood",
    "apache2": "HTTP Flood"
}

# Evaluation parameters
TRAFFIC_RATES = [1000, 5000, 10000, 15000, 20000, 25000]  # packets per second
NUM_CROSS_VAL_RUNS = 10
BATCH_SIZE = 256
DECISION_THRESHOLDS = [0.01, 0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]