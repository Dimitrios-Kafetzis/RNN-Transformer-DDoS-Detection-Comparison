"""
Main script to run the DoS detection project.
Supports both Bot-IoT and NSL-KDD datasets, with options for hping3 testing.
"""

import os
import argparse
import logging
import tensorflow as tf
import numpy as np
import pandas as pd
import json
from typing import Dict, Any

# Import project modules
from config import *
from data.loader import (
    load_bot_iot_dataset, 
    load_nslkdd_dataset,
    create_tf_dataset, 
    create_sequence_dataset
)
from data.preprocessor import (
    preprocess_bot_iot_dataset,
    preprocess_nsl_kdd_dataset
)
from models.shallow_dnn import create_shallow_dnn_model
from models.dnn import create_dnn_model
from models.lstm import create_lstm_model
from models.gru import create_gru_model
from models.transformer_OLD import create_transformer_model
from training.trainer_OLD import train_all_models
from evaluation.evaluator import evaluate_all_models
from inference.real_time_detector import detect_dos_attacks

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("dos_detection.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

def setup_argparse():
    """Set up command line arguments."""
    parser = argparse.ArgumentParser(description="DoS Detection using multiple datasets")
    
    parser.add_argument("--dataset", type=str, choices=['bot_iot', 'nsl_kdd'], default='nsl_kdd',
                      help="Dataset to use for training and evaluation (default: nsl_kdd)")
    parser.add_argument("--data_dir", type=str, default=DATA_DIR,
                       help="Directory containing the dataset")
    parser.add_argument("--processed_data_dir", type=str, default=PROCESSED_DATA_DIR,
                       help="Directory to save processed data")
    parser.add_argument("--model_save_dir", type=str, default=MODEL_SAVE_DIR,
                       help="Directory to save trained models")
    parser.add_argument("--plot_save_dir", type=str, default="plots",
                       help="Directory to save plots")
    parser.add_argument("--sample", action="store_true",
                       help="Use a smaller sample of the dataset for faster development")
    parser.add_argument("--epochs", type=int, default=EPOCHS,
                       help="Number of training epochs")
    parser.add_argument("--batch_size", type=int, default=BATCH_SIZE,
                       help="Batch size for training and evaluation")
    parser.add_argument("--skip_training", action="store_true",
                       help="Skip training and only evaluate existing models")
    parser.add_argument("--skip_evaluation", action="store_true",
                       help="Skip evaluation and only train models")
    parser.add_argument("--load_models", action="store_true",
                       help="Load existing models instead of creating new ones")
    parser.add_argument("--models", type=str, nargs="+", default=["shallow_dnn", "dnn", "lstm", "gru", "transformer"],
                       help="List of models to train/evaluate")
    
    # Add NSL-KDD specific options
    parser.add_argument("--target_type", type=str, choices=['binary', 'multiclass', 'dos_specific'], 
                       default='binary', help="Target type for NSL-KDD dataset")
    
    # Add hping3 inference options
    parser.add_argument("--inference", action="store_true",
                       help="Run inference on a PCAP file from hping3")
    parser.add_argument("--pcap_file", type=str,
                       help="PCAP file to analyze (for inference mode)")
    parser.add_argument("--inference_model", type=str,
                       help="Model to use for inference (default: best performing model)")
    parser.add_argument("--threshold", type=float, default=0.5,
                       help="Detection threshold for inference")
    parser.add_argument("--results_dir", type=str, default="results",
                       help="Directory to save inference results")
    
    # Add benchmark options
    parser.add_argument("--convert_to_tflite", action="store_true",
                       help="Convert models to TFLite format after training")
    parser.add_argument("--tflite_dir", type=str, default="tflite_models",
                       help="Directory to save TFLite models")
    parser.add_argument("--quantize", action="store_true",
                       help="Apply quantization when converting to TFLite")
    parser.add_argument("--benchmark", action="store_true",
                       help="Run benchmarks on trained models")
    parser.add_argument("--benchmark_dir", type=str, default="benchmark_results",
                       help="Directory to save benchmark results")
    parser.add_argument("--early_warning", action="store_true",
                       help="Evaluate early warning capability")
    parser.add_argument("--early_warning_dir", type=str, default="early_warning_results",
                       help="Directory to save early warning results")
    
    return parser.parse_args()

def load_and_preprocess_bot_iot_data(args):
    """Load and preprocess the Bot-IoT dataset."""
    logger.info("Loading and preprocessing Bot-IoT data...")
    
    # Load raw data
    df = load_bot_iot_dataset(args.data_dir, sample=args.sample)
    
    # Preprocess data
    preprocessed_data = preprocess_bot_iot_dataset(
        df=df,
        train_ratio=TRAIN_RATIO,
        val_ratio=VAL_RATIO,
        test_ratio=TEST_RATIO,
        scaler_type="standard",
        imbalance_strategy="class_weight",
        add_engineered_features=True,
        save_dir=args.processed_data_dir,
        random_state=RANDOM_SEED
    )
    
    # Create TensorFlow datasets
    train_dataset = create_tf_dataset(
        features=preprocessed_data["X_train"],
        labels=preprocessed_data["y_train"],
        batch_size=args.batch_size,
        shuffle=True,
        buffer_size=SHUFFLE_BUFFER
    )
    
    val_dataset = create_tf_dataset(
        features=preprocessed_data["X_val"],
        labels=preprocessed_data["y_val"],
        batch_size=args.batch_size,
        shuffle=False
    )
    
    test_dataset = create_tf_dataset(
        features=preprocessed_data["X_test"],
        labels=preprocessed_data["y_test"],
        batch_size=args.batch_size,
        shuffle=False
    )
    
    # Create sequence datasets for recurrent models
    train_sequence_dataset = create_sequence_dataset(
        features=preprocessed_data["X_train"],
        labels=preprocessed_data["y_train"],
        window_size=WINDOW_SIZE,
        step_size=STEP_SIZE,
        batch_size=args.batch_size,
        shuffle=True,
        buffer_size=SHUFFLE_BUFFER
    )
    
    val_sequence_dataset = create_sequence_dataset(
        features=preprocessed_data["X_val"],
        labels=preprocessed_data["y_val"],
        window_size=WINDOW_SIZE,
        step_size=STEP_SIZE,
        batch_size=args.batch_size,
        shuffle=False
    )
    
    test_sequence_dataset = create_sequence_dataset(
        features=preprocessed_data["X_test"],
        labels=preprocessed_data["y_test"],
        window_size=WINDOW_SIZE,
        step_size=STEP_SIZE,
        batch_size=args.batch_size,
        shuffle=False
    )
    
    data = {
        "X_train": preprocessed_data["X_train"],
        "X_val": preprocessed_data["X_val"],
        "X_test": preprocessed_data["X_test"],
        "y_train": preprocessed_data["y_train"],
        "y_val": preprocessed_data["y_val"],
        "y_test": preprocessed_data["y_test"],
        "train_dataset": train_dataset,
        "val_dataset": val_dataset,
        "test_dataset": test_dataset,
        "train_sequence_dataset": train_sequence_dataset,
        "val_sequence_dataset": val_sequence_dataset,
        "test_sequence_dataset": test_sequence_dataset,
        "class_weights": preprocessed_data["class_weights"],
        "preprocessing_info": preprocessed_data["preprocessing_info"]
    }
    
    logger.info("Bot-IoT data loading and preprocessing completed")
    return data

def load_and_preprocess_nslkdd_data(args):
    """Load and preprocess the NSL-KDD dataset."""
    logger.info("Loading and preprocessing NSL-KDD data...")
    
    # Load raw data
    train_df, test_df = load_nslkdd_dataset(args.data_dir, split="both")
    
    # Combine train and test for consistent preprocessing if needed
    if args.sample:
        # Sample from both train and test sets
        combined_df = pd.concat([train_df, test_df], ignore_index=True)
        # Shuffle and take a sample
        combined_df = combined_df.sample(n=min(len(combined_df), 10000), random_state=RANDOM_SEED)
        # Split back into train and test
        train_size = int(0.8 * len(combined_df))
        train_df = combined_df[:train_size]
        test_df = combined_df[train_size:]
    
    # Preprocess the training data
    preprocessed_data = preprocess_nsl_kdd_dataset(
        df=train_df,
        target_type=args.target_type,
        train_ratio=TRAIN_RATIO,
        val_ratio=VAL_RATIO,
        test_ratio=TEST_RATIO,
        scaler_type="standard",
        imbalance_strategy="class_weight",
        add_engineered_features=True,
        save_dir=args.processed_data_dir,
        random_state=RANDOM_SEED
    )
    
    # Create TensorFlow datasets
    train_dataset = create_tf_dataset(
        features=preprocessed_data["X_train"],
        labels=preprocessed_data["y_train"],
        batch_size=args.batch_size,
        shuffle=True,
        buffer_size=SHUFFLE_BUFFER
    )
    
    val_dataset = create_tf_dataset(
        features=preprocessed_data["X_val"],
        labels=preprocessed_data["y_val"],
        batch_size=args.batch_size,
        shuffle=False
    )
    
    test_dataset = create_tf_dataset(
        features=preprocessed_data["X_test"],
        labels=preprocessed_data["y_test"],
        batch_size=args.batch_size,
        shuffle=False
    )
    
    # Create sequence datasets for recurrent models
    window_size = WINDOW_SIZE
    step_size = STEP_SIZE
    
    # For NSL-KDD, we might need shorter sequences
    if args.dataset == 'nsl_kdd':
        window_size = min(WINDOW_SIZE, 5)  # Use smaller window for NSL-KDD
        step_size = min(STEP_SIZE, 2)
    
    train_sequence_dataset = create_sequence_dataset(
        features=preprocessed_data["X_train"],
        labels=preprocessed_data["y_train"],
        window_size=window_size,
        step_size=step_size,
        batch_size=args.batch_size,
        shuffle=True,
        buffer_size=SHUFFLE_BUFFER
    )
    
    val_sequence_dataset = create_sequence_dataset(
        features=preprocessed_data["X_val"],
        labels=preprocessed_data["y_val"],
        window_size=window_size,
        step_size=step_size,
        batch_size=args.batch_size,
        shuffle=False
    )
    
    test_sequence_dataset = create_sequence_dataset(
        features=preprocessed_data["X_test"],
        labels=preprocessed_data["y_test"],
        window_size=window_size,
        step_size=step_size,
        batch_size=args.batch_size,
        shuffle=False
    )
    
    data = {
        "X_train": preprocessed_data["X_train"],
        "X_val": preprocessed_data["X_val"],
        "X_test": preprocessed_data["X_test"],
        "y_train": preprocessed_data["y_train"],
        "y_val": preprocessed_data["y_val"],
        "y_test": preprocessed_data["y_test"],
        "train_dataset": train_dataset,
        "val_dataset": val_dataset,
        "test_dataset": test_dataset,
        "train_sequence_dataset": train_sequence_dataset,
        "val_sequence_dataset": val_sequence_dataset,
        "test_sequence_dataset": test_sequence_dataset,
        "class_weights": preprocessed_data["class_weights"],
        "preprocessing_info": preprocessed_data["preprocessing_info"],
        "target_type": preprocessed_data.get("target_type", "binary"),
        "label_mapping": preprocessed_data.get("label_mapping", None)
    }
    
    logger.info("NSL-KDD data loading and preprocessing completed")
    return data

def create_or_load_models(args, data):
    """Create or load models based on command line arguments."""
    models = {}
    
    if args.dataset == 'bot_iot':
        input_dim = data["X_train"].shape[1]
    else:  # NSL-KDD
        input_dim = data["X_train"].shape[1]
    
    # For sequence models
    window_size = WINDOW_SIZE
    if args.dataset == 'nsl_kdd':
        window_size = min(WINDOW_SIZE, 5)  # Smaller window for NSL-KDD
    
    sequence_input_shape = (window_size, input_dim)
    
    if args.load_models:
        # Load existing models
        logger.info("Loading existing models...")
        for model_name in args.models:
            model_path = os.path.join(args.model_save_dir, f"{model_name}_model.h5")
            if os.path.exists(model_path):
                models[model_name] = tf.keras.models.load_model(model_path)
                logger.info(f"Loaded {model_name} model from {model_path}")
            else:
                logger.warning(f"Model {model_path} not found. Creating a new one.")
                models[model_name] = create_model(model_name, input_dim, sequence_input_shape, args.dataset)
    else:
        # Create new models
        logger.info("Creating models...")
        for model_name in args.models:
            models[model_name] = create_model(model_name, input_dim, sequence_input_shape, args.dataset)
    
    return models

def create_model(model_name, input_dim, sequence_input_shape, dataset_type):
    """Create a model based on the model name."""
    # Adjust parameters based on dataset
    if dataset_type == 'nsl_kdd':
        # For NSL-KDD, we might want to use different hyperparameters
        shallow_dnn_hidden = [128, 64]
        dnn_hidden = [256, 128, 64]
        dropout_rate = 0.4
    else:
        # Use default parameters for Bot-IoT
        shallow_dnn_hidden = SHALLOWDNN_HIDDEN_UNITS
        dnn_hidden = DNN_HIDDEN_UNITS
        dropout_rate = DNN_DROPOUT_RATE
    
    if model_name == "shallow_dnn":
        return create_shallow_dnn_model(
            input_dim=input_dim,
            hidden_units=shallow_dnn_hidden,
            dropout_rate=dropout_rate,
            l2_reg=0.001,
            learning_rate=LEARNING_RATE,
            name="shallow_dnn_model"
        )
    elif model_name == "dnn":
        return create_dnn_model(
            input_dim=input_dim,
            hidden_units=dnn_hidden,
            dropout_rate=dropout_rate,
            l2_reg=0.001,
            learning_rate=LEARNING_RATE,
            name="dnn_model"
        )
    elif model_name == "lstm":
        return create_lstm_model(
            input_shape=sequence_input_shape,
            lstm_units=LSTM_UNITS,
            dropout_rate=LSTM_DROPOUT_RATE,
            recurrent_dropout_rate=0.1,
            dense_units=[32],
            l2_reg=0.001,
            learning_rate=LEARNING_RATE,
            bidirectional=True,
            name="lstm_model"
        )
    elif model_name == "gru":
        return create_gru_model(
            input_shape=sequence_input_shape,
            gru_units=GRU_UNITS,
            dropout_rate=GRU_DROPOUT_RATE,
            recurrent_dropout_rate=0.1,
            dense_units=[32],
            l2_reg=0.001,
            learning_rate=LEARNING_RATE,
            bidirectional=True,
            name="gru_model"
        )
    elif model_name == "transformer":
        return create_transformer_model(
            input_shape=sequence_input_shape,
            num_layers=TRANSFORMER_NUM_LAYERS,
            d_model=TRANSFORMER_D_MODEL,
            num_heads=TRANSFORMER_NUM_HEADS,
            dff=TRANSFORMER_DENSE_UNITS,
            dense_units=[32],
            dropout_rate=TRANSFORMER_DROPOUT_RATE,
            l2_reg=0.001,
            learning_rate=LEARNING_RATE,
            name="transformer_model"
        )
    else:
        raise ValueError(f"Unknown model name: {model_name}")

def export_classification_results(
    eval_results: Dict[str, Dict[str, Any]], 
    output_dir: str
):
    """
    Export classification metrics to be used in benchmarking.
    
    Args:
        eval_results: Dictionary of evaluation results from evaluate_all_models
        output_dir: Directory to save the results
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # Extract relevant metrics
    classification_metrics = {}
    
    for model_name, results in eval_results.items():
        classification_metrics[model_name] = {
            "accuracy": results["accuracy"],
            "precision": results["precision"],
            "recall": results["recall"],
            "f1_score": results["f1_score"],
            "roc_auc": results["roc_auc"]
        }
    
    # Save as JSON
    output_path = os.path.join(output_dir, "classification_metrics.json")
    with open(output_path, 'w') as f:
        json.dump(classification_metrics, f, indent=2)
    
    logger.info(f"Classification metrics saved to {output_path}")
    return classification_metrics

def convert_models_to_tflite(args, models):
    """
    Convert trained models to TensorFlow Lite format.
    
    Args:
        args: Command line arguments
        models: Dictionary of trained models
    """
    # Create TFLite directory
    os.makedirs(args.tflite_dir, exist_ok=True)
    
    try:
        from utils.tflite_converter import convert_all_models
        
        logger.info("Converting models to TFLite format...")
        
        # Get list of model files from models dictionary
        model_files = []
        for model_name in models.keys():
            model_path = os.path.join(args.model_save_dir, f"{model_name}_model.h5")
            if os.path.exists(model_path):
                model_files.append(model_path)
        
        # Convert models
        converted_paths = convert_all_models(
            models_dir=args.model_save_dir,
            output_dir=args.tflite_dir,
            dataset_type=args.dataset,
            data_dir=args.data_dir,
            model_types=list(models.keys()),
            quantize=args.quantize,
            optimize_for_latency=True
        )
        
        logger.info(f"Converted {len(converted_paths)} models to TFLite format")
        return converted_paths
    
    except ImportError:
        logger.error("Could not import tflite_converter module. Make sure it's in the utils directory.")
        return []

def run_inference(args, eval_results=None):
    """
    Run inference on a PCAP file for DoS detection.
    
    Args:
        args: Command line arguments
        eval_results: Evaluation results to determine best model (optional)
    """
    if not args.pcap_file:
        logger.error("No PCAP file specified for inference")
        return
    
    # Create results directory
    os.makedirs(args.results_dir, exist_ok=True)
    
    # Determine which model to use
    model_name = args.inference_model
    if not model_name:
        if eval_results:
            # Find the best performing model
            best_model = max(eval_results.items(), key=lambda x: x[1]["f1_score"])[0]
            model_name = best_model
            logger.info(f"Using best performing model for inference: {model_name}")
        else:
            # Default to DNN if no evaluation results
            model_name = "dnn"
            logger.info(f"Using default model for inference: {model_name}")
    
    # Paths for model and artifacts
    model_path = os.path.join(args.model_save_dir, f"{model_name}_model.h5")
    
    # Check if model exists
    if not os.path.exists(model_path):
        logger.error(f"Model file not found: {model_path}")
        return
    
    # Define results file path
    results_file = os.path.join(args.results_dir, 
                               f"{os.path.basename(args.pcap_file).split('.')[0]}_results.json")
    
    # Run detection
    logger.info(f"Running inference on {args.pcap_file} using {model_name} model")
    results = detect_dos_attacks(
        pcap_file=args.pcap_file,
        model_path=model_path,
        model_artifacts_dir=args.processed_data_dir,
        model_type=model_name,
        dataset_type=args.dataset,
        threshold=args.threshold,
        output_file=results_file
    )
    
    # Print summary
    print("\n" + "="*80)
    print("DoS ATTACK DETECTION RESULTS".center(80))
    print("="*80)
    print(f"PCAP file: {args.pcap_file}")
    print(f"Model used: {model_name}")
    print(f"Total flows analyzed: {results['total_count']}")
    print(f"Flows classified as attacks: {results['attack_count']} ({results['attack_ratio']*100:.2f}%)")
    print(f"Attack detected: {'YES' if results['is_attack_detected'] else 'NO'}")
    print(f"Results saved to: {results_file}")
    print("="*80)

def run_benchmarks(args, tflite_models=None):
    """
    Run benchmarks on TFLite models.
    
    Args:
        args: Command line arguments
        tflite_models: List of TFLite model paths (optional)
    """
    # Create benchmark directory
    os.makedirs(args.benchmark_dir, exist_ok=True)
    
    try:
        # Import dynamically to avoid issues if modules aren't available
        from inference.pi_benchmark import run_all_benchmarks
        
        logger.info("Running model benchmarks...")
        
        # If no TFLite models provided, look in the TFLite directory
        if not tflite_models:
            tflite_dir = args.tflite_dir
            if not os.path.exists(tflite_dir):
                logger.error(f"TFLite directory not found: {tflite_dir}")
                return
            
            # Find all TFLite models for specified model types
            tflite_models = []
            for model_type in args.models:
                model_file = os.path.join(tflite_dir, f"{model_type}_model.tflite")
                if os.path.exists(model_file):
                    tflite_models.append(model_file)
                else:
                    # Try alternative naming convention
                    model_file = os.path.join(tflite_dir, f"{model_type}.tflite")
                    if os.path.exists(model_file):
                        tflite_models.append(model_file)
        
        if not tflite_models:
            logger.error("No TFLite models found for benchmarking")
            return
        
        # Run benchmarks
        benchmark_results = run_all_benchmarks(
            models_dir=os.path.dirname(tflite_models[0]),  # Use directory of first model
            dataset_type=args.dataset,
            data_dir=args.data_dir,
            output_dir=args.benchmark_dir,
            model_types=args.models,
            sequence_length=5 if args.dataset == 'nsl_kdd' else 10,
            latency_runs=100,
            resource_duration=60  # Run for 1 minute per model
        )
        
        logger.info(f"Benchmarks completed for {len(benchmark_results)} models")
        return benchmark_results
    
    except ImportError as e:
        logger.error(f"Could not import benchmark modules: {e}")
        return None

def run_early_warning_evaluation(args, tflite_models=None):
    """
    Evaluate early warning capability of models.
    
    Args:
        args: Command line arguments
        tflite_models: List of TFLite model paths (optional)
    """
    # Create early warning directory
    os.makedirs(args.early_warning_dir, exist_ok=True)
    
    try:
        # Import early warning modules
        from inference.early_warning_generator import EarlyWarningScenarionGenerator, EarlyWarningEvaluator
        
        logger.info("Running early warning evaluation...")
        
        # Create scenarios directory
        scenarios_dir = os.path.join(args.early_warning_dir, "scenarios")
        os.makedirs(scenarios_dir, exist_ok=True)
        
        # Generate attack scenarios
        generator = EarlyWarningScenarionGenerator(
            dataset_type=args.dataset,
            data_dir=args.data_dir,
            output_dir=scenarios_dir
        )
        
        scenarios = generator.generate_scenarios(
            n_scenarios=5,
            min_flows=20,
            saturation_range=(0.6, 0.8)
        )
        
        # If no TFLite models provided, look in the TFLite directory
        if not tflite_models:
            tflite_dir = args.tflite_dir
            if not os.path.exists(tflite_dir):
                logger.error(f"TFLite directory not found: {tflite_dir}")
                return
            
            # Find all TFLite models for specified model types
            tflite_models = []
            for model_type in args.models:
                model_file = os.path.join(tflite_dir, f"{model_type}_model.tflite")
                if os.path.exists(model_file):
                    tflite_models.append(model_file)
                else:
                    # Try alternative naming convention
                    model_file = os.path.join(tflite_dir, f"{model_type}.tflite")
                    if os.path.exists(model_file):
                        tflite_models.append(model_file)
        
        if not tflite_models:
            logger.error("No TFLite models found for early warning evaluation")
            return
        
        # Evaluate models on scenarios
        evaluator = EarlyWarningEvaluator(
            scenarios_dir=scenarios_dir,
            dataset_type=args.dataset,
            data_dir=args.data_dir,
            output_dir=args.early_warning_dir
        )
        
        early_warning_results = evaluator.evaluate_all_models(
            model_paths=tflite_models,
            sequence_length=5 if args.dataset == 'nsl_kdd' else 10,
            threshold=args.threshold
        )
        
        logger.info("Early warning evaluation completed")
        return early_warning_results
    
    except ImportError as e:
        logger.error(f"Could not import early warning modules: {e}")
        return None

def run_comprehensive_analysis(args):
    """
    Run comprehensive analysis using aggregate_results module.
    
    Args:
        args: Command line arguments
    """
    try:
        from evaluation.aggregate_results import ResultsAggregator
        
        logger.info("Running comprehensive analysis...")
        
        # Create aggregator
        aggregator = ResultsAggregator(
            results_dir=args.benchmark_dir,
            output_dir=os.path.join(args.benchmark_dir, "analysis"),
            model_types=args.models
        )
        
        # Generate all outputs
        aggregator.generate_all_outputs()
        
        logger.info("Comprehensive analysis completed")
        
    except ImportError as e:
        logger.error(f"Could not import aggregate_results module: {e}")

def main():
    """Main function to run the project."""
    # Parse command line arguments
    args = setup_argparse()
    
    # Configure TensorFlow
    configure_tensorflow()
    
    # Make directories
    os.makedirs(args.data_dir, exist_ok=True)
    os.makedirs(args.processed_data_dir, exist_ok=True)
    os.makedirs(args.model_save_dir, exist_ok=True)
    os.makedirs(args.plot_save_dir, exist_ok=True)
    
    # Load and preprocess data based on dataset choice
    if args.dataset == 'bot_iot':
        data = load_and_preprocess_bot_iot_data(args)
    else:  # NSL-KDD
        data = load_and_preprocess_nslkdd_data(args)
    
    # Create or load models
    models = create_or_load_models(args, data)
    
    # Print data and model information
    logger.info(f"Data shapes: X_train {data['X_train'].shape}, y_train {data['y_train'].shape}")
    logger.info(f"Class distribution: {data['preprocessing_info']['class_distribution']}")
    logger.info(f"Models to train/evaluate: {list(models.keys())}")
    
    # Handle inference mode
    if args.inference:
        run_inference(args)
        return
    
    # Training parameters
    train_params = {
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "early_stopping_patience": EARLY_STOPPING_PATIENCE,
        "model_save_dir": args.model_save_dir,
        "plot_save_dir": args.plot_save_dir,
        "verbose": 1
    }
    
    # Evaluation parameters
    eval_params = {
        "batch_size": args.batch_size,
        "threshold": 0.5,
        "generate_plots": True,
        "compare_models": True,
        "plot_save_dir": args.plot_save_dir,
        "comparison_metrics": ["accuracy", "precision", "recall", "f1_score", "roc_auc", "pr_auc"]
    }
    
    # Train models
    train_results = None
    if not args.skip_training:
        logger.info("Training models...")
        train_results = train_all_models(models, data, train_params)
        logger.info("Training completed")
    
    # Evaluate models
    eval_results = None
    if not args.skip_evaluation:
        logger.info("Evaluating models...")
        eval_results = evaluate_all_models(models, data, eval_params)
        logger.info("Evaluation completed")
        
        # Export classification metrics for benchmarking
        if eval_results:
            export_classification_results(eval_results, args.benchmark_dir)
    
    # Convert models to TFLite if requested
    tflite_models = None
    if args.convert_to_tflite:
        tflite_models = convert_models_to_tflite(args, models)
    
    # Run benchmarks if requested
    if args.benchmark:
        run_benchmarks(args, tflite_models)
    
    # Run early warning evaluation if requested
    if args.early_warning:
        run_early_warning_evaluation(args, tflite_models)
    
    # Run comprehensive analysis if both benchmark and early warning were performed
    if args.benchmark and args.early_warning:
        run_comprehensive_analysis(args)
    
    # Run inference if requested
    if args.inference and args.pcap_file:
        run_inference(args, eval_results)
    
    logger.info(f"DoS detection project execution completed successfully")

if __name__ == "__main__":
    main()