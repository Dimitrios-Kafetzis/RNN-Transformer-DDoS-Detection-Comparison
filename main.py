#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
File: main.py
Author: Dimitrios Kafetzis (kafetzis@aueb.gr)
Created: May 2025
License: MIT

Description:
    Main entry point for the RNN-Transformer-DDoS-Comparison framework.
    Provides three operational modes:
    1. Training and evaluation pipeline for comparing model architectures
    2. Inference mode for detecting attacks in PCAP traffic files
    3. Generate mode for creating synthetic PCAP files with controlled attack patterns
    
    In training mode, handles dataset loading, model creation/training,
    and comprehensive evaluation across multiple metrics.
    
    In inference mode, processes PCAP files into compatible format,
    performs detection using trained models, and generates detailed
    attack reports.
    
    In generate mode, creates synthetic PCAP files with various attack
    patterns for testing the inference pipeline.

Usage:
    # Training and evaluation mode:
    $ python main.py train --dataset nsl_kdd [options]
    
    Examples:
    $ python main.py train --dataset nsl_kdd --data_dir data/nsl_kdd_dataset
    $ python main.py train --dataset nsl_kdd --skip_training --models dnn lstm gru transformer
    
    # Inference mode:
    $ python main.py infer --pcap-file traffic.pcap [options]
    
    Examples:
    $ python main.py infer --pcap-file captures/sample.pcap --model gru
    $ python main.py infer --pcap-file captures/sample.pcap --model-dir saved_models/gru_1746776936
    $ python main.py infer --pcap-file captures/sample.pcap --report-dir reports/sample --threshold 0.3
    
    # Generate mode:
    $ python main.py generate --output captures/test.pcap [options]
    
    Examples:
    $ python main.py generate --output captures/syn_flood.pcap --attack-type syn_flood
    $ python main.py generate --output captures/mixed.pcap --duration 600 --attack-duration 300
"""

import os
import sys
import argparse
import logging
from pathlib import Path
import glob
import time
import json

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("ddos_detector.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

def setup_train_args(parser):
    """Add arguments for training mode."""
    parser.add_argument("--dataset", type=str, choices=['nsl_kdd'], default='nsl_kdd',
                       help="Dataset to use for training and evaluation (default: nsl_kdd)")
    parser.add_argument("--data-dir", type=str, default="data/nsl_kdd_dataset",
                       help="Directory containing the dataset")
    parser.add_argument("--processed-data-dir", type=str, default="data/nsl_kdd_dataset",
                       help="Directory to save processed data")
    parser.add_argument("--model-save-dir", type=str, default="saved_models",
                       help="Directory to save trained models")
    parser.add_argument("--plot-save-dir", type=str, default="plots",
                       help="Directory to save plots")
    parser.add_argument("--sample", action="store_true",
                       help="Use a smaller sample of the dataset for faster development")
    parser.add_argument("--skip-training", action="store_true",
                       help="Skip training and only evaluate existing models")
    parser.add_argument("--skip-evaluation", action="store_true",
                       help="Skip evaluation and only train models")
    parser.add_argument("--eval-only", action="store_true",
                       help="Only run evaluation on existing models")
    parser.add_argument("--models", type=str, nargs="+", 
                       default=["shallow_dnn", "dnn", "lstm", "gru", "transformer", "threshold_detector"],
                       help="List of models to train/evaluate")
    parser.add_argument("--epochs", type=int, default=20,
                       help="Number of training epochs")
    parser.add_argument("--batch-size", type=int, default=64,
                       help="Batch size for training and evaluation")

def setup_infer_args(parser):
    """Add arguments for inference mode."""
    parser.add_argument("--pcap-file", type=str, required=True,
                       help="PCAP file to analyze")
    parser.add_argument("--model", type=str, default="gru",
                       help="Model type to use for inference (default: gru)")
    parser.add_argument("--model-dir", type=str,
                       help="Specific model directory to use (if not specified, latest model of type will be used)")
    # Add the missing argument
    parser.add_argument("--model-save-dir", type=str, default="saved_models",
                       help="Directory containing saved models (default: saved_models)")
    parser.add_argument("--threshold", type=float, default=0.5,
                       help="Detection threshold (default: 0.5)")
    parser.add_argument("--report-dir", type=str, default="reports",
                       help="Directory to save inference reports")
    parser.add_argument("--save-json", action="store_true",
                       help="Save report in JSON format")
    parser.add_argument("--window-size", type=int, default=10,
                       help="Number of packets to include in each analysis window")

def setup_generate_args(parser):
    """Add arguments for PCAP generation mode."""
    parser.add_argument("--output", type=str, required=True,
                       help="Path to output PCAP file")
    parser.add_argument("--duration", type=int, default=300,
                       help="Total duration of the capture in seconds (default: 300)")
    parser.add_argument("--attack-type", choices=["syn_flood", "udp_flood", "http_flood", 
                                                 "icmp_flood", "low_and_slow", "mixed"],
                       default="mixed", help="Type of attack to generate (default: mixed)")
    parser.add_argument("--attack-start", type=int, default=60,
                       help="When the attack starts in seconds (default: 60)")
    parser.add_argument("--attack-duration", type=int, default=120,
                       help="Duration of the attack in seconds (default: 120)")
    parser.add_argument("--normal-ratio", type=float, default=0.3,
                       help="Ratio of normal traffic during attack periods (default: 0.3)")
    parser.add_argument("--packet-rate", type=int, default=100,
                       help="Average packets per second (default: 100)")
    parser.add_argument("--victim-ip", default="192.168.1.100",
                       help="Target IP address for attacks (default: 192.168.1.100)")
    parser.add_argument("--seed", type=int, default=None,
                       help="Random seed for reproducibility")

def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="RNN-Transformer-DDoS-Comparison Framework")
    subparsers = parser.add_subparsers(dest="mode", help="Operation mode")
    
    # Training parser
    train_parser = subparsers.add_parser("train", help="Training and evaluation mode")
    setup_train_args(train_parser)
    
    # Inference parser
    infer_parser = subparsers.add_parser("infer", help="Inference mode")
    setup_infer_args(infer_parser)
    
    # Generate parser
    generate_parser = subparsers.add_parser("generate", help="Generate test PCAP files")
    setup_generate_args(generate_parser)
    
    return parser.parse_args()

def find_latest_model(model_type, model_dir):
    """Find the latest model of the specified type."""
    # Look for directories matching pattern model_type_*
    directories = glob.glob(os.path.join(model_dir, f"{model_type}_*"))
    
    if not directories:
        logger.error(f"No {model_type} models found in {model_dir}")
        return None
    
    # Sort by timestamp (assuming directory name format is model_type_timestamp)
    # and return the latest one
    return sorted(directories)[-1]

def run_training_pipeline(args):
    """Run the training and evaluation pipeline."""
    try:
        # Import necessary modules for training
        from trainer import main as train_main
        
        logger.info("Starting training pipeline")
        
        # Call trainer.py main function
        train_main()
        
        logger.info("Training completed successfully")
        
        # Run evaluation if not skipped
        if not args.skip_evaluation:
            from improved_run_evaluation import run_evaluation
            
            test_file = os.path.join(args.data_dir, "NSL-KDD-Hard.csv")
            if not os.path.exists(test_file):
                logger.warning(f"Test file {test_file} not found. Generating synthetic data.")
                
                # Generate synthetic data
                from gen_synth_nsld_kdd import generate
                generate(
                    train_csv=os.path.join(args.data_dir, "NSL-KDD-Train.csv"),
                    out_csv=test_file,
                    n_benign=5000,
                    n_attack=5000
                )
            
            # Run evaluation
            logger.info("Starting evaluation pipeline")
            run_evaluation(
                test_file=test_file,
                model_dir=args.model_save_dir,
                skip_steps=[],
                output_json="evaluation_results/complete_results.json"
            )
            logger.info("Evaluation completed successfully")
        
        return True
    
    except Exception as e:
        logger.error(f"Error in training pipeline: {e}", exc_info=True)
        return False

def run_inference_pipeline(args):
    """Run the inference pipeline."""
    try:
        # Ensure inference modules are imported
        try:
            from inference import pcap_to_nslkdd_features, run_inference, generate_text_report, save_json_report, print_summary
        except ImportError:
            logger.error("Inference modules not found. Please ensure the inference package is installed.")
            return False
        
        # Verify PCAP file exists
        if not os.path.exists(args.pcap_file):
            logger.error(f"PCAP file not found: {args.pcap_file}")
            return False
        
        # Determine model directory
        model_dir = args.model_dir
        if not model_dir:
            # Use hardcoded "saved_models" instead of args.model_save_dir
            model_dir = find_latest_model(args.model, "saved_models")
            if not model_dir:
                logger.error(f"No model directory found for type: {args.model}")
                return False
        
        logger.info(f"Using model: {model_dir}")
        
        # Create report directory if needed
        os.makedirs(args.report_dir, exist_ok=True)
        
        # Generate report filenames
        pcap_basename = os.path.splitext(os.path.basename(args.pcap_file))[0]
        timestamp = int(time.time())
        report_txt = os.path.join(args.report_dir, f"{pcap_basename}_{args.model}_{timestamp}.txt")
        report_json = os.path.join(args.report_dir, f"{pcap_basename}_{args.model}_{timestamp}.json")
        
        # Start inference process
        logger.info(f"Processing PCAP file: {args.pcap_file}")
        
        # Convert PCAP to NSL-KDD features
        features, timestamps = pcap_to_nslkdd_features(args.pcap_file, model_dir)
        
        if features.shape[0] == 0:
            logger.warning("No features extracted from PCAP file. Report will be empty.")
        
        # Run inference
        results = run_inference(
            features=features,
            timestamps=timestamps,
            model_path=model_dir,
            model_type=args.model,
            threshold=args.threshold
        )
        
        # Generate and save report
        report_text = generate_text_report(results, output_file=report_txt)
        
        # Save JSON if requested
        if args.save_json:
            save_json_report(results, report_json)
        
        # Print summary to console
        print_summary(results)
        
        logger.info(f"Inference completed successfully. Report saved to {report_txt}")
        return True
    
    except Exception as e:
        logger.error(f"Error in inference pipeline: {e}", exc_info=True)
        return False

def run_generate_pipeline(args):
    """Run the PCAP generation pipeline."""
    try:
        # Import PCAP generator
        try:
            from inference.pcap_generator import PcapGenerator
        except ImportError:
            logger.error("PCAP generator module not found. Please ensure the inference package is installed.")
            return False
        
        # Create the output directory if it doesn't exist
        os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
        
        # Create generator
        generator = PcapGenerator(
            output_file=args.output,
            duration=args.duration,
            attack_type=args.attack_type,
            attack_start=args.attack_start,
            attack_duration=args.attack_duration,
            normal_ratio=args.normal_ratio,
            packet_rate=args.packet_rate,
            victim_ip=args.victim_ip,
            random_seed=args.seed
        )
        
        # Generate PCAP file
        generator.generate()
        
        print("\n" + "=" * 80)
        print("PCAP GENERATION COMPLETE".center(80))
        print("=" * 80)
        print(f"Generated PCAP file: {args.output}")
        print(f"Duration: {args.duration} seconds")
        print(f"Attack type: {args.attack_type}")
        print(f"Attack period: {args.attack_start}s to {args.attack_start + args.attack_duration}s")
        print("=" * 80)
        
        return True
    
    except Exception as e:
        logger.error(f"Error in PCAP generation pipeline: {e}", exc_info=True)
        return False

def main():
    """Main function."""
    # Parse arguments
    args = parse_args()
    
    # Check mode
    if args.mode == "train":
        success = run_training_pipeline(args)
    elif args.mode == "infer":
        success = run_inference_pipeline(args)
    elif args.mode == "generate":
        success = run_generate_pipeline(args)
    else:
        print("Error: No mode specified. Use 'train', 'infer', or 'generate'.")
        print("Example: python main.py train --dataset nsl_kdd")
        print("Example: python main.py infer --pcap-file traffic.pcap --model gru")
        print("Example: python main.py generate --output captures/test.pcap --attack-type syn_flood")
        return 1
    
    return 0 if success else 1

if __name__ == "__main__":
    sys.exit(main())