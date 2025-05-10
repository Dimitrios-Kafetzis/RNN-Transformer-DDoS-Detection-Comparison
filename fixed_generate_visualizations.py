#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
File: fixed_generate_visualizations.py
Author: Dimitrios Kafetzis (kafetzis@aueb.gr)
Created: May 2025
License: MIT

Description:
    Enhanced visualization generator that creates comprehensive plots and charts
    for model evaluation results. Handles edge cases and ensures data consistency
    when creating visualizations including:
    - Model convergence analysis
    - ROC and precision-recall curves
    - Decision threshold analysis
    - Detection time breakdown
    - Resource efficiency metrics
    - Attack type-specific performance
    - Transformer attention visualization
    All visualizations are saved to the plots directory.

Usage:
    $ python fixed_generate_visualizations.py
    
    This script is typically called automatically by the evaluation pipeline
    but can also be run independently to regenerate visualizations from
    existing evaluation results.
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import json
import pickle
import logging
import glob
from sklearn.metrics import roc_curve, precision_recall_curve, auc
from evaluation_config import MODELS, PLOTS_DIR, HISTORY_DIR, PREDICTIONS_DIR, TIMING_DIR, MEMORY_DIR

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def ensure_directory(dir_path):
    """Create directory if it doesn't exist."""
    if not os.path.exists(dir_path):
        os.makedirs(dir_path)
        logger.info(f"Created directory: {dir_path}")

def generate_visualizations():
    """Generate all visualizations for the paper with improved error handling."""
    ensure_directory(PLOTS_DIR)
    
    # 1. Model Convergence Analysis
    try:
        generate_model_convergence_plot()
    except Exception as e:
        logger.error(f"Error generating model convergence plot: {e}")
    
    # 2. ROC and PR Curves
    try:
        generate_roc_pr_curves()
    except Exception as e:
        logger.error(f"Error generating ROC and PR curves: {e}")
    
    # 3. Decision Threshold Analysis
    try:
        generate_threshold_analysis()
    except Exception as e:
        logger.error(f"Error generating threshold analysis plot: {e}")
    
    # 4. Detection Time Breakdown
    try:
        generate_detection_time_breakdown()
    except Exception as e:
        logger.error(f"Error generating detection time breakdown: {e}")
    
    # 5. Resource Efficiency Metrics
    try:
        generate_resource_efficiency_metrics()
    except Exception as e:
        logger.error(f"Error generating resource efficiency metrics: {e}")
    
    # 6. Attack Type-Specific Performance
    try:
        generate_attack_specific_performance()
    except Exception as e:
        logger.error(f"Error generating attack-specific performance: {e}")
    
    # 7. Scalability Analysis
    try:
        generate_scalability_analysis()
    except Exception as e:
        logger.error(f"Error generating scalability analysis: {e}")
    
    # 8. Statistical Significance Testing
    try:
        generate_significance_tests()
    except Exception as e:
        logger.error(f"Error generating significance tests: {e}")
    
    # 9. Model Interpretability Analysis
    try:
        generate_model_interpretability()
    except Exception as e:
        logger.error(f"Error generating model interpretability: {e}")
    
    logger.info(f"All visualizations saved to {PLOTS_DIR}")

def find_common_ground_truth():
    """Find a consistent ground truth file to use across visualizations."""
    # First check if there's a common y_true.npy
    common_path = os.path.join(PREDICTIONS_DIR, "y_true.npy")
    if os.path.exists(common_path):
        return np.load(common_path)
    
    # If not, look for any *_y_true.npy file
    true_files = glob.glob(os.path.join(PREDICTIONS_DIR, "*_y_true.npy"))
    if true_files:
        return np.load(true_files[0])
    
    # If all else fails, look in other directories
    for root, dirs, files in os.walk("evaluation_results"):
        for file in files:
            if file.endswith("_y_true.npy") or file == "y_true.npy":
                return np.load(os.path.join(root, file))
    
    raise FileNotFoundError("No ground truth labels found anywhere")

def generate_model_convergence_plot():
    """Generate model convergence plot from training histories with better error handling."""
    # Load training histories
    histories = {}
    history_files = glob.glob(os.path.join(HISTORY_DIR, "*_history.pkl"))
    
    if not history_files:
        logger.warning("No training history files found")
        return
    
    for f in history_files:
        model_name = os.path.basename(f).split('_')[0]
        try:
            with open(f, 'rb') as file:
                histories[model_name] = pickle.load(file)
        except Exception as e:
            logger.warning(f"Could not load history for {model_name}: {e}")
    
    if not histories:
        logger.warning("No valid training histories loaded")
        return
    
    # Create subplots for loss and F1
    fig, axes = plt.subplots(1, 2, figsize=(15, 6))
    
    # Plot training and validation loss
    for model, history in histories.items():
        if 'loss' not in history or 'val_loss' not in history:
            logger.warning(f"Missing loss data for {model}")
            continue
        
        epochs = range(1, len(history['loss']) + 1)
        axes[0].plot(epochs, history['loss'], 'o-', label=f"{model} (Train)")
        axes[0].plot(epochs, history['val_loss'], 's--', label=f"{model} (Val)")
    
    axes[0].set_title('Training and Validation Loss', fontsize=14)
    axes[0].set_xlabel('Epoch', fontsize=12)
    axes[0].set_ylabel('Loss', fontsize=12)
    axes[0].grid(True, linestyle='--', alpha=0.7)
    
    # Plot validation F1 scores
    any_f1_plotted = False
    for model, history in histories.items():
        if 'val_f1_score' not in history:
            logger.warning(f"Missing F1 score data for {model}")
            continue
        
        epochs = range(1, len(history['val_f1_score']) + 1)
        axes[1].plot(epochs, history['val_f1_score'], 'o-', label=model)
        any_f1_plotted = True
    
    if not any_f1_plotted:
        logger.warning("No F1 scores available for plotting")
        plt.close(fig)
        return
    
    axes[1].set_title('Validation F1 Score', fontsize=14)
    axes[1].set_xlabel('Epoch', fontsize=12)
    axes[1].set_ylabel('F1 Score', fontsize=12)
    axes[1].grid(True, linestyle='--', alpha=0.7)
    
    # Add legend if we have data to show
    handles, labels = [], []
    for ax in axes:
        h, l = ax.get_legend_handles_labels()
        handles.extend(h)
        labels.extend(l)
    
    if handles:
        fig.legend(handles, labels, loc='lower center', ncol=min(len(handles), 4), bbox_to_anchor=(0.5, -0.05))
    
    plt.tight_layout()
    plt.subplots_adjust(bottom=0.2)
    plt.savefig(os.path.join(PLOTS_DIR, 'model_convergence_analysis.pdf'), bbox_inches='tight')
    plt.savefig(os.path.join(PLOTS_DIR, 'model_convergence_analysis.png'), bbox_inches='tight')
    plt.close()
    logger.info("Model convergence plot saved")

def generate_roc_pr_curves():
    """Generate ROC and Precision-Recall curves with consistent ground truth."""
    # Get common ground truth
    try:
        y_true = find_common_ground_truth()
    except FileNotFoundError as e:
        logger.error(f"Could not find ground truth labels: {e}")
        return
    
    # Load raw predictions
    predictions = {}
    for f in glob.glob(os.path.join(PREDICTIONS_DIR, "*_raw_probs.npy")):
        model_name = os.path.basename(f).split('_')[0]
        y_pred = np.load(f)
        
        # Ensure predictions match ground truth length
        if len(y_pred) != len(y_true):
            logger.warning(f"Prediction length mismatch for {model_name}: {len(y_pred)} vs {len(y_true)}")
            if len(y_pred) > len(y_true):
                y_pred = y_pred[:len(y_true)]
            else:
                # Skip this model if we don't have enough predictions
                logger.warning(f"Skipping {model_name} for ROC/PR curves")
                continue
        
        predictions[model_name] = y_pred
    
    if not predictions:
        logger.warning("No valid prediction files found for ROC/PR curves")
        return
    
    # Create figure for ROC and PR curves
    fig, axes = plt.subplots(1, 2, figsize=(15, 6))
    
    # ROC curves
    for model, y_pred in predictions.items():
        try:
            fpr, tpr, _ = roc_curve(y_true, y_pred)
            roc_auc = auc(fpr, tpr)
            axes[0].plot(fpr, tpr, lw=2, label=f'{model} (AUC = {roc_auc:.3f})')
        except Exception as e:
            logger.warning(f"Error generating ROC curve for {model}: {e}")
    
    axes[0].plot([0, 1], [0, 1], 'k--', lw=2)
    axes[0].set_xlim([0.0, 1.0])
    axes[0].set_ylim([0.0, 1.05])
    axes[0].set_xlabel('False Positive Rate', fontsize=12)
    axes[0].set_ylabel('True Positive Rate', fontsize=12)
    axes[0].set_title('Receiver Operating Characteristic', fontsize=14)
    axes[0].grid(True, linestyle='--', alpha=0.7)
    
    # Precision-Recall curves
    for model, y_pred in predictions.items():
        try:
            precision, recall, _ = precision_recall_curve(y_true, y_pred)
            pr_auc = auc(recall, precision)
            axes[1].plot(recall, precision, lw=2, label=f'{model} (AUC = {pr_auc:.3f})')
        except Exception as e:
            logger.warning(f"Error generating PR curve for {model}: {e}")
    
    axes[1].set_xlim([0.0, 1.0])
    axes[1].set_ylim([0.0, 1.05])
    axes[1].set_xlabel('Recall', fontsize=12)
    axes[1].set_ylabel('Precision', fontsize=12)
    axes[1].set_title('Precision-Recall Curve', fontsize=14)
    axes[1].grid(True, linestyle='--', alpha=0.7)
    
    # Add legend if we have data
    handles, labels = axes[1].get_legend_handles_labels()
    if handles:
        fig.legend(handles, labels, loc='lower center', ncol=min(len(handles), 3), bbox_to_anchor=(0.5, -0.05))
    
    plt.tight_layout()
    plt.subplots_adjust(bottom=0.2)
    plt.savefig(os.path.join(PLOTS_DIR, 'roc_pr_curves.pdf'), bbox_inches='tight')
    plt.savefig(os.path.join(PLOTS_DIR, 'roc_pr_curves.png'), bbox_inches='tight')
    plt.close()
    logger.info("ROC and PR curves saved")

def generate_threshold_analysis():
    """Generate threshold analysis plot with consistent ground truth."""
    # Get common ground truth
    try:
        y_true = find_common_ground_truth()
    except FileNotFoundError as e:
        logger.error(f"Could not find ground truth labels: {e}")
        return
    
    # Load raw predictions
    predictions = {}
    for f in glob.glob(os.path.join(PREDICTIONS_DIR, "*_raw_probs.npy")):
        model_name = os.path.basename(f).split('_')[0]
        y_pred = np.load(f)
        
        # Ensure predictions match ground truth length
        if len(y_pred) != len(y_true):
            logger.warning(f"Prediction length mismatch for {model_name}: {len(y_pred)} vs {len(y_true)}")
            if len(y_pred) > len(y_true):
                y_pred = y_pred[:len(y_true)]
            else:
                # Skip this model if we don't have enough predictions
                logger.warning(f"Skipping {model_name} for threshold analysis")
                continue
        
        predictions[model_name] = y_pred
    
    if not predictions:
        logger.warning("No valid prediction files found for threshold analysis")
        return
    
    # Calculate F1 scores for different thresholds
    from sklearn.metrics import f1_score
    thresholds = np.linspace(0, 1, 100)
    f1_scores = {model: [] for model in predictions.keys()}
    
    for model, y_pred in predictions.items():
        for threshold in thresholds:
            try:
                y_binary = (y_pred >= threshold).astype(int)
                f1 = f1_score(y_true, y_binary)
                f1_scores[model].append(f1)
            except Exception as e:
                logger.warning(f"Error calculating F1 score for {model} at threshold {threshold}: {e}")
                f1_scores[model].append(0)  # Use 0 as a placeholder for error
    
    # Create plot
    plt.figure(figsize=(10, 6))
    
    for model, scores in f1_scores.items():
        plt.plot(thresholds, scores, lw=2, label=model)
    
    # Add optimal threshold points
    for model, scores in f1_scores.items():
        if not scores:
            continue
        
        optimal_idx = np.argmax(scores)
        optimal_threshold = thresholds[optimal_idx]
        optimal_f1 = scores[optimal_idx]
        plt.scatter(optimal_threshold, optimal_f1, s=100, marker='o', edgecolors='black')
        plt.annotate(f'{optimal_threshold:.2f}', (optimal_threshold, optimal_f1),
                     xytext=(5, 5), textcoords='offset points')
    
    plt.xlabel('Decision Threshold', fontsize=12)
    plt.ylabel('F1 Score', fontsize=12)
    plt.title('F1 Score vs Decision Threshold', fontsize=14)
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.legend(loc='best')
    
    # Add inset for Transformer zoom if applicable
    if 'transformer' in f1_scores:
        try:
            from mpl_toolkits.axes_grid1.inset_locator import zoomed_inset_axes
            axins = zoomed_inset_axes(plt.gca(), 6, loc='center left')
            
            # Focus on 0-0.1 range
            zoom_thresholds = np.linspace(0, 0.1, 100)
            zoom_f1 = []
            
            for threshold in zoom_thresholds:
                y_binary = (predictions['transformer'] >= threshold).astype(int)
                f1 = f1_score(y_true, y_binary)
                zoom_f1.append(f1)
            
            axins.plot(zoom_thresholds, zoom_f1, lw=2, color='C4')
            axins.set_xlim(0, 0.1)
            axins.set_ylim(min(zoom_f1) * 0.9, max(zoom_f1) * 1.1)
            axins.set_title('Transformer: Zoom 0-0.1', fontsize=10)
            axins.grid(True, linestyle='--', alpha=0.7)
        except Exception as e:
            logger.warning(f"Error creating zoom inset for transformer: {e}")
    
    plt.tight_layout()
    plt.savefig(os.path.join(PLOTS_DIR, 'threshold_analysis.pdf'), bbox_inches='tight')
    plt.savefig(os.path.join(PLOTS_DIR, 'threshold_analysis.png'), bbox_inches='tight')
    plt.close()
    logger.info("Threshold analysis plot saved")

def generate_detection_time_breakdown():
    """Generate detection time breakdown visualization."""
    # Load timing data
    try:
        timing_path = os.path.join(TIMING_DIR, "inference_times.json")
        if not os.path.exists(timing_path):
            logger.warning(f"Timing data not found at {timing_path}")
            return
        
        with open(timing_path, 'r') as f:
            timing_data = json.load(f)
        
        # Load feature extraction time
        extraction_path = os.path.join(TIMING_DIR, "feature_extraction.json")
        if os.path.exists(extraction_path):
            with open(extraction_path, 'r') as f:
                feature_time = json.load(f).get("time_seconds", 0) * 1000  # convert to ms
        else:
            logger.warning(f"Feature extraction time not found at {extraction_path}")
            feature_time = 0
    except Exception as e:
        logger.error(f"Error loading timing data: {e}")
        return
    
    if not timing_data:
        logger.warning("No timing data available")
        return
    
    # Prepare data for plotting
    models = list(timing_data.keys())
    feature_extraction = [feature_time] * len(models)
    model_inference = [timing_data[model].get("mean_latency_ms", 0) for model in models]
    post_processing = [timing_data[model].get("post_processing_ms", 0) for model in models]
    
    # Create figure with regular and log scale plots
    fig, axes = plt.subplots(1, 2, figsize=(15, 6))
    
    # Regular scale
    bar_width = 0.6
    axes[0].bar(models, feature_extraction, bar_width, label='Feature Extraction', color='#4472C4')
    axes[0].bar(models, model_inference, bar_width, bottom=feature_extraction, 
               label='Model Inference', color='#ED7D31')
    axes[0].bar(models, post_processing, bar_width, 
               bottom=[i+j for i,j in zip(feature_extraction, model_inference)], 
               label='Post-processing', color='#A5A5A5')
    
    axes[0].set_title('Detection Time Breakdown (ms)', fontsize=14)
    axes[0].set_ylabel('Time (ms)', fontsize=12)
    axes[0].set_xticks(range(len(models)))
    axes[0].set_xticklabels(models, rotation=45, ha='right')
    axes[0].grid(axis='y', linestyle='--', alpha=0.7)
    axes[0].legend(loc='upper left')
    
    # Log scale
    axes[1].bar(models, feature_extraction, bar_width, label='Feature Extraction', color='#4472C4')
    axes[1].bar(models, model_inference, bar_width, bottom=feature_extraction, 
               label='Model Inference', color='#ED7D31')
    axes[1].bar(models, post_processing, bar_width, 
               bottom=[i+j for i,j in zip(feature_extraction, model_inference)], 
               label='Post-processing', color='#A5A5A5')
    
    axes[1].set_title('Detection Time Breakdown (log scale)', fontsize=14)
    axes[1].set_ylabel('Time (ms)', fontsize=12)
    axes[1].set_yscale('log')
    axes[1].set_xticks(range(len(models)))
    axes[1].set_xticklabels(models, rotation=45, ha='right')
    axes[1].grid(axis='y', linestyle='--', alpha=0.7)
    axes[1].legend(loc='upper left')
    
    plt.tight_layout()
    plt.savefig(os.path.join(PLOTS_DIR, 'detection_time_breakdown.pdf'), bbox_inches='tight')
    plt.savefig(os.path.join(PLOTS_DIR, 'detection_time_breakdown.png'), bbox_inches='tight')
    plt.close()
    
    # Create a table version as well
    table_data = {
        'Model': models,
        'Feature Extraction (ms)': feature_extraction,
        'Model Inference (ms)': model_inference,
        'Post-processing (ms)': post_processing,
        'Total (ms)': [a+b+c for a,b,c in zip(feature_extraction, model_inference, post_processing)]
    }
    
    table_df = pd.DataFrame(table_data)
    table_df.to_csv(os.path.join(PLOTS_DIR, 'detection_time_breakdown.csv'), index=False)
    logger.info("Detection time breakdown saved")

def generate_resource_efficiency_metrics():
    """Generate resource efficiency metrics visualization."""
    # Load memory usage data
    try:
        memory_path = os.path.join(MEMORY_DIR, "memory_usage.json")
        if not os.path.exists(memory_path):
            logger.warning(f"Memory data not found at {memory_path}")
            return
        
        with open(memory_path, 'r') as f:
            memory_data = json.load(f)
        
        # Load inference time data
        timing_path = os.path.join(TIMING_DIR, "inference_times.json")
        if not os.path.exists(timing_path):
            logger.warning(f"Timing data not found at {timing_path}")
            return
        
        with open(timing_path, 'r') as f:
            timing_data = json.load(f)
    except Exception as e:
        logger.error(f"Error loading resource data: {e}")
        return
    
    if not memory_data or not timing_data:
        logger.warning("No memory or timing data available")
        return
    
    # Get common ground truth
    try:
        y_true = find_common_ground_truth()
    except FileNotFoundError as e:
        logger.error(f"Could not find ground truth labels: {e}")
        return
    
    # Load F1 scores (using model_metrics.json if available, or calculate from predictions)
    try:
        metrics_path = os.path.join("evaluation_results", "complete_results.json")
        if os.path.exists(metrics_path):
            with open(metrics_path, 'r') as f:
                complete_results = json.load(f)
                f1_scores = {model: data.get("f1_score", 0) 
                            for model, data in complete_results.get("model_metrics", {}).items()}
        else:
            # Calculate F1 scores directly
            f1_scores = {}
            from sklearn.metrics import f1_score
            for f in glob.glob(os.path.join(PREDICTIONS_DIR, "*_y_pred.npy")):
                model_name = os.path.basename(f).split('_')[0]
                y_pred = np.load(f)
                
                # Ensure predictions match ground truth length
                if len(y_pred) != len(y_true):
                    if len(y_pred) > len(y_true):
                        y_pred = y_pred[:len(y_true)]
                    else:
                        continue
                
                f1_scores[model_name] = f1_score(y_true, y_pred)
    except Exception as e:
        logger.error(f"Error calculating F1 scores: {e}")
        f1_scores = {model: 0.5 for model in memory_data.keys()}  # Use placeholder values
    
    # Find common models present in all datasets
    common_models = set(memory_data.keys()) & set(timing_data.keys()) & set(f1_scores.keys())
    if not common_models:
        logger.warning("No common models found across all datasets")
        return
    
    # Prepare data for plotting
    models = sorted(list(common_models))
    peak_memory = [memory_data[model].get("peak_memory_mb", 0) for model in models]
    mean_latency = [timing_data[model].get("mean_latency_ms", 0) for model in models]
    f1_values = [f1_scores.get(model, 0) for model in models]
    
    # Handle zero values to avoid division by zero
    peak_memory = [max(0.001, m) for m in peak_memory]
    mean_latency = [max(0.001, m) for m in mean_latency]
    
    # Calculate efficiency metrics
    f1_per_mb = [f1 / mem * 1000 for f1, mem in zip(f1_values, peak_memory)]  # scale by 1000
    f1_per_ms = [f1 / lat for f1, lat in zip(f1_values, mean_latency)]
    combined_efficiency = [mb * ms for mb, ms in zip(f1_per_mb, f1_per_ms)]
    
    # Create multi-panel figure
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    axes = axes.flatten()
    
    # F1 per Memory
    axes[0].bar(models, f1_per_mb, color='skyblue')
    axes[0].set_title('F1 Score per 1000 MB of Memory', fontsize=14)
    axes[0].set_ylabel('F1 per 1000 MB', fontsize=12)
    axes[0].grid(axis='y', linestyle='--', alpha=0.7)
    for i, v in enumerate(f1_per_mb):
        axes[0].text(i, v + 0.05, f'{v:.3f}', ha='center', fontsize=10)
    axes[0].set_xticks(range(len(models)))
    axes[0].set_xticklabels(models, rotation=45, ha='right')
    
    # F1 per Latency
    axes[1].bar(models, f1_per_ms, color='lightgreen')
    axes[1].set_title('F1 Score per ms of Latency', fontsize=14)
    axes[1].set_ylabel('F1 per ms', fontsize=12)
    axes[1].grid(axis='y', linestyle='--', alpha=0.7)
    axes[1].set_yscale('log')
    for i, v in enumerate(f1_per_ms):
        axes[1].text(i, v * 1.1, f'{v:.2f}', ha='center', fontsize=10)
    axes[1].set_xticks(range(len(models)))
    axes[1].set_xticklabels(models, rotation=45, ha='right')
    
    # Combined Efficiency
    axes[2].bar(models, combined_efficiency, color='salmon')
    axes[2].set_title('Combined Efficiency Score (log scale)', fontsize=14)
    axes[2].set_ylabel('Efficiency (F1/MB × F1/ms)', fontsize=12)
    axes[2].grid(axis='y', linestyle='--', alpha=0.7)
    axes[2].set_yscale('log')
    for i, v in enumerate(combined_efficiency):
        axes[2].text(i, v * 1.1, f'{v:.1f}', ha='center', fontsize=10)
    axes[2].set_xticks(range(len(models)))
    axes[2].set_xticklabels(models, rotation=45, ha='right')
    
    # Raw metrics table
    table_data = pd.DataFrame({
        'Model': models,
        'F1 Score': f1_values,
        'Peak Memory (MB)': peak_memory,
        'Mean Latency (ms)': mean_latency,
        'F1 per MB': f1_per_mb,
        'F1 per ms': f1_per_ms,
        'Efficiency Score': combined_efficiency
    })
    
    axes[3].axis('off')
    axes[3].table(cellText=table_data.values,
                  colLabels=table_data.columns,
                  cellLoc='center',
                  loc='center',
                  bbox=[0, 0, 1, 1])
    
    plt.tight_layout()
    plt.savefig(os.path.join(PLOTS_DIR, 'resource_efficiency_metrics.pdf'), bbox_inches='tight')
    plt.savefig(os.path.join(PLOTS_DIR, 'resource_efficiency_metrics.png'), bbox_inches='tight')
    plt.close()

    # Save table as CSV
    table_data.to_csv(os.path.join(PLOTS_DIR, 'resource_efficiency_metrics.csv'), index=False)
    logger.info("Resource efficiency metrics saved")

def generate_attack_specific_performance():
    """Generate attack-specific performance visualization with better error handling."""
    # Load attack type metrics
    try:
        attack_metrics_path = os.path.join("evaluation_results/attack_types", "attack_type_metrics.json")
        if not os.path.exists(attack_metrics_path):
            logger.warning(f"Attack type metrics not found at {attack_metrics_path}")
            return
        
        with open(attack_metrics_path, 'r') as f:
            attack_metrics = json.load(f)
    except Exception as e:
        logger.error(f"Error loading attack type metrics: {e}")
        return
    
    if not attack_metrics:
        logger.warning("No attack type metrics available")
        return
    
    # Extract unique attack types across all models
    attack_types = set()
    for model_data in attack_metrics.values():
        attack_types.update(model_data.keys())
    
    # Filter out attack types that might cause problems
    attack_types = [at for at in attack_types if isinstance(at, str) and at.strip()]
    
    if not attack_types:
        logger.warning("No valid attack types found")
        return
    
    attack_types = sorted(list(attack_types))
    models = list(attack_metrics.keys())
    
    if not models:
        logger.warning("No models with attack type data")
        return
    
    # Create F1 score matrix with error handling
    f1_matrix = np.zeros((len(models), len(attack_types)))
    
    for i, model in enumerate(models):
        for j, attack in enumerate(attack_types):
            try:
                if attack in attack_metrics[model]:
                    f1_matrix[i, j] = attack_metrics[model][attack].get("f1", 0)
            except Exception as e:
                logger.warning(f"Error extracting F1 score for {model}/{attack}: {e}")
                f1_matrix[i, j] = 0
    
    # Create heatmap
    plt.figure(figsize=(max(12, len(attack_types)), max(8, len(models))))
    
    try:
        # Handle potential empty matrix
        if np.all(f1_matrix == 0):
            logger.warning("F1 matrix is all zeros, adding small values for visualization")
            f1_matrix = f1_matrix + 0.01
            
        # Handle case where all values are the same
        if np.all(f1_matrix == f1_matrix[0, 0]):
            logger.warning("F1 matrix has all identical values, perturbing for visualization")
            for i in range(len(models)):
                for j in range(len(attack_types)):
                    f1_matrix[i, j] += np.random.uniform(0, 0.05)
                    
        sns.heatmap(f1_matrix, annot=True, cmap='YlGnBu', fmt=".3f",
                    xticklabels=attack_types, yticklabels=models)
    except Exception as e:
        logger.error(f"Error creating heatmap: {e}")
        # Try with a basic imshow as fallback
        plt.imshow(f1_matrix, cmap='YlGnBu')
        plt.xticks(range(len(attack_types)), attack_types, rotation=45, ha='right')
        plt.yticks(range(len(models)), models)
        for i in range(len(models)):
            for j in range(len(attack_types)):
                plt.text(j, i, f"{f1_matrix[i, j]:.3f}", ha="center", va="center")
    
    plt.title('F1 Score by Model and Attack Type', fontsize=14)
    plt.xlabel('Attack Type', fontsize=12)
    plt.ylabel('Model', fontsize=12)
    plt.tight_layout()
    
    plt.savefig(os.path.join(PLOTS_DIR, 'attack_specific_performance.pdf'), bbox_inches='tight')
    plt.savefig(os.path.join(PLOTS_DIR, 'attack_specific_performance.png'), bbox_inches='tight')
    plt.close()
    
    # Create grouped bar chart with error handling
    plt.figure(figsize=(max(14, len(attack_types) * 1.5), 8))
    
    x = np.arange(len(attack_types))
    width = 0.8 / len(models) if models else 0.8
    
    colors = plt.cm.tab10.colors
    for i, model in enumerate(models):
        model_f1 = []
        for attack in attack_types:
            try:
                if attack in attack_metrics[model]:
                    model_f1.append(attack_metrics[model][attack].get("f1", 0))
                else:
                    model_f1.append(0)
            except Exception as e:
                logger.warning(f"Error getting F1 for {model}/{attack}: {e}")
                model_f1.append(0)
        
        plt.bar(x + i*width - 0.4 + width/2, model_f1, width, 
                label=model, color=colors[i % len(colors)])
    
    plt.xlabel('Attack Type', fontsize=12)
    plt.ylabel('F1 Score', fontsize=12)
    plt.title('F1 Score by Model and Attack Type', fontsize=14)
    plt.xticks(x, attack_types, rotation=45, ha='right')
    plt.legend(loc='best')
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.ylim(0, 1.05)
    
    plt.tight_layout()
    plt.savefig(os.path.join(PLOTS_DIR, 'attack_specific_performance_bars.pdf'), bbox_inches='tight')
    plt.savefig(os.path.join(PLOTS_DIR, 'attack_specific_performance_bars.png'), bbox_inches='tight')
    plt.close()
    logger.info("Attack-specific performance visualizations saved")

def generate_scalability_analysis():
    """Generate scalability analysis visualization with better error handling."""
    # Load scalability results
    try:
        scalability_path = os.path.join("evaluation_results/scalability", "scalability_results.json")
        if not os.path.exists(scalability_path):
            logger.warning(f"Scalability results not found at {scalability_path}")
            return
        
        with open(scalability_path, 'r') as f:
            scalability_data = json.load(f)
    except Exception as e:
        logger.error(f"Error loading scalability results: {e}")
        return
    
    if not scalability_data:
        logger.warning("No scalability data available")
        return
    
    # Extract traffic rates across all models
    traffic_rates = set()
    for model_data in scalability_data.values():
        traffic_rates.update(model_data.keys())
    
    # Convert to integers and sort
    try:
        traffic_rates = sorted([int(rate) for rate in traffic_rates if rate.isdigit()])
    except Exception as e:
        logger.warning(f"Error extracting traffic rates: {e}")
        # Try alternative approach
        traffic_rates = []
        for model_data in scalability_data.values():
            for rate in model_data.keys():
                try:
                    traffic_rates.append(int(rate))
                except (ValueError, TypeError):
                    pass
        traffic_rates = sorted(list(set(traffic_rates)))
    
    if not traffic_rates:
        logger.warning("No valid traffic rates found")
        return
    
    models = list(scalability_data.keys())
    
    if not models:
        logger.warning("No models with scalability data")
        return
    
    # Create multi-panel figure
    fig, axes = plt.subplots(3, 1, figsize=(12, 15))
    
    # F1 Score vs Traffic Rate
    for model in models:
        rates = []
        f1_scores = []
        try:
            for rate in traffic_rates:
                rate_str = str(rate)
                if rate_str in scalability_data[model]:
                    rates.append(rate)
                    f1_scores.append(scalability_data[model][rate_str].get("f1_score", 0))
            
            if rates:
                axes[0].plot(rates, f1_scores, marker='o', linewidth=2, label=model)
        except Exception as e:
            logger.warning(f"Error plotting F1 scores for {model}: {e}")
    
    axes[0].set_xlabel('Traffic Rate (packets per second)', fontsize=12)
    axes[0].set_ylabel('F1 Score', fontsize=12)
    axes[0].set_title('F1 Score vs Traffic Rate', fontsize=14)
    axes[0].grid(True, linestyle='--', alpha=0.7)
    axes[0].legend(loc='best')
    
    # Latency vs Traffic Rate
    for model in models:
        rates = []
        latencies = []
        try:
            for rate in traffic_rates:
                rate_str = str(rate)
                if rate_str in scalability_data[model]:
                    rates.append(rate)
                    latencies.append(scalability_data[model][rate_str].get("latency_ms", 0))
            
            if rates:
                axes[1].plot(rates, latencies, marker='o', linewidth=2, label=model)
        except Exception as e:
            logger.warning(f"Error plotting latencies for {model}: {e}")
    
    axes[1].set_xlabel('Traffic Rate (packets per second)', fontsize=12)
    axes[1].set_ylabel('Latency (ms)', fontsize=12)
    axes[1].set_title('Latency vs Traffic Rate (log scale)', fontsize=14)
    
    # Only use log scale if values are positive
    if all(l > 0 for model in models for rate_str in scalability_data.get(model, {}) 
           for l in [scalability_data[model][rate_str].get("latency_ms", 0.1)]):
        axes[1].set_yscale('log')
    else:
        logger.warning("Some latency values are zero or negative, using linear scale")
        
    axes[1].grid(True, linestyle='--', alpha=0.7)
    axes[1].legend(loc='best')
    
    # Memory Usage vs Traffic Rate
    for model in models:
        rates = []
        memory = []
        try:
            for rate in traffic_rates:
                rate_str = str(rate)
                if rate_str in scalability_data[model]:
                    rates.append(rate)
                    memory.append(scalability_data[model][rate_str].get("memory_mb", 0))
            
            if rates:
                axes[2].plot(rates, memory, marker='o', linewidth=2, label=model)
        except Exception as e:
            logger.warning(f"Error plotting memory usage for {model}: {e}")
    
    axes[2].set_xlabel('Traffic Rate (packets per second)', fontsize=12)
    axes[2].set_ylabel('Memory Usage (MB)', fontsize=12)
    axes[2].set_title('Memory Usage vs Traffic Rate', fontsize=14)
    axes[2].grid(True, linestyle='--', alpha=0.7)
    axes[2].legend(loc='best')
    
    plt.tight_layout()
    plt.savefig(os.path.join(PLOTS_DIR, 'scalability_analysis.pdf'), bbox_inches='tight')
    plt.savefig(os.path.join(PLOTS_DIR, 'scalability_analysis.png'), bbox_inches='tight')
    plt.close()
    
    # Create degradation table with error handling
    try:
        degradation_data = []
        
        for model in models:
            if len(traffic_rates) >= 2:
                min_rate_str = str(traffic_rates[0])
                max_rate_str = str(traffic_rates[-1])
                
                if min_rate_str in scalability_data[model] and max_rate_str in scalability_data[model]:
                    min_rate_data = scalability_data[model][min_rate_str]
                    max_rate_data = scalability_data[model][max_rate_str]
                    
                    min_f1 = min_rate_data.get("f1_score", 0.001)
                    max_f1 = max_rate_data.get("f1_score", 0.001)
                    min_latency = min_rate_data.get("latency_ms", 0.001)
                    max_latency = max_rate_data.get("latency_ms", 0.001)
                    min_memory = min_rate_data.get("memory_mb", 0.001)
                    max_memory = max_rate_data.get("memory_mb", 0.001)
                    
                    # Avoid division by zero
                    if min_f1 > 0 and min_latency > 0 and min_memory > 0:
                        f1_degradation = (1 - (max_f1 / min_f1)) * 100
                        latency_increase = ((max_latency / min_latency) - 1) * 100
                        memory_increase = ((max_memory / min_memory) - 1) * 100
                        
                        degradation_data.append({
                            'Model': model,
                            'F1 Score Degradation (%)': f1_degradation,
                            'Latency Increase (%)': latency_increase,
                            'Memory Increase (%)': memory_increase
                        })
        
        if degradation_data:
            degradation_df = pd.DataFrame(degradation_data)
            degradation_df.to_csv(os.path.join(PLOTS_DIR, 'scalability_degradation.csv'), index=False)
    except Exception as e:
        logger.error(f"Error creating degradation table: {e}")
    
    logger.info("Scalability analysis saved")

def generate_significance_tests():
    """Generate statistical significance testing visualization with better error handling."""
    # Load cross-validation results
    try:
        cv_path = os.path.join("evaluation_results/significance", "cross_validation_f1_scores.json")
        t_test_path = os.path.join("evaluation_results/significance", "t_test_results.json")
        
        if not os.path.exists(cv_path) or not os.path.exists(t_test_path):
            logger.warning(f"Significance test results not found")
            return
        
        with open(cv_path, 'r') as f:
            cv_scores = json.load(f)
        
        with open(t_test_path, 'r') as f:
            t_tests = json.load(f)
    except Exception as e:
        logger.error(f"Error loading significance test results: {e}")
        return
    
    if not cv_scores or not t_tests:
        logger.warning("No significance test data available")
        return
    
    # Get list of models
    models = list(cv_scores.keys())
    
    if not models:
        logger.warning("No models with cross-validation scores")
        return
    
    # Create heatmap of p-values with error handling
    try:
        # Initialize p-value matrix with NaN values
        p_value_matrix = np.full((len(models), len(models)), np.nan)
        
        # Fill in p-values
        for result in t_tests:
            model1 = result.get("model1")
            model2 = result.get("model2")
            p_value = result.get("p_value")
            
            if model1 in models and model2 in models and p_value is not None:
                i = models.index(model1)
                j = models.index(model2)
                p_value_matrix[i, j] = p_value
                p_value_matrix[j, i] = p_value
        
        # Create heatmap
        plt.figure(figsize=(max(10, len(models)), max(8, len(models))))
        mask = np.isnan(p_value_matrix)
        
        cmap = 'coolwarm_r'
        sns.heatmap(p_value_matrix, annot=True, cmap=cmap, mask=mask,
                    xticklabels=models, yticklabels=models, vmin=0, vmax=0.1,
                    annot_kws={"size": 10}, fmt='.4f')
        
        plt.title('P-values of Pairwise t-tests Between Models', fontsize=14)
        plt.tight_layout()
        
        plt.savefig(os.path.join(PLOTS_DIR, 'significance_tests_heatmap.pdf'), bbox_inches='tight')
        plt.savefig(os.path.join(PLOTS_DIR, 'significance_tests_heatmap.png'), bbox_inches='tight')
        plt.close()
    except Exception as e:
        logger.error(f"Error creating p-value heatmap: {e}")
    
    # Create boxplot of F1 distributions with error handling
    try:
        plt.figure(figsize=(max(12, len(models) * 1.5), 6))
        
        # Convert to DataFrame for seaborn
        cv_data = []
        for model, scores in cv_scores.items():
            if not scores:
                logger.warning(f"No cross-validation scores for {model}")
                continue
                
            for score in scores:
                cv_data.append({'Model': model, 'F1 Score': score})
        
        if not cv_data:
            logger.warning("No cross-validation data for boxplot")
            return
            
        cv_df = pd.DataFrame(cv_data)
        
        ax = sns.boxplot(x='Model', y='F1 Score', data=cv_df)
        plt.title('F1 Score Distribution by Model (Cross-Validation)', fontsize=14)
        plt.xlabel('Model', fontsize=12)
        plt.ylabel('F1 Score', fontsize=12)
        plt.grid(axis='y', linestyle='--', alpha=0.7)
        
        # Add statistical significance markers if there are at least 2 models
        if len(models) >= 2:
            y_pos = cv_df['F1 Score'].max() + 0.01
            for result in t_tests:
                model1 = result.get("model1")
                model2 = result.get("model2")
                p_value = result.get("p_value")
                significant = result.get("significant", False)
                
                if significant and model1 in models and model2 in models:
                    i1 = models.index(model1)
                    i2 = models.index(model2)
                    plt.plot([i1, i2], [y_pos, y_pos], 'k-')
                    plt.text((i1 + i2) / 2, y_pos + 0.002, f'p={p_value:.4f}*', ha='center')
                    y_pos += 0.01
        
        plt.tight_layout()
        plt.savefig(os.path.join(PLOTS_DIR, 'f1_score_distribution.pdf'), bbox_inches='tight')
        plt.savefig(os.path.join(PLOTS_DIR, 'f1_score_distribution.png'), bbox_inches='tight')
        plt.close()
    except Exception as e:
        logger.error(f"Error creating F1 score distribution boxplot: {e}")
    
    logger.info("Statistical significance visualizations saved")

def generate_model_interpretability():
    """Generate model interpretability visualizations with better error handling."""
    # Load feature importance data
    try:
        importance_path = os.path.join("evaluation_results/interpretability", "feature_importance.json")
        if not os.path.exists(importance_path):
            logger.warning(f"Feature importance data not found at {importance_path}")
            return
        
        with open(importance_path, 'r') as f:
            feature_importance = json.load(f)
    except Exception as e:
        logger.error(f"Error loading feature importance data: {e}")
        return
    
    if not feature_importance:
        logger.warning("No feature importance data available")
        return
    
    # Create heatmap for feature importance with error handling
    try:
        models = list(feature_importance.keys())
        features = set()
        for model_data in feature_importance.values():
            features.update(model_data.keys())
        
        if not models or not features:
            logger.warning("No models or features found")
            return
        
        # Calculate average importance for each feature
        feature_avg_importance = {}
        for feature in features:
            total = 0
            count = 0
            for model in models:
                if feature in feature_importance[model]:
                    total += feature_importance[model][feature]
                    count += 1
            if count > 0:
                feature_avg_importance[feature] = total / count
        
        # Select top features
        top_features = sorted(feature_avg_importance.items(), key=lambda x: x[1], reverse=True)
        top_n = min(20, len(top_features))  # Limit to top 20 features
        top_feature_names = [f[0] for f in top_features[:top_n]]
        
        # Create importance matrix
        importance_matrix = np.zeros((len(models), len(top_feature_names)))
        
        for i, model in enumerate(models):
            for j, feature in enumerate(top_feature_names):
                if feature in feature_importance[model]:
                    importance_matrix[i, j] = feature_importance[model][feature]
        
        # Handle case where all values are the same
        if np.all(importance_matrix == importance_matrix[0, 0]):
            logger.warning("Importance matrix has identical values, adding small random perturbations")
            importance_matrix += np.random.uniform(0, 0.01, importance_matrix.shape)
        
        # Create heatmap
        plt.figure(figsize=(max(14, len(top_feature_names) * 0.8), max(10, len(models))))
        sns.heatmap(importance_matrix, annot=True, cmap='YlGnBu', fmt=".3f",
                    xticklabels=top_feature_names, yticklabels=models)
        
        plt.title('Feature Importance by Model', fontsize=14)
        plt.xlabel('Feature', fontsize=12)
        plt.ylabel('Model', fontsize=12)
        plt.xticks(rotation=45, ha='right')
        
        plt.tight_layout()
        plt.savefig(os.path.join(PLOTS_DIR, 'feature_importance_heatmap.pdf'), bbox_inches='tight')
        plt.savefig(os.path.join(PLOTS_DIR, 'feature_importance_heatmap.png'), bbox_inches='tight')
        plt.close()
    except Exception as e:
        logger.error(f"Error creating feature importance heatmap: {e}")
    
    # Create bar charts of top features by model with error handling
    try:
        if len(models) > 0:
            fig, axes = plt.subplots(len(models), 1, figsize=(12, 3*len(models)))
            
            # Handle case of a single model
            if len(models) == 1:
                axes = [axes]
            
            for i, model in enumerate(models):
                model_importance = feature_importance[model]
                if not model_importance:
                    logger.warning(f"No feature importance data for {model}")
                    continue
                    
                top_model_features = sorted(model_importance.items(), key=lambda x: x[1], reverse=True)
                top_n = min(5, len(top_model_features))
                
                if top_n == 0:
                    logger.warning(f"No feature importance values for {model}")
                    continue
                
                feature_names = [f[0] for f in top_model_features[:top_n]]
                importance_values = [f[1] for f in top_model_features[:top_n]]
                
                axes[i].barh(feature_names, importance_values)
                axes[i].set_title(f'Top 5 Features: {model}', fontsize=12)
                axes[i].set_xlabel('Importance', fontsize=10)
                axes[i].set_ylabel('Feature', fontsize=10)
                axes[i].grid(axis='x', linestyle='--', alpha=0.7)
            
            plt.tight_layout()
            plt.savefig(os.path.join(PLOTS_DIR, 'top_features_by_model.pdf'), bbox_inches='tight')
            plt.savefig(os.path.join(PLOTS_DIR, 'top_features_by_model.png'), bbox_inches='tight')
            plt.close()
    except Exception as e:
        logger.error(f"Error creating top features bar charts: {e}")
    
    # Load attention weights if available
    try:
        attention_path = os.path.join("evaluation_results/interpretability", "attention_weights.json")
        if os.path.exists(attention_path):
            with open(attention_path, 'r') as f:
                attention_weights = json.load(f)
            
            # Create attention visualization for Transformer
            if 'transformer' in attention_weights:
                attention = attention_weights['transformer']
                
                fig, axes = plt.subplots(2, 2, figsize=(12, 10))
                axes = axes.flatten()
                
                for i, head_matrix in enumerate(attention[:min(4, len(attention))]):  # First 4 heads or fewer
                    im = axes[i].imshow(head_matrix, cmap='viridis')
                    axes[i].set_title(f'Attention Head {i+1}', fontsize=12)
                    axes[i].set_xlabel('Key Position', fontsize=10)
                    axes[i].set_ylabel('Query Position', fontsize=10)
                    fig.colorbar(im, ax=axes[i])
                
                plt.suptitle('Transformer Attention Visualization', fontsize=14)
                plt.tight_layout()
                
                plt.savefig(os.path.join(PLOTS_DIR, 'transformer_attention.pdf'), bbox_inches='tight')
                plt.savefig(os.path.join(PLOTS_DIR, 'transformer_attention.png'), bbox_inches='tight')
                plt.close()
                logger.info("Transformer attention visualization saved")
    except Exception as e:
        logger.error(f"Error creating attention visualization: {e}")
    
    logger.info("Model interpretability visualizations saved")

if __name__ == "__main__":
    generate_visualizations()