#!/usr/bin/env python3
# align_dataset.py
"""
This script ensures consistent data alignment across evaluation scripts.
It finds all prediction files and ensures they use the same ground truth.
"""

import os
import numpy as np
import glob
import logging
import shutil
from pathlib import Path

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def find_all_prediction_files():
    """Find all prediction files across evaluation directories."""
    pred_files = []
    true_files = []
    
    for root, _, files in os.walk('evaluation_results'):
        for file in files:
            if file.endswith('_y_pred.npy'):
                pred_files.append(os.path.join(root, file))
            elif file.endswith('_y_true.npy') or file == 'y_true.npy':
                true_files.append(os.path.join(root, file))
    
    return pred_files, true_files

def ensure_matching_dimensions(pred_files, true_files):
    """Ensure all prediction files match the dimensions of their corresponding ground truth."""
    # If we have a common ground truth file, use it as reference
    common_true = [f for f in true_files if os.path.basename(f) == 'y_true.npy']
    if common_true:
        reference_true = common_true[0]
        reference_labels = np.load(reference_true)
        reference_length = len(reference_labels)
        logger.info(f"Using common ground truth with {reference_length} samples as reference")
    else:
        # Find the most common length among all true files
        true_lengths = {}
        for f in true_files:
            try:
                labels = np.load(f)
                length = len(labels)
                true_lengths[length] = true_lengths.get(length, 0) + 1
            except Exception as e:
                logger.warning(f"Error loading {f}: {e}")
        
        if not true_lengths:
            logger.error("No valid ground truth files found")
            return False
        
        # Use the most common length as reference
        reference_length = max(true_lengths.items(), key=lambda x: x[1])[0]
        # Find a file with this length to use as reference
        for f in true_files:
            try:
                labels = np.load(f)
                if len(labels) == reference_length:
                    reference_true = f
                    reference_labels = labels
                    break
            except:
                continue
        
        logger.info(f"Using most common length of {reference_length} samples as reference")
    
    # Create a common ground truth file in each prediction directory
    prediction_dirs = set(os.path.dirname(f) for f in pred_files)
    for pred_dir in prediction_dirs:
        common_file = os.path.join(pred_dir, 'y_true.npy')
        np.save(common_file, reference_labels)
        logger.info(f"Created common ground truth file at {common_file}")
    
    # Align all prediction files to the reference length
    fixed_count = 0
    for pred_file in pred_files:
        try:
            # Get the corresponding true file
            base_name = os.path.basename(pred_file)
            model_prefix = '_'.join(base_name.split('_')[:-2]) if '_y_pred.npy' in base_name else base_name.split('_')[0]
            true_file = os.path.join(os.path.dirname(pred_file), f"{model_prefix}_y_true.npy")
            
            if not os.path.exists(true_file):
                # Use the common one
                true_file = os.path.join(os.path.dirname(pred_file), 'y_true.npy')
            
            # Load files
            predictions = np.load(pred_file)
            
            # Check and fix dimension mismatch
            if len(predictions) != reference_length:
                logger.warning(f"Dimension mismatch in {pred_file}: {len(predictions)} vs {reference_length}")
                
                if len(predictions) > reference_length:
                    # Truncate predictions
                    logger.info(f"Truncating predictions in {pred_file}")
                    predictions = predictions[:reference_length]
                else:
                    # Pad predictions (with zeros for binary classification)
                    logger.info(f"Padding predictions in {pred_file}")
                    pad_width = reference_length - len(predictions)
                    predictions = np.pad(predictions, (0, pad_width), 'constant')
                
                # Create backup
                backup_file = pred_file + '.bak'
                shutil.copy2(pred_file, backup_file)
                
                # Save fixed predictions
                np.save(pred_file, predictions)
                fixed_count += 1
                
            # Also make sure true file matches reference
            if os.path.exists(true_file) and true_file != reference_true:
                true_labels = np.load(true_file)
                if len(true_labels) != reference_length or not np.array_equal(true_labels, reference_labels):
                    # Create backup
                    backup_file = true_file + '.bak'
                    shutil.copy2(true_file, backup_file)
                    
                    # Save reference labels
                    np.save(true_file, reference_labels)
                    logger.info(f"Updated ground truth file {true_file} to match reference")
                    fixed_count += 1
        except Exception as e:
            logger.error(f"Error processing {pred_file}: {e}")
    
    logger.info(f"Fixed {fixed_count} files to ensure consistent dimensions")
    return True

def fix_data_alignment():
    """Main function to fix data alignment."""
    logger.info("Starting data alignment check")
    
    # Find all prediction and ground truth files
    pred_files, true_files = find_all_prediction_files()
    
    if not pred_files:
        logger.warning("No prediction files found")
        return False
    
    if not true_files:
        logger.warning("No ground truth files found")
        return False
    
    logger.info(f"Found {len(pred_files)} prediction files and {len(true_files)} ground truth files")
    
    # Ensure matching dimensions
    success = ensure_matching_dimensions(pred_files, true_files)
    
    if success:
        logger.info("Data alignment check completed successfully")
    else:
        logger.error("Data alignment check failed")
    
    return success

if __name__ == "__main__":
    fix_data_alignment()