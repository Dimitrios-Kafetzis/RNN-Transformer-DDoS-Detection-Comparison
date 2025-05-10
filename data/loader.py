#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
File: data/loader.py
Author: Dimitrios Kafetzis (kafetzis@aueb.gr)
Created: May 2025
License: MIT

Description:
    Data loading functionality for network intrusion detection datasets.
    Supports loading and preprocessing NSL-KDD and Bot-IoT datasets.
    Provides functions for:
    - Downloading datasets
    - Processing raw data
    - Creating aligned feature spaces
    - Normalizing features
    - Creating TensorFlow datasets for model training

Usage:
    This module is imported by other scripts and not meant to be run directly.
    
    Import examples:
    from data.loader import load_nslkdd_dataset, process_nslkdd_dataset
    from data.loader import normalize_features, create_aligned_feature_space
    
    Usage examples:
    train_df, test_df = load_nslkdd_dataset(data_dir, split="both")
    processed_df = process_nslkdd_dataset(raw_df)
    X_norm, mean, std = normalize_features(X)
"""

import os
import pandas as pd
import numpy as np
import tensorflow as tf
import logging
import requests
from typing import Tuple, List, Dict, Optional, Union

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# URLs for NSL-KDD dataset
NSL_KDD_TRAIN_URL = "https://iscxdownloads.cs.unb.ca/iscxdownloads/NSL-KDD/NSL-KDD-Train.txt"
NSL_KDD_TEST_URL  = "https://iscxdownloads.cs.unb.ca/iscxdownloads/NSL-KDD/NSL-KDD-Test.txt"
NSL_KDD_TRAIN_ALT_URL  = "https://raw.githubusercontent.com/defcom17/NSL_KDD/master/KDDTrain%2B.txt"
NSL_KDD_TEST_ALT_URL   = "https://raw.githubusercontent.com/defcom17/NSL_KDD/master/KDDTest%2B.txt"
NSL_KDD_TRAIN_ALT2_URL = "https://raw.githubusercontent.com/jmnwong/NSL-KDD-Dataset/master/KDDTrain%2B.txt"
NSL_KDD_TEST_ALT2_URL  = "https://raw.githubusercontent.com/jmnwong/NSL-KDD-Dataset/master/KDDTest%2B.txt"

NSL_KDD_FEATURES = [
    'duration', 'protocol_type', 'service', 'flag', 'src_bytes', 'dst_bytes',
    'land', 'wrong_fragment', 'urgent', 'hot', 'num_failed_logins', 'logged_in',
    'num_compromised', 'root_shell', 'su_attempted', 'num_root',
    'num_file_creations', 'num_shells', 'num_access_files', 'num_outbound_cmds',
    'is_host_login', 'is_guest_login', 'count', 'srv_count', 'serror_rate',
    'srv_serror_rate', 'rerror_rate', 'srv_rerror_rate', 'same_srv_rate',
    'diff_srv_rate', 'srv_diff_host_rate', 'dst_host_count', 'dst_host_srv_count',
    'dst_host_same_srv_rate', 'dst_host_diff_srv_rate',
    'dst_host_same_src_port_rate', 'dst_host_srv_diff_host_rate',
    'dst_host_serror_rate', 'dst_host_srv_serror_rate', 'dst_host_rerror_rate',
    'dst_host_srv_rerror_rate', 'label', 'difficulty'
]

NSL_KDD_ATTACK_TYPES: Dict[str, str] = {
    'normal': 'normal',
    'back': 'dos', 'land': 'dos', 'neptune': 'dos', 'pod': 'dos',
    'smurf': 'dos', 'teardrop': 'dos', 'mailbomb': 'dos',
    'apache2': 'dos', 'processtable': 'dos', 'udpstorm': 'dos',
    'ipsweep': 'probe', 'nmap': 'probe', 'portsweep': 'probe',
    'satan': 'probe', 'mscan': 'probe', 'saint': 'probe',
    'ftp_write': 'r2l', 'guess_passwd': 'r2l', 'imap': 'r2l',
    # ... (other mappings as before)
}


def _get_nslkdd_colnames() -> List[str]:
    """
    Fetch the official NSL-KDD feature names (including label & difficulty).
    """
    url = "https://raw.githubusercontent.com/defcom17/NSL_KDD/master/Field%20Names.txt"
    resp = requests.get(url, timeout=10)
    resp.raise_for_status()
    # each line is one feature name
    return [line.strip() for line in resp.text.splitlines() if line.strip()]


def download_nslkdd_dataset(data_dir: str) -> Tuple[str, str]:
    """
    Download NSL-KDD if needed, prefixing a correct header row.
    Returns (train_csv_path, test_csv_path).
    """
    os.makedirs(data_dir, exist_ok=True)
    train_file = os.path.join(data_dir, "NSL-KDD-Train.csv")
    test_file  = os.path.join(data_dir, "NSL-KDD-Test.csv")

    # build header from FEATURES list
    header = ",".join(NSL_KDD_FEATURES)

    # helper to download-or-copy for a single split
    def _ensure(file_path: str, local_names: List[str], urls: List[str]) -> None:
        if os.path.exists(file_path):
            return
        # try locals
        for ln in local_names:
            if os.path.exists(ln):
                logger.info(f"Using local file {ln}")
                with open(ln) as src, open(file_path, "w") as dst:
                    dst.write(header + "\n" + src.read())
                return
        # try remotes
        for url in urls:
            try:
                logger.info(f"Downloading from {url}")
                r = requests.get(url, timeout=30)
                r.raise_for_status()
                with open(file_path, "w") as f:
                    f.write(header + "\n" + r.text)
                return
            except Exception as e:
                logger.warning(f"Failed {url}: {e}")
        raise RuntimeError(f"Could not fetch NSL-KDD split for {file_path}")

    _ensure(
        train_file,
        [os.path.join(data_dir, n) for n in ("KDDTrain+.txt","KDDTrain.txt","NSL-KDD-Train.txt")],
        [NSL_KDD_TRAIN_URL, NSL_KDD_TRAIN_ALT_URL, NSL_KDD_TRAIN_ALT2_URL]
    )
    _ensure(
        test_file,
        [os.path.join(data_dir, n) for n in ("KDDTest+.txt","KDDTest.txt","NSL-KDD-Test.txt")],
        [NSL_KDD_TEST_URL, NSL_KDD_TEST_ALT_URL, NSL_KDD_TEST_ALT2_URL]
    )

    return train_file, test_file


def process_nslkdd_dataset(df: pd.DataFrame) -> pd.DataFrame:
    """
    Clean NSL-KDD DataFrame:
      - drop 'difficulty'
      - lower-case/strip 'label'
      - map to 'attack_type', 'binary_label', 'is_dos'
    """
    if 'difficulty' in df.columns:
        df = df.drop('difficulty', axis=1)

    if 'label' in df.columns:
        df['label'] = df['label'].str.strip().str.lower()
        df['attack_type'] = df['label'].map(NSL_KDD_ATTACK_TYPES).fillna('unknown')
        df['binary_label'] = (df['label'] != 'normal').astype(int)
        df['is_dos'] = (df['attack_type'] == 'dos').astype(int)

    return df


def load_nslkdd_dataset(
    data_dir: str,
    split: str = 'both'
) -> Union[pd.DataFrame, Tuple[pd.DataFrame, pd.DataFrame]]:
    """
    Returns train_df, test_df (if split='both'), else one of them.
    """
    logger.info(f"Loading NSL-KDD ({split}) from {data_dir}")
    train_csv, test_csv = download_nslkdd_dataset(data_dir)

    out_train: Optional[pd.DataFrame] = None
    out_test:  Optional[pd.DataFrame] = None

    if split.lower() in ('train','both'):
        df = pd.read_csv(train_csv)
        out_train = process_nslkdd_dataset(df)
        logger.info(f"  train shape: {out_train.shape}")

    if split.lower() in ('test','both'):
        df = pd.read_csv(test_csv)
        out_test = process_nslkdd_dataset(df)
        logger.info(f"  test  shape: {out_test.shape}")

    if split.lower() == 'both':
        return out_train, out_test  # type: ignore
    return out_train or out_test  # type: ignore


def prepare_combined_dataset(
    data_dir: str,
    hard_csv: str
) -> Tuple[
    Tuple[np.ndarray, np.ndarray],
    Tuple[np.ndarray, np.ndarray],
    Tuple[np.ndarray, np.ndarray],
    List[str],
    Tuple[np.ndarray, np.ndarray]  # Added mean and std
]:
    """
    Load train+test, plus a 'hard' CSV, one-hot-encode all together,
    split back into:
      - (X_train, y_train)
      - (X_easy,  y_easy)
      - (X_hard,  y_hard)
    and return feature column list plus normalization parameters.
    """
    # 1) load
    train_df, test_df = load_nslkdd_dataset(data_dir, split="both")
    
    # 2) load hard set
    try:
        # First try assuming it's in raw format
        hard_df = pd.read_csv(hard_csv, header=None, names=train_df.columns)
        is_raw = True
    except:
        # If that fails, assume it's already processed
        hard_df = pd.read_csv(hard_csv)
        is_raw = False
    
    # 3) Process all datasets with consistent feature space
    if is_raw:
        # If hard set is raw, process it properly
        train_enc, test_enc, feat_cols = create_aligned_feature_space(train_df, test_df)
        proc_hard = process_nslkdd_dataset(hard_df)
        hard_enc, _, _ = create_aligned_feature_space(train_df, proc_hard)
    else:
        # If hard set is pre-processed, make sure it aligns with training
        train_enc, test_enc, feat_cols = create_aligned_feature_space(train_df, test_df)
        
        # Need to ensure hard set has the same columns
        for col in feat_cols:
            if col not in hard_df.columns:
                hard_df[col] = 0
                
        # Create properly aligned hard set
        hard_features = [col for col in feat_cols if col in hard_df.columns]
        missing_features = [col for col in feat_cols if col not in hard_df.columns]
        
        hard_enc = pd.DataFrame()
        for col in feat_cols:
            if col in hard_df.columns:
                hard_enc[col] = hard_df[col]
            else:
                hard_enc[col] = 0
                
        # Add labels
        if "binary_label" in hard_df.columns:
            hard_enc["binary_label"] = hard_df["binary_label"]
        elif "label" in hard_df.columns:
            hard_enc["binary_label"] = (hard_df["label"] != "normal").astype(int)
        else:
            # Default to assuming all are attacks if no label
            hard_enc["binary_label"] = 1
            
    # Add binary labels if not present
    train_enc["binary_label"] = (train_enc["label"] != "normal").astype(int)
    test_enc["binary_label"] = (test_enc["label"] != "normal").astype(int)
    
    # 5) to NumPy
    X_train = train_enc[feat_cols].astype(np.float32).to_numpy()
    y_train = train_enc["binary_label"].to_numpy()
    X_easy  = test_enc[feat_cols].astype(np.float32).to_numpy()
    y_easy  = test_enc["binary_label"].to_numpy()
    X_hard  = hard_enc[feat_cols].astype(np.float32).to_numpy()
    y_hard  = hard_enc["binary_label"].to_numpy()
    
    # 6) normalize features
    X_train, mean, std = normalize_features(X_train)
    X_easy, _, _ = normalize_features(X_easy, mean, std)
    X_hard, _, _ = normalize_features(X_hard, mean, std)

    logger.info(f"Combined dataset: train={X_train.shape}, easy={X_easy.shape}, hard={X_hard.shape}")
    return (X_train, y_train), (X_easy, y_easy), (X_hard, y_hard), feat_cols, (mean, std)

def normalize_features(features, mean=None, std=None):
    """
    Normalize features with robust handling of outliers and NaN values.
    
    Args:
        features: NumPy array of features
        mean: Optional pre-computed mean values
        std: Optional pre-computed standard deviation values
        
    Returns:
        Normalized features, mean, std
    """
    # Replace NaN values with zeros
    features = np.nan_to_num(features, nan=0.0)
    
    if mean is None or std is None:
        # More robust mean calculation using nanmean
        mean = np.nanmean(features, axis=0)
        
        # More robust std calculation with handling of zeros
        std = np.nanstd(features, axis=0)
        std[std < 1e-8] = 1.0  # Increased threshold for numerical stability
    
    # Clip extreme values to 5 standard deviations
    features_clipped = np.clip(
        features, 
        mean - 5 * std, 
        mean + 5 * std
    )
    
    # Normalize
    normalized = (features_clipped - mean) / std
    
    # Final check for any remaining NaN or inf values
    normalized = np.nan_to_num(normalized, nan=0.0, posinf=0.0, neginf=0.0)
    
    return normalized, mean, std

def create_sequence_dataset(
    features: np.ndarray,
    labels: np.ndarray,
    window_size: int,
    step_size: int,
    batch_size: int,
    shuffle: bool = True
) -> tf.data.Dataset:
    """
    Create a TensorFlow Dataset for sequence models from flat features.
    
    Args:
        features: Feature matrix [n_samples, n_features]
        labels: Labels [n_samples]
        window_size: Sliding window size
        step_size: Stride for sliding window
        batch_size: Batch size for TF Dataset
        shuffle: Whether to shuffle the data
    
    Returns:
        tf.data.Dataset with sequential data
    """
    n_samples = features.shape[0]
    n_features = features.shape[1]
    
    # For very small datasets, use a smaller window to ensure we have samples
    if n_samples < window_size:
        window_size = max(1, n_samples // 2)
        step_size = 1
    
    # Create sliding windows
    windows = []
    window_labels = []
    
    for i in range(0, n_samples - window_size + 1, step_size):
        windows.append(features[i:i+window_size])
        # Use the label of the last element in the window
        window_labels.append(labels[i+window_size-1])
    
    # Handle edge case if we don't have enough data for windows
    if len(windows) == 0:
        # Just create a single window with the available data, padded if needed
        if n_samples < window_size:
            pad = np.zeros((window_size - n_samples, n_features))
            padded_features = np.vstack([features, pad])
            windows.append(padded_features)
            window_labels.append(labels[-1])
        else:
            windows.append(features[:window_size])
            window_labels.append(labels[window_size-1])
    
    # Convert to arrays
    X_seq = np.array(windows, dtype=np.float32)
    y_seq = np.array(window_labels, dtype=np.int32)
    
    # Create dataset
    dataset = tf.data.Dataset.from_tensor_slices((X_seq, y_seq))
    if shuffle:
        dataset = dataset.shuffle(buffer_size=len(X_seq))
    
    return dataset.batch(batch_size).prefetch(tf.data.AUTOTUNE)

def create_aligned_feature_space(train_df: pd.DataFrame, test_df: pd.DataFrame = None) -> tuple:
    """
    Create a consistent feature space for training and test datasets.
    
    Args:
        train_df: Training DataFrame (used to define the feature space)
        test_df: Optional test DataFrame to align to the training feature space
        
    Returns:
        tuple containing:
        - processed training DataFrame with one-hot encoding
        - processed test DataFrame aligned to training features (if provided)
        - list of feature column names
    """
    # Process training data
    proc_train = process_nslkdd_dataset(train_df.copy())
    train_enc = pd.get_dummies(proc_train, columns=["protocol_type", "service", "flag"])
    
    # Extract feature columns
    non_feats = {"label", "binary_label", "attack_type", "difficulty"}
    feature_cols = [c for c in train_enc.columns if c not in non_feats]
    
    # Process test data if provided
    if test_df is not None:
        proc_test = process_nslkdd_dataset(test_df.copy())
        test_enc = pd.get_dummies(proc_test, columns=["protocol_type", "service", "flag"])
        
        # Align test data to training feature space
        for col in feature_cols:
            if col not in test_enc.columns:
                test_enc[col] = 0
                
        # Create aligned test dataset with only the columns from training
        aligned_test = pd.DataFrame()
        
        # Copy feature columns
        for col in feature_cols:
            aligned_test[col] = test_enc[col] if col in test_enc.columns else 0
            
        # Copy label columns
        for col in non_feats:
            if col in test_enc.columns:
                aligned_test[col] = test_enc[col]
        
        return train_enc, aligned_test, feature_cols
    
    return train_enc, None, feature_cols