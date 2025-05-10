#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
File: inference/inference_engine.py
Author: Dimitrios Kafetzis (kafetzis@aueb.gr)
Created: May 2025
License: MIT

Description:
    Provides inference functionality for DDoS detection models.
    Loads trained models, processes input features, and generates
    predictions with detection reports.

Usage:
    This module is imported by other scripts and not meant to be run directly.
    
    Import example:
    from inference.inference_engine import run_inference
    
    Usage example:
    results = run_inference(features, timestamps, model_path, threshold=0.5)
"""

import os
import time
import json
import logging
import numpy as np
import tensorflow as tf
from typing import Dict, List, Tuple, Any, Optional, Union
from datetime import datetime, timedelta

logger = logging.getLogger(__name__)

class InferenceEngine:
    """Engine to perform DDoS attack detection inference."""
    
    def __init__(self, model_path: str, model_type: str = None, threshold: float = 0.5):
        """
        Initialize inference engine.
        
        Args:
            model_path: Path to the trained model directory
            model_type: Type of model for special handling (e.g., "transformer", "threshold_detector")
            threshold: Decision threshold for binary classification
        """
        self.model_path = model_path
        self.model_type = model_type or os.path.basename(model_path).split('_')[0]
        self.threshold = threshold
        self.model = None
        
        # Load model based on type
        self._load_model()
    
    def _load_model(self):
        """Load the appropriate model based on model type."""
        try:
            # Special handling for threshold detector
            if self.model_type == "threshold_detector":
                from models.threshold_detector import ThresholdDetector
                
                thresholds_path = os.path.join(self.model_path, "thresholds.json")
                if not os.path.exists(thresholds_path):
                    raise FileNotFoundError(f"Thresholds file not found: {thresholds_path}")
                
                with open(thresholds_path, 'r') as f:
                    thresholds = json.load(f)
                
                detector = ThresholdDetector()
                detector.thresholds = {int(k): float(v) for k, v in thresholds.items()}
                detector.is_calibrated = True
                self.model = detector
                logger.info(f"Loaded threshold detector with {len(detector.thresholds)} thresholds")
            else:
                # Load TensorFlow model
                self.model = tf.keras.models.load_model(self.model_path, compile=False)
                logger.info(f"Loaded TensorFlow model from {self.model_path}")
                
                # Get model input shape for later reshaping if needed
                self.input_shape = self.model.input_shape
                logger.info(f"Model input shape: {self.input_shape}")
        
        except Exception as e:
            logger.error(f"Error loading model: {e}")
            raise
    
    def preprocess_input(self, features: np.ndarray) -> np.ndarray:
        """
        Preprocess features for model input.
        
        Args:
            features: Raw feature array
            
        Returns:
            Preprocessed features ready for model inference
        """
        # Handle threshold detector separately
        if self.model_type == "threshold_detector":
            return features
        
        # Handle sequence models (LSTM, GRU, Transformer)
        if len(self.input_shape) > 2:  # This is a sequence model
            timesteps = self.input_shape[1]
            feature_dim = self.input_shape[2]
            
            # Ensure feature dimensions match
            if features.shape[1] != feature_dim:
                logger.warning(f"Feature dimension mismatch: expected {feature_dim}, got {features.shape[1]}")
                
                if features.shape[1] < feature_dim:
                    # Pad with zeros
                    pad_width = ((0, 0), (0, feature_dim - features.shape[1]))
                    features = np.pad(features, pad_width, mode='constant')
                else:
                    # Truncate
                    features = features[:, :feature_dim]
            
            # Reshape to match expected input shape (batch_size, timesteps, features)
            batch_size = features.shape[0]
            
            # Handle case where there are fewer samples than timesteps
            if batch_size < timesteps:
                # Pad with zeros to match timesteps
                pad_size = timesteps - batch_size
                padding = np.zeros((pad_size, features.shape[1]))
                features = np.vstack([features, padding])
                batch_size = timesteps
            
            # Create sliding windows of size timesteps
            windows = []
            for i in range(batch_size - timesteps + 1):
                windows.append(features[i:i+timesteps])
            
            # Convert to numpy array
            if windows:
                return np.array(windows)
            else:
                # Create a single window with padded data
                return np.array([features[:timesteps]])
        
        # Regular models (non-sequence)
        return features
    
    def predict(self, features: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Generate predictions from features.
        
        Args:
            features: Preprocessed feature array
            
        Returns:
            Tuple of (binary_predictions, probability_scores)
        """
        start_time = time.time()
        
        # Handle threshold detector separately
        if self.model_type == "threshold_detector":
            feature_indices = list(range(min(features.shape[1], len(self.model.thresholds))))
            binary_preds = self.model.predict(features, feature_indices)
            
            # Generate pseudo-probabilities
            scores = np.zeros(len(features))
            for idx in feature_indices:
                if idx < features.shape[1] and idx < len(self.model.thresholds):
                    threshold = self.model.thresholds[idx]
                    excess = features[:, idx] - threshold
                    scores = np.maximum(scores, excess)
            
            # Convert to probability-like values
            probs = 1 / (1 + np.exp(-scores))
            
            elapsed = time.time() - start_time
            logger.info(f"Threshold detector inference completed in {elapsed:.4f} seconds")
            return binary_preds, probs
        
        # Regular model inference
        inputs = self.preprocess_input(features)
        
        # Handle empty inputs
        if inputs.shape[0] == 0:
            logger.warning("Empty input provided to model")
            return np.array([]), np.array([])
        
        # Get raw probabilities
        raw_probs = self.model.predict(inputs).flatten()
        
        # Handle Transformer model's special threshold needs
        if self.model_type == "transformer":
            # Transformer models often output compressed probabilities
            effective_threshold = min(0.1, self.threshold)
            logger.info(f"Using adjusted threshold for transformer: {effective_threshold}")
        else:
            effective_threshold = self.threshold
        
        # Convert to binary predictions
        binary_preds = (raw_probs >= effective_threshold).astype(int)
        
        elapsed = time.time() - start_time
        logger.info(f"Model inference completed in {elapsed:.4f} seconds")
        
        return binary_preds, raw_probs

def run_inference(
    features: np.ndarray, 
    timestamps: List[float], 
    model_path: str, 
    model_type: str = None, 
    threshold: float = 0.5
) -> Dict[str, Any]:
    """
    Run complete inference pipeline and generate detection report.
    
    Args:
        features: Preprocessed feature array
        timestamps: List of timestamps for each window
        model_path: Path to the trained model
        model_type: Type of model (optional)
        threshold: Decision threshold
        
    Returns:
        Dictionary with inference results and attack report
    """
    # Initialize inference engine
    engine = InferenceEngine(model_path, model_type, threshold)
    
    # Run inference
    binary_preds, probabilities = engine.predict(features)
    
    # Generate attack report
    report = generate_attack_report(binary_preds, probabilities, timestamps, engine.model_type)
    
    # Return combined results
    return {
        "model_type": engine.model_type,
        "model_path": model_path,
        "threshold": threshold,
        "predictions": binary_preds.tolist() if isinstance(binary_preds, np.ndarray) else binary_preds,
        "probabilities": probabilities.tolist() if isinstance(probabilities, np.ndarray) else probabilities,
        "timestamps": [datetime.fromtimestamp(ts).isoformat() for ts in timestamps],
        "report": report
    }

def generate_attack_report(
    predictions: np.ndarray, 
    probabilities: np.ndarray, 
    timestamps: List[float],
    model_type: str
) -> Dict[str, Any]:
    """
    Generate a textual report of detected attacks.
    
    Args:
        predictions: Binary predictions array
        probabilities: Prediction probabilities
        timestamps: List of timestamps
        model_type: Type of model used
        
    Returns:
        Dictionary with attack report information
    """
    # Handle empty predictions
    if len(predictions) == 0 or len(timestamps) == 0:
        return {
            "is_attack_detected": False,
            "attack_count": 0,
            "total_count": 0,
            "attack_ratio": 0.0,
            "attack_windows": [],
            "max_confidence": 0.0,
            "mean_confidence": 0.0
        }
    
    # Convert timestamps to datetime objects
    dt_timestamps = [datetime.fromtimestamp(ts) for ts in timestamps]
    
    # Count attacks
    attack_count = np.sum(predictions)
    total_count = len(predictions)
    attack_ratio = float(attack_count) / total_count if total_count > 0 else 0
    
    # Find attack windows (consecutive detected attacks)
    attack_windows = []
    current_window = None
    
    for i, (pred, prob, ts) in enumerate(zip(predictions, probabilities, dt_timestamps)):
        if pred == 1:  # Attack detected
            if current_window is None:
                # Start new window
                current_window = {
                    "start_idx": i,
                    "end_idx": i,
                    "start_time": ts.isoformat(),
                    "end_time": ts.isoformat(),
                    "confidence": [float(prob)],
                    "duration_seconds": 0
                }
            else:
                # Extend current window
                current_window["end_idx"] = i
                current_window["end_time"] = ts.isoformat()
                current_window["confidence"].append(float(prob))
        elif current_window is not None:
            # End of attack window
            # Calculate duration
            start_time = datetime.fromisoformat(current_window["start_time"])
            end_time = datetime.fromisoformat(current_window["end_time"])
            current_window["duration_seconds"] = (end_time - start_time).total_seconds()
            
            # Calculate average confidence
            current_window["avg_confidence"] = np.mean(current_window["confidence"])
            current_window["max_confidence"] = np.max(current_window["confidence"])
            
            # Add to list of attack windows
            attack_windows.append(current_window)
            current_window = None
    
    # Add final window if it exists
    if current_window is not None:
        start_time = datetime.fromisoformat(current_window["start_time"])
        end_time = datetime.fromisoformat(current_window["end_time"])
        current_window["duration_seconds"] = (end_time - start_time).total_seconds()
        current_window["avg_confidence"] = np.mean(current_window["confidence"])
        current_window["max_confidence"] = np.max(current_window["confidence"])
        attack_windows.append(current_window)
    
    # Calculate confidence metrics
    attack_indices = np.where(predictions == 1)[0]
    max_confidence = float(np.max(probabilities[attack_indices])) if len(attack_indices) > 0 else 0
    mean_confidence = float(np.mean(probabilities[attack_indices])) if len(attack_indices) > 0 else 0
    
    # Determine attack types based on heuristics
    attack_types = determine_attack_types(predictions, probabilities, model_type)
    
    return {
        "is_attack_detected": attack_count > 0,
        "attack_count": int(attack_count),
        "total_count": int(total_count),
        "attack_ratio": float(attack_ratio),
        "attack_windows": attack_windows,
        "max_confidence": max_confidence,
        "mean_confidence": mean_confidence,
        "suspected_attack_types": attack_types
    }

def determine_attack_types(predictions: np.ndarray, probabilities: np.ndarray, model_type: str) -> List[str]:
    """
    Simple heuristic to determine likely attack types based on detection patterns.
    
    Args:
        predictions: Binary predictions array
        probabilities: Prediction probabilities
        model_type: Type of model used
        
    Returns:
        List of likely attack types
    """
    attack_types = []
    
    # Return early if no attacks detected
    if np.sum(predictions) == 0:
        return attack_types
    
    # Calculate some basic metrics
    attack_ratio = np.mean(predictions)
    max_confidence = np.max(probabilities[predictions == 1])
    
    # Look for consecutive attacks
    consecutive_attacks = 0
    max_consecutive = 0
    for p in predictions:
        if p == 1:
            consecutive_attacks += 1
            max_consecutive = max(max_consecutive, consecutive_attacks)
        else:
            consecutive_attacks = 0
    
    # Apply simple heuristics
    # Note: These are very simplified heuristics and would need to be refined
    # with domain expertise for a production system
    
    # High volume attacks (high percentage of windows flagged)
    if attack_ratio > 0.7:
        attack_types.append("TCP SYN Flood")
        attack_types.append("UDP Flood")
    
    # Moderate volume, high confidence
    elif attack_ratio > 0.3 and max_confidence > 0.8:
        attack_types.append("HTTP Flood")
        attack_types.append("ICMP Flood")
    
    # Long consecutive attack windows
    if max_consecutive > 5:
        attack_types.append("TCP SYN Flood")
    
    # Low ratio but still detected
    if attack_ratio < 0.2 and attack_ratio > 0:
        attack_types.append("Low-and-Slow")
    
    # Default if no specific pattern identified
    if not attack_types:
        attack_types.append("Unknown DDoS")
    
    return list(set(attack_types))  # Remove duplicates