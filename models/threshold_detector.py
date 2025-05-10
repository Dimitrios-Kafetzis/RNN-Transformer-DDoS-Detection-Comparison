#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
File: models/threshold_detector.py
Author: Dimitrios Kafetzis (kafetzis@aueb.gr)
Created: May 2025
License: MIT

Description:
    Implementation of a simple percentile-based threshold detector.
    Calibrates thresholds based on benign training data and flags attacks
    when any feature exceeds its threshold. Serves as a baseline detector
    for comparison against more sophisticated neural network approaches.

Usage:
    This module is imported by other scripts and not meant to be run directly.
    
    Import example:
    from models.threshold_detector import ThresholdDetector
    
    Usage example:
    detector = ThresholdDetector(percentile=99.5)
    detector.calibrate(X_train, y_train, feature_indices)
    y_pred = detector.predict(X_test, feature_indices)
"""

import numpy as np
import time
import logging
import json
import os
from typing import List, Dict, Optional, Any, Tuple

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

class ThresholdDetector:
    """
    A simple percentile‐based threshold detector.
    """
    def __init__(self,
                 name: str = "threshold_detector",
                 percentile: float = 99.5):
        self.name = name
        self.percentile = percentile
        self.thresholds: Dict[int, float] = {}
        self.is_calibrated = False

    def calibrate(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        feature_indices: List[int]
    ) -> "ThresholdDetector":
        logger.info(f"Calibrating {self.name} on {len(X_train)} samples")
        benign = X_train[y_train == 0]
        for idx in feature_indices:
            self.thresholds[idx] = float(np.percentile(benign[:, idx], self.percentile))
            logger.info(f"  feature {idx} → threshold {self.thresholds[idx]:.4f}")
        self.is_calibrated = True
        return self

    def predict(
        self,
        X: np.ndarray,
        feature_indices: List[int]
    ) -> np.ndarray:
        if not self.is_calibrated:
            raise RuntimeError("Must calibrate first")
        # flag attack if **any** feature exceeds its threshold
        flags = np.zeros(len(X), dtype=int)
        for idx in feature_indices:
            flags |= (X[:, idx] > self.thresholds[idx]).astype(int)
        return flags

    def save_thresholds(self, path: str):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "w") as f:
            json.dump(self.thresholds, f)
