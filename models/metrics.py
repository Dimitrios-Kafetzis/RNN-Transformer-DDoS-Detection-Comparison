#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
File: models/metrics.py
Author: Dimitrios Kafetzis (kafetzis@aueb.gr)
Created: May 2025
License: MIT

Description:
    Custom metrics for model evaluation in TensorFlow.
    Implements an F1 score metric compatible with Keras, enabling
    F1 scoring during model training and evaluation.

Usage:
    This module is imported by other scripts and not meant to be run directly.
    
    Import example:
    from models.metrics import F1Score
    
    Usage examples:
    # Direct usage
    f1 = F1Score()
    f1.update_state(y_true, y_pred)
    result = f1.result()
    
    # In model compilation
    model.compile(metrics=[F1Score()])
"""

import tensorflow as tf
from tensorflow.keras.utils import register_keras_serializable

@register_keras_serializable()
class F1Score(tf.keras.metrics.Metric):
    """Binary F1 score with a fixed threshold."""
    def __init__(self, name="f1_score", threshold=0.5, **kwargs):
        super().__init__(name=name, **kwargs)
        self.threshold = threshold
        self.tp = self.add_weight(name="tp", initializer="zeros")
        self.fp = self.add_weight(name="fp", initializer="zeros")
        self.fn = self.add_weight(name="fn", initializer="zeros")

    def update_state(self, y_true, y_pred, sample_weight=None):
        y_pred = tf.cast(y_pred > self.threshold, tf.float32)
        y_true = tf.cast(y_true, tf.float32)
        tp = tf.reduce_sum(y_true * y_pred)
        fp = tf.reduce_sum((1 - y_true) * y_pred)
        fn = tf.reduce_sum(y_true * (1 - y_pred))
        self.tp.assign_add(tp)
        self.fp.assign_add(fp)
        self.fn.assign_add(fn)

    def result(self):
        precision = self.tp / (self.tp + self.fp + tf.keras.backend.epsilon())
        recall = self.tp / (self.tp + self.fn + tf.keras.backend.epsilon())
        return 2 * (precision * recall) / (precision + recall + tf.keras.backend.epsilon())

    def reset_states(self):
        for var in (self.tp, self.fp, self.fn):
            var.assign(0.0)
