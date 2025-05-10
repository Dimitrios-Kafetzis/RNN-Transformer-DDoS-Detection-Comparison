#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
File: models/linear_regressor.py
Author: Dimitrios Kafetzis (kafetzis@aueb.gr)
Created: May 2025
License: MIT

Description:
    Implementation of a simple linear model for DDoS detection.
    Creates a single dense layer with sigmoid activation for binary classification.
    Serves as a baseline model to evaluate the benefit of more complex architectures.

Usage:
    This module is imported by other scripts and not meant to be run directly.
    
    Import example:
    from models.linear_regressor import create_linear_model
    
    Usage example:
    model = create_linear_model(input_dim=41, learning_rate=0.001)
    model.fit(X_train, y_train, epochs=10)
"""

import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.keras.optimizers import Adam
from models.metrics import F1Score

def create_linear_model(input_dim, learning_rate=1e-3):
    inputs = tf.keras.Input(shape=(input_dim,))
    outputs = layers.Dense(1, activation="sigmoid")(inputs)
    model = tf.keras.Model(inputs, outputs, name="linear_model")
    model.compile(
        optimizer=Adam(learning_rate),
        loss="binary_crossentropy",
        metrics=["accuracy", F1Score()]
    )
    return model
