#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
File: models/dnn.py
Author: Dimitrios Kafetzis (kafetzis@aueb.gr)
Created: May 2025
License: MIT

Description:
    Implementation of a deep neural network for DDoS detection.
    Creates a network with three hidden layers, batch normalization, and dropout.
    Includes L2 regularization and gradient clipping to prevent overfitting.

Usage:
    This module is imported by other scripts and not meant to be run directly.
    
    Import example:
    from models.dnn import create_dnn_model
    
    Usage example:
    model = create_dnn_model(input_dim=41, hidden1=128, hidden2=64, hidden3=32)
    model.fit(X_train, y_train, epochs=20)
"""

import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.metrics import Precision, Recall, AUC
from models.metrics import F1Score

def create_dnn_model(
    input_dim,
    hidden1=128,
    hidden2=64,
    hidden3=32,
    rate=0.2,
    learning_rate=1e-3,
    l2_reg=1e-5
):
    inputs = tf.keras.Input(shape=(input_dim,))
    x = layers.Dense(hidden1, activation="relu", kernel_initializer='he_normal', kernel_regularizer=tf.keras.regularizers.l2(l2_reg))(inputs)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(rate)(x)
    x = layers.Dense(hidden2, activation="relu", kernel_initializer='he_normal', kernel_regularizer=tf.keras.regularizers.l2(l2_reg))(x)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(rate)(x)
    x = layers.Dense(hidden3, activation="relu", kernel_initializer='he_normal', kernel_regularizer=tf.keras.regularizers.l2(l2_reg))(x)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(rate)(x)
    outputs = layers.Dense(1, activation="sigmoid", kernel_initializer='glorot_uniform')(x)
    model = tf.keras.Model(inputs, outputs, name="dnn")
    model.compile(
        optimizer=Adam(learning_rate=learning_rate, clipnorm=1.0),  # Add gradient clipping
        loss="binary_crossentropy",
        metrics=["accuracy", Precision(), Recall(), AUC(), F1Score()]
    )
    return model