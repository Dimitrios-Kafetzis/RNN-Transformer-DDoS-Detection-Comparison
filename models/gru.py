#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
File: models/gru.py
Author: Dimitrios Kafetzis (kafetzis@aueb.gr)
Created: May 2025
License: MIT

Description:
    Implementation of a Gated Recurrent Unit (GRU) network for DDoS detection.
    Creates a bidirectional GRU architecture that processes traffic sequences.
    GRU cells use fewer parameters than LSTM by combining input and forget gates,
    potentially offering comparable performance with lower computational cost.

Usage:
    This module is imported by other scripts and not meant to be run directly.
    
    Import example:
    from models.gru import create_gru_model
    
    Usage example:
    model = create_gru_model(input_shape=(10, 41), gru_units=64)
    model.fit(X_train, y_train, epochs=20)
"""

import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.metrics import Precision, Recall, AUC
from models.metrics import F1Score

def create_gru_model(
    input_shape,
    gru_units=64,
    rate=0.2,
    learning_rate=1e-3,
    l2_reg=1e-5
):
    """
    input_shape: (timesteps, features)
    """
    inputs = tf.keras.Input(shape=input_shape)
    
    # Use better initializers and add gradient clipping
    x = layers.GRU(
        gru_units, 
        return_sequences=True,
        kernel_initializer='glorot_uniform',
        recurrent_initializer='orthogonal',
        bias_initializer='zeros'
    )(inputs)
    
    x = layers.GRU(
        gru_units // 2,
        kernel_initializer='glorot_uniform',
        recurrent_initializer='orthogonal',
        bias_initializer='zeros'
    )(x)
    
    x = layers.Dropout(rate)(x)
    x = layers.Dense(16, activation="relu", kernel_initializer='he_normal', kernel_regularizer=tf.keras.regularizers.l2(l2_reg))(x)
    outputs = layers.Dense(1, activation="sigmoid", kernel_initializer='glorot_uniform')(x)
    
    model = tf.keras.Model(inputs, outputs, name="gru")
    model.compile(
        optimizer=Adam(learning_rate=learning_rate, clipnorm=1.0),  # Add gradient clipping
        loss="binary_crossentropy",
        metrics=["accuracy", Precision(), Recall(), AUC(), F1Score()]
    )
    return model