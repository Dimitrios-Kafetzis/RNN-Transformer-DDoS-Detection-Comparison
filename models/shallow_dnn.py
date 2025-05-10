# models/shallow_dnn.py
import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.metrics import Precision, Recall, AUC
from models.metrics import F1Score

def create_shallow_dnn_model(
    input_dim,
    hidden1=64,
    hidden2=32,
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
    outputs = layers.Dense(1, activation="sigmoid", kernel_initializer='glorot_uniform')(x)
    model = tf.keras.Model(inputs, outputs, name="shallow_dnn")
    model.compile(
        optimizer=Adam(learning_rate=learning_rate, clipnorm=1.0),  # Add gradient clipping
        loss="binary_crossentropy",
        metrics=["accuracy", Precision(), Recall(), AUC(), F1Score()]
    )
    return model