# models/linear_regressor.py
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
