# models/transformer.py
import tensorflow as tf
import numpy as np
from tensorflow.keras import layers
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.metrics import Precision, Recall, AUC
from models.metrics import F1Score

class TransformerBlock(layers.Layer):
    def __init__(self, embed_dim, num_heads, ff_dim, rate=0.1):
        super().__init__()
        self.att = layers.MultiHeadAttention(num_heads=num_heads, key_dim=embed_dim)
        self.ffn = tf.keras.Sequential([
            layers.Dense(ff_dim, activation="relu", kernel_initializer='he_normal'),
            layers.Dense(embed_dim, kernel_initializer='glorot_uniform'),
        ])
        self.ln1 = layers.LayerNormalization(epsilon=1e-6)
        self.ln2 = layers.LayerNormalization(epsilon=1e-6)
        self.do1 = layers.Dropout(rate)
        self.do2 = layers.Dropout(rate)

    def call(self, x, training):
        attn = self.att(x, x)
        attn = self.do1(attn, training=training)
        out1 = self.ln1(x + attn)
        ffn = self.ffn(out1)
        ffn = self.do2(ffn, training=training)
        return self.ln2(out1 + ffn)

def positional_encoding(length, depth):
    """Create positional encodings for the Transformer"""
    depth = depth/2
    positions = np.arange(length)[:, np.newaxis]
    depths = np.arange(depth)[np.newaxis, :]/depth
    angle_rates = 1 / (10000**depths)
    angle_rads = positions * angle_rates
    pos_encoding = np.concatenate([np.sin(angle_rads), np.cos(angle_rads)], axis=-1)
    return tf.cast(pos_encoding, dtype=tf.float32)

def create_transformer_model(
    input_shape,
    embed_dim=32,
    num_heads=2,
    ff_dim=32,
    rate=0.1,
    learning_rate=1e-3,
    l2_reg=1e-5
):
    inputs = tf.keras.Input(shape=input_shape)
    
    # Apply proper embedding
    x = layers.Dense(embed_dim, kernel_initializer='he_normal')(inputs)
    
    # Add positional encoding for sequence data
    if len(input_shape) > 1 and input_shape[0] > 1:  # For sequential data
        pos_encoding = positional_encoding(input_shape[0], embed_dim)
        x = x + pos_encoding[:input_shape[0], :]
    
    x = TransformerBlock(embed_dim, num_heads, ff_dim, rate)(x)
    x = layers.GlobalAveragePooling1D()(x)
    x = layers.Dropout(rate)(x)
    x = layers.Dense(16, activation="relu", kernel_initializer='he_normal', kernel_regularizer=tf.keras.regularizers.l2(l2_reg))(x)
    outputs = layers.Dense(1, activation="sigmoid", kernel_initializer='glorot_uniform')(x)
    
    model = tf.keras.Model(inputs, outputs, name="transformer")
    model.compile(
        optimizer=Adam(learning_rate=learning_rate, clipnorm=1.0),  # Add gradient clipping
        loss="binary_crossentropy",
        metrics=["accuracy", Precision(), Recall(), AUC(), F1Score()]
    )
    return model