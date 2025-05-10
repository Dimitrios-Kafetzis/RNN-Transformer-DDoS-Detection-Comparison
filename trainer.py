# trainer.py
import time
import json
import inspect
import numpy as np
import tensorflow as tf
import pickle
import os
from evaluation_config import HISTORY_DIR
from pathlib import Path
from data.loader import prepare_combined_dataset
from models.linear_regressor import create_linear_model
from models.threshold_detector import ThresholdDetector
from models.shallow_dnn import create_shallow_dnn_model
from models.dnn import create_dnn_model
from models.lstm import create_lstm_model
from models.gru import create_gru_model
from models.transformer import create_transformer_model

import logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

def train_and_save(model_fn, name, X, y, timestamp, mean=None, std=None, **fit_kwargs):
    """
    Instantiate model_fn, possibly reshape X for RNNs, train on (X,y),
    and save under saved_models/<name>_<timestamp>/.
    """
    sig = inspect.signature(model_fn)
    # detect whether factory expects input_dim or input_shape
    if "input_dim" in sig.parameters:
        model = model_fn(input_dim=X.shape[1])
        X_train = X
    else:
        # assume RNN: expects input_shape=(timesteps, features)
        # choose timesteps=1, features=X.shape[1]
        timesteps = 1
        features = X.shape[1]
        model = model_fn(input_shape=(timesteps, features))
        X_train = X.reshape((-1, timesteps, features))
        
    # Add callbacks for NaN detection
    callbacks = fit_kwargs.get('callbacks', [])
    callbacks.append(tf.keras.callbacks.TerminateOnNaN())
    fit_kwargs['callbacks'] = callbacks
    
    # Save normalization parameters
    outdir = Path("saved_models") / f"{name}_{timestamp}"
    outdir.mkdir(parents=True, exist_ok=True)
    if mean is not None and std is not None:
        np.save(str(outdir / "X_mean.npy"), mean)
        np.save(str(outdir / "X_std.npy"), std)
    
    logger.info(f"Training {name} on {len(X_train)} samples…")
    history = model.fit(X_train, y, **fit_kwargs)

    # Save training history
    history_path = os.path.join(HISTORY_DIR, f"{name}_{timestamp}_history.pkl")
    with open(history_path, 'wb') as f:
        pickle.dump(history.history, f)

    logger.info(f"Saving {name} to {outdir}")
    model.save(str(outdir), save_format="tf")
    return outdir

def main():
    ts = int(time.time())
    PROJECT_ROOT = Path(__file__).parent.resolve()
    data_dir = PROJECT_ROOT / "data" / "nsl_kdd_dataset"
    hard_csv = data_dir / "NSL-KDD-Hard.csv"

    # 1) Prepare datasets
    (X_train, y_train), (X_easy, y_easy), (X_hard, y_hard), feat_cols, (mean, std) = \
        prepare_combined_dataset(str(data_dir), str(hard_csv))

    # 2) Common fit args
    fit_args = dict(
      epochs=20,
      batch_size=64,
      validation_split=0.1,
      callbacks=[
          tf.keras.callbacks.EarlyStopping(
              patience=5, 
              restore_best_weights=True, 
              monitor='val_loss'
          ),
          tf.keras.callbacks.ReduceLROnPlateau(
              monitor='val_loss', 
              factor=0.5, 
              patience=2
          )
      ],
      verbose=2
    )

    # 3) Train each model family
    saved = {}
    saved['linear_model'] = train_and_save(
        create_linear_model, "linear_model",
        X_train, y_train, ts, mean, std, **fit_args
    )

    # Threshold detector: no training but we save its calibrated thresholds
    thr = ThresholdDetector(percentile=99.5)
    thr.calibrate(X_train, y_train, feature_indices=list(range(X_train.shape[1])))
    outdir = Path("saved_models") / f"threshold_detector_{ts}"
    outdir.mkdir(parents=True, exist_ok=True)
    with open(outdir / "thresholds.json", "w") as f:
        json.dump(thr.thresholds, f)
    # Save normalization parameters for consistency
    np.save(str(outdir / "X_mean.npy"), mean)
    np.save(str(outdir / "X_std.npy"), std)
    saved['threshold_detector'] = str(outdir)

    saved['shallow_dnn'] = train_and_save(
        create_shallow_dnn_model, "shallow_dnn",
        X_train, y_train, ts, mean, std, **fit_args
    )
    
    saved['dnn'] = train_and_save(
        create_dnn_model, "dnn",
        X_train, y_train, ts, mean, std, **fit_args
    )
    
    saved['lstm'] = train_and_save(
        create_lstm_model, "lstm",
        X_train, y_train, ts, mean, std, **fit_args
    )
    
    saved['gru'] = train_and_save(
        create_gru_model, "gru",
        X_train, y_train, ts, mean, std, **fit_args
    )
    
    saved['transformer'] = train_and_save(
        create_transformer_model, "transformer",
        X_train, y_train, ts, mean, std, **fit_args
    )

    # 4) Write out manifest
    saved = {k: str(v) for k, v in saved.items()}  # Convert all paths to strings
    with open("saved_models/manifest.json", "w") as f:
        json.dump(saved, f, indent=2)
    
    logger.info("All models retrained and saved.")

if __name__ == "__main__":
    main()
