# preprocess_hard.py
import pandas as pd
import numpy as np
from data.loader import load_nslkdd_dataset, process_nslkdd_dataset

# Load and process train data to get feature space
train_df, _ = load_nslkdd_dataset("data/nsl_kdd_dataset", split="both")
proc_train = process_nslkdd_dataset(train_df.copy())
train_enc = pd.get_dummies(proc_train, columns=["protocol_type", "service", "flag"])
non_feats = {"label", "binary_label", "attack_type", "difficulty"}
feature_cols = [c for c in train_enc.columns if c not in non_feats]

# Load hard dataset (assuming it's raw format)
hard_csv = "data/nsl_kdd_dataset/NSL-KDD-Hard.csv"
header = pd.read_csv(hard_csv, nrows=0).columns.tolist()

if "protocol_type" in header:
    # Raw format
    print("Processing raw hard dataset...")
    raw_hard = pd.read_csv(hard_csv, header=None, names=train_df.columns)
    proc_hard = process_nslkdd_dataset(raw_hard)
    hard_enc = pd.get_dummies(proc_hard, columns=["protocol_type", "service", "flag"])
else:
    # Already one-hot encoded
    print("Hard dataset already one-hot encoded...")
    hard_enc = pd.read_csv(hard_csv)

# Ensure all training features exist in hard dataset
for col in feature_cols:
    if col not in hard_enc.columns:
        hard_enc[col] = 0

# Ensure all columns are aligned
hard_aligned = hard_enc.reindex(columns=train_enc.columns, fill_value=0)

# Save properly aligned dataset
hard_aligned.to_csv("data/nsl_kdd_dataset/NSL-KDD-Hard-Aligned.csv", index=False)
print(f"Original shape: {hard_enc.shape}")
print(f"Aligned shape: {hard_aligned.shape}")
print("Saved aligned hard dataset to NSL-KDD-Hard-Aligned.csv")