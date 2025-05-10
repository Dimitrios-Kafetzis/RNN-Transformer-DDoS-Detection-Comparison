#!/usr/bin/env python3
import numpy as np
import pandas as pd
from pathlib import Path

def fit_distributions(train_df, num_cols):
    """
    Compute the key percentiles on each numeric feature.
    Returns two Series: p98 and p99.
    """
    p98 = train_df[num_cols].quantile(0.98)
    p99 = train_df[num_cols].quantile(0.99)
    return p98, p99

def sample_benign(train_df, n_samples):
    """
    Sample benign flows with replacement from the original training set.
    """
    return train_df.sample(n=n_samples, replace=True).reset_index(drop=True)

def make_stealth_attacks(train_df, n_samples, num_cols, p98, p99, boost_frac=0.3):
    """
    Take random draws from train_df, then boost a random subset
    of their numeric features into [p98, p99].
    boost_frac controls what fraction of features per sample we tweak.
    """
    attacks = train_df.sample(n=n_samples, replace=True).reset_index(drop=True)
    for col in num_cols:
        # for each feature, randomly select some rows to nudge upward
        mask = np.random.rand(n_samples) < boost_frac
        k = mask.sum()
        if k > 0:
            attacks.loc[mask, col] = np.random.uniform(
                low=p98[col], high=p99[col], size=k
            )
    return attacks

def generate(
    train_csv: Path,
    out_csv:   Path,
    n_benign: int = 5000,
    n_attack:int = 5000
):
    # 1) Load train
    train = pd.read_csv(train_csv)
    
    # 2) Identify numeric vs. categorical cols
    cat_cols = ["protocol_type", "service", "flag"]
    num_cols = [c for c in train.columns if c not in cat_cols + ["label"]]
    
    # 3) Fit distributions
    p98, p99 = fit_distributions(train, num_cols)
    
    # 4) Sample benign and stealth‐attack flows
    benign = sample_benign(train, n_benign)
    attacks = make_stealth_attacks(train, n_attack, num_cols, p98, p99, boost_frac=0.3)
    
    # 5) Label them
    benign["label"] = "normal"
    attacks["label"] = "attack"
    
    # 6) Concatenate and shuffle
    synth = pd.concat([benign, attacks], ignore_index=True)
    synth = synth.sample(frac=1.0, random_state=42).reset_index(drop=True)
    
    # 7) One‐hot encode categoricals exactly as your loader does
    synth_enc = pd.get_dummies(synth, columns=cat_cols)
    
    # 8) Align to original training schema (fills any missing dummy columns with 0)
    train_enc = pd.get_dummies(train,      columns=cat_cols)
    synth_enc, _ = synth_enc.align(train_enc, join="right", axis=1, fill_value=0)
    
    # 9) Restore the label column at the end
    synth_enc["label"] = synth["label"]
    
    # 10) Dump
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    synth_enc.to_csv(out_csv, index=False)
    print(f"Wrote synthetic dataset with {len(synth_enc)} samples to {out_csv}")

if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--train-csv",  type=Path, default="data/nsl_kdd_dataset/NSL-KDD-Train.csv")
    p.add_argument("--out-csv",    type=Path, default="data/nsl_kdd_dataset/NSL-KDD-Hard.csv")
    p.add_argument("--n-benign",   type=int,  default=5000)
    p.add_argument("--n-attack",   type=int,  default=5000)
    args = p.parse_args()
    generate(args.train_csv, args.out_csv, args.n_benign, args.n_attack)
