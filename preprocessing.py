"""Load and preprocess the UCI Adult dataset

Setup: download https://archive.ics.uci.edu/dataset/2/adult, unzip it, and put
adult.data and adult.test in a data/ folder next to this file.

Usage:
    train, test = load_adult()
    y_train, y_test = train["income"], test["income"]  # 1 if income > $50K
    X_train, X_test = onehot(train, test)
"""

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.preprocessing import KBinsDiscretizer, OneHotEncoder, OrdinalEncoder, StandardScaler

DATA_DIR = Path(__file__).parent / "data"

FEATURES = [
    "age", "workclass", "fnlwgt", "education", "education-num", "marital-status",
    "occupation", "relationship", "race", "sex", "capital-gain", "capital-loss",
    "hours-per-week", "native-country",
]  # fmt: skip


def _read(path, skiprows=0):
    df = pd.read_csv(
        path,
        header=None,
        names=FEATURES + ["income"],
        skipinitialspace=True,
        skiprows=skiprows,
    )
    df = df.replace("?", "Unknown")
    # Test labels end in a period ("<=50K."), train labels don't.
    df["income"] = df["income"].str.rstrip(".").eq(">50K").astype(int)
    return df


def load_adult():
    """Return the official train and test split: 14 features plus income as 0/1"""
    train = _read(DATA_DIR / "adult.data")
    test = _read(DATA_DIR / "adult.test", skiprows=1)  # the first line is junk
    return train, test


def onehot(train, test):
    """Apply log1p to the capital columns, scale numerics and one-hot encode categories"""
    X_train, X_test = train[FEATURES].copy(), test[FEATURES].copy()

    capital = ["capital-gain", "capital-loss"]
    X_train[capital] = np.log1p(X_train[capital])
    X_test[capital] = np.log1p(X_test[capital])

    numeric = X_train.select_dtypes("number").columns
    categorical = X_train.select_dtypes(exclude="number").columns

    # Fit on the training set only, then apply to both sets.
    scaler = StandardScaler()
    X_train[numeric] = scaler.fit_transform(X_train[numeric])
    X_test[numeric] = scaler.transform(X_test[numeric])

    encoder = OneHotEncoder(handle_unknown="ignore", sparse_output=False)
    encoder.set_output(transform="pandas")
    train_categories = encoder.fit_transform(X_train[categorical])
    test_categories = encoder.transform(X_test[categorical])

    return (
        pd.concat([X_train[numeric], train_categories], axis=1),
        pd.concat([X_test[numeric], test_categories], axis=1),
    )

def to_categories(train, test, n_bins=10):
    """Apply log1p to the capital columns, bin numerics and encode categories as integers"""
    X_train, X_test = train[FEATURES].copy(), test[FEATURES].copy()

    # Same as onehot(); it also puts the zeros in a bin of their own
    capital = ["capital-gain", "capital-loss"]
    X_train[capital] = np.log1p(X_train[capital])
    X_test[capital] = np.log1p(X_test[capital])

    numeric = X_train.select_dtypes("number").columns
    categorical = X_train.select_dtypes(exclude="number").columns

    # Fit on the training set only, then apply to both sets
    binner = KBinsDiscretizer(n_bins=n_bins, encode="ordinal", strategy="uniform")
    X_train[numeric] = binner.fit_transform(X_train[numeric])
    X_test[numeric] = binner.transform(X_test[numeric])

    encoder = OrdinalEncoder()
    X_train[categorical] = encoder.fit_transform(X_train[categorical])
    X_test[categorical] = encoder.transform(X_test[categorical])

    return X_train.astype(int), X_test.astype(int)