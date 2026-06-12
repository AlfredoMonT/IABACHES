import joblib
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split


DEFAULT_PARAMS = dict(
    n_estimators=300,
    max_depth=None,
    min_samples_leaf=2,
    max_features="sqrt",
    class_weight="balanced",
    n_jobs=-1,
    random_state=42,
)


def build_model(params=None):
    p = DEFAULT_PARAMS.copy()
    if params:
        p.update(params)
    return RandomForestClassifier(**p)


def train(X, y, test_size=0.20, params=None):
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=42, stratify=y
    )
    model = build_model(params)
    model.fit(X_train, y_train)
    return model, X_train, X_test, y_train, y_test


def save_model(model, path):
    joblib.dump(model, path)


def load_model(path):
    return joblib.load(path)


def predict(model, X):
    return model.predict(X), model.predict_proba(X)
