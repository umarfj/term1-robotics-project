"""
Classifier module following OOP principles.

Defines an abstract BaseClassifier and concrete implementations
for SVM and Random Forest. Centralizes save/load paths under src/learning.
"""

from abc import ABC, abstractmethod
from typing import Optional, Dict, Any
import os
import pickle

import numpy as np
from sklearn.metrics import accuracy_score, classification_report
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, GridSearchCV
import pandas as pd


MODELS_DIR = os.path.join(os.path.dirname(__file__), "")  # src/learning


class BaseClassifier(ABC):
    """Abstract base classifier with a common interface."""

    def __init__(self):
        self.model: Optional[Any] = None

    @abstractmethod
    def build(self):
        """Instantiate the underlying model/pipeline."""
        pass

    def train(self, X: np.ndarray, y: np.ndarray):
        if self.model is None:
            self.build()
        self.model.fit(X, y)

    def predict(self, X: np.ndarray) -> np.ndarray:
        if self.model is None:
            raise RuntimeError("Model not built or loaded.")
        return self.model.predict(X)

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        if self.model is None:
            raise RuntimeError("Model not built or loaded.")
        # Not all models support predict_proba
        if hasattr(self.model, "predict_proba"):
            return self.model.predict_proba(X)
        raise AttributeError("This model does not support predict_proba.")

    def evaluate(self, X_test: np.ndarray, y_test: np.ndarray) -> Dict[str, Any]:
        preds = self.predict(X_test)
        acc = accuracy_score(y_test, preds)
        report = classification_report(y_test, preds)
        return {"accuracy": acc, "classification_report": report, "predictions": preds}

    def save(self, filename: str):
        if self.model is None:
            raise RuntimeError("No model to save.")
        path = filename if os.path.isabs(filename) else os.path.join(MODELS_DIR, filename)
        with open(path, "wb") as f:
            pickle.dump(self.model, f)

    def load(self, filename: str):
        path = filename if os.path.isabs(filename) else os.path.join(MODELS_DIR, filename)
        with open(path, "rb") as f:
            self.model = pickle.load(f)


class SVMClassifier(BaseClassifier):
    """SVM classifier wrapped in a scaler+SVC pipeline."""

    def __init__(self, probability: bool = True, C: float = 1.0, gamma: str | float = "scale", kernel: str = "rbf", cache_size: int = 200):
        super().__init__()
        self.probability = probability
        self.C = C
        self.gamma = gamma
        self.kernel = kernel
        self.cache_size = cache_size

    def build(self):
        svc = SVC(probability=self.probability, C=self.C, gamma=self.gamma, kernel=self.kernel, cache_size=self.cache_size, random_state=42)
        self.model = Pipeline([
            ("scaler", StandardScaler()),
            ("svc", svc)
        ])

    def train_from_csv_with_grid(self, data_path: str, model_filename: str = "best_grasp_classifier_svm.pkl", test_size: float = 0.2, random_state: int = 42,
                                 param_grid: Dict[str, Any] | None = None, cv: int = 5, n_jobs: int = -1, verbose: int = 1) -> Dict[str, Any]:
        """Replicates main_train_svm.py behavior: load, balance, split, grid search, evaluate, save."""
        if param_grid is None:
            param_grid = {
                'svc__C': [0.1, 1, 10, 100],
                'svc__gamma': ['scale', 'auto', 0.1, 1],
                'svc__kernel': ['rbf', 'linear']
            }

        # Load and balance
        if not os.path.exists(data_path):
            raise FileNotFoundError(f"CSV not found: {data_path}")
        df = pd.read_csv(data_path)
        success_df = df[df['label'] == 1]
        fail_df = df[df['label'] == 0]
        min_count = min(len(success_df), len(fail_df))
        df_balanced = pd.concat([
            success_df.sample(n=min_count, random_state=random_state),
            fail_df.sample(n=min_count, random_state=random_state)
        ]).sample(frac=1, random_state=random_state).reset_index(drop=True)

        feature_cols = ['pos_x', 'pos_y', 'pos_z', 'orn_x', 'orn_y', 'orn_z', 'orn_w']
        X = df_balanced[feature_cols].values
        y = df_balanced['label'].values

        X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=test_size, random_state=random_state)

        # Build base pipeline and run grid search
        self.build()
        grid = GridSearchCV(
            estimator=self.model,
            param_grid=param_grid,
            cv=cv,
            n_jobs=n_jobs,
            verbose=verbose,
            scoring='accuracy'
        )
        grid.fit(X_train, y_train)
        self.model = grid.best_estimator_

        # Evaluate
        y_pred = self.model.predict(X_val)
        acc = accuracy_score(y_val, y_pred)
        report = classification_report(y_val, y_pred, target_names=["Fail (0)", "Success (1)"])

        # Save
        self.save(model_filename)

        return {"best_params": grid.best_params_, "cv_score": grid.best_score_, "val_accuracy": acc, "classification_report": report}


class RFClassifier(BaseClassifier):
    """Random Forest classifier."""

    def __init__(self, n_estimators: int = 100, max_depth: Optional[int] = None, n_jobs: int = -1, random_state: int = 42):
        super().__init__()
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.n_jobs = n_jobs
        self.random_state = random_state

    def build(self):
        self.model = RandomForestClassifier(
            n_estimators=self.n_estimators,
            max_depth=self.max_depth,
            n_jobs=self.n_jobs,
            random_state=self.random_state
        )

    def train_from_csv(self, data_path: str, model_filename: str = "best_grasp_classifier_rf.pkl", test_size: float = 0.2, random_state: int = 42) -> Dict[str, Any]:
        """Load CSV, split, train RF, evaluate, save. Mirrors main_train.py behavior for RF."""
        if not os.path.exists(data_path):
            raise FileNotFoundError(f"CSV not found: {data_path}")
        df = pd.read_csv(data_path)
        feature_cols = ['pos_x', 'pos_y', 'pos_z', 'orn_x', 'orn_y', 'orn_z', 'orn_w']
        X = df[feature_cols].values
        y = df['label'].values

        X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=test_size, random_state=random_state, stratify=y)

        self.build()
        self.model.fit(X_train, y_train)
        y_pred = self.model.predict(X_val)
        acc = accuracy_score(y_val, y_pred)
        report = classification_report(y_val, y_pred, target_names=["Fail (0)", "Success (1)"])

        self.save(model_filename)
        return {"val_accuracy": acc, "classification_report": report}
