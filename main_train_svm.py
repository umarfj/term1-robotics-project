import pandas as pd
import numpy as np
import joblib
import sys
import os

from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, classification_report

# --- CONFIGURATION ---
DATA_PATH = "data/grasp_dataset.csv" 
MODEL_SAVE_PATH = "best_grasp_classifier_svm.pkl"
RANDOM_STATE = 42

def load_and_balance_data(filepath):
    """
    Loads the CSV and balances the dataset by downsampling the majority class.
    """
    if not os.path.exists(filepath):
        print(f"Error: File '{filepath}' not found.")
        sys.exit(1)
        
    df = pd.read_csv(filepath)
    print(f"Total samples loaded: {len(df)}")
    print(f"Initial Distribution:\n{df['label'].value_counts()}")

    success_df = df[df['label'] == 1]
    fail_df = df[df['label'] == 0]

    min_count = min(len(success_df), len(fail_df))

    success_balanced = success_df.sample(n=min_count, random_state=RANDOM_STATE)
    fail_balanced = fail_df.sample(n=min_count, random_state=RANDOM_STATE)

    balanced_df = pd.concat([success_balanced, fail_balanced])
    balanced_df = balanced_df.sample(frac=1, random_state=RANDOM_STATE).reset_index(drop=True)
    
    print(f"\nBalanced Distribution:\n{balanced_df['label'].value_counts()}")
    return balanced_df

def main():
    # 1. Load and Balance
    df = load_and_balance_data(DATA_PATH)

    # 2. Prepare Features
    feature_cols = ['pos_x', 'pos_y', 'pos_z', 'orn_x', 'orn_y', 'orn_z', 'orn_w']
    X = df[feature_cols].values
    y = df['label'].values

    # 3. Split Data
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=RANDOM_STATE)

    # 4. Create a Pipeline (Scaler + SVM)
    # SVMs work poorly on raw data, so we MUST scale it first.
    pipeline = Pipeline([
        ('scaler', StandardScaler()),
        ('svc', SVC(probability=True, random_state=RANDOM_STATE)) 
    ])

    # 5. Define Hyperparameter Grid
    # C: Controls how strict the margin is (High C = stricter, Low C = smoother)
    # Gamma: Controls how "curved" the decision boundary is
    param_grid = {
        'svc__C': [0.1, 1, 10, 100],
        'svc__gamma': ['scale', 'auto', 0.1, 1],
        'svc__kernel': ['rbf', 'linear'] # RBF is usually better for physics
    }

    print("\n" + "="*40)
    print("STARTING SVM TUNING (GridSearchCV)")
    print("Testing different Kernel and C values...")
    print("="*40)

    # 6. Run Grid Search
    grid_search = GridSearchCV(
        estimator=pipeline, 
        param_grid=param_grid, 
        cv=5, 
        n_jobs=-1, 
        verbose=1,
        scoring='accuracy'
    )

    grid_search.fit(X_train, y_train)

    # 7. Report Results
    best_model = grid_search.best_estimator_
    print(f"\nBest Parameters: {grid_search.best_params_}")
    print(f"Best Cross-Validation Score: {grid_search.best_score_:.2%}")

    # 8. Final Validation
    y_pred = best_model.predict(X_val)
    final_accuracy = accuracy_score(y_val, y_pred)

    print("\n" + "="*40)
    print(f"FINAL SVM ACCURACY: {final_accuracy:.2%}")
    print("="*40)
    print("\nClassification Report:")
    print(classification_report(y_val, y_pred, target_names=["Fail (0)", "Success (1)"]))

    # Note: SVM (RBF kernel) does not provide simple Feature Importances like Random Forest.

    # 9. Save
    joblib.dump(best_model, MODEL_SAVE_PATH)
    print(f"\n>> Best SVM model saved to '{MODEL_SAVE_PATH}'")
    print(">> You can now run 'test_model_performance.py' (update the MODEL_PATH variable inside it first!)")

if __name__ == "__main__":
    main()