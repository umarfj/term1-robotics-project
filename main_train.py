import pandas as pd
import numpy as np
import joblib
import sys
import os
import time

from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report

# --- CONFIGURATION ---
DATA_PATH = "data/grasp_dataset.csv" 
MODEL_SAVE_PATH = "best_grasp_classifier_rf_30k.pkl"
RANDOM_STATE = 42

def load_and_balance_data(filepath):
    # (Same balancing logic as before)
    if not os.path.exists(filepath):
        print(f"Error: File '{filepath}' not found.")
        sys.exit(1)
        
    df = pd.read_csv(filepath)
    print(f"Total samples loaded: {len(df)}")
    
    success_df = df[df['label'] == 1]
    fail_df = df[df['label'] == 0]
    min_count = min(len(success_df), len(fail_df))

    success_balanced = success_df.sample(n=min_count, random_state=RANDOM_STATE)
    fail_balanced = fail_df.sample(n=min_count, random_state=RANDOM_STATE)

    balanced_df = pd.concat([success_balanced, fail_balanced])
    balanced_df = balanced_df.sample(frac=1, random_state=RANDOM_STATE).reset_index(drop=True)
    
    print(f"Balanced Distribution:\n{balanced_df['label'].value_counts()}")
    return balanced_df

def main():
    start_time = time.time()
    
    # 1. Load Data
    df = load_and_balance_data(DATA_PATH)
    
    # 2. Prepare Features
    feature_cols = ['pos_x', 'pos_y', 'pos_z', 'orn_x', 'orn_y', 'orn_z', 'orn_w']
    X = df[feature_cols].values
    y = df['label'].values

    # 3. Split
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=RANDOM_STATE)

    # 4. Hyperparameter Grid
    # We test deeper trees because we have more data now
    param_grid = {
        'n_estimators': [100, 300], 
        'max_depth': [None, 20, 30],       # Deeper trees for 30k samples
        'min_samples_split': [2, 5, 10],
        'max_features': ['sqrt', None]     # 'None' means use all 7 features every split
    }

    print("\n" + "="*40)
    print("STARTING RANDOM FOREST TUNING (30k Samples)")
    print("="*40)

    # 5. Run Grid Search
    rf = RandomForestClassifier(random_state=RANDOM_STATE)
    grid_search = GridSearchCV(
        estimator=rf, 
        param_grid=param_grid, 
        cv=3,                 # 3-fold is enough for this large dataset
        n_jobs=-1,            # Use all CPU cores
        verbose=1,
        scoring='accuracy'
    )

    grid_search.fit(X_train, y_train)

    # 6. Report
    best_clf = grid_search.best_estimator_
    print(f"\nBest Parameters: {grid_search.best_params_}")
    print(f"Best CV Score: {grid_search.best_score_:.2%}")

    # 7. Final Validation
    y_pred = best_clf.predict(X_val)
    final_accuracy = accuracy_score(y_val, y_pred)

    print("\n" + "="*40)
    print(f"FINAL RF VALIDATION ACCURACY: {final_accuracy:.2%}")
    print("="*40)
    
    # 8. Feature Importance (Bonus for report!)
    print("\nFeature Importances:")
    importances = best_clf.feature_importances_
    sorted_indices = np.argsort(importances)[::-1]
    for i in sorted_indices:
        print(f"{feature_cols[i]}: {importances[i]:.4f}")

    joblib.dump(best_clf, MODEL_SAVE_PATH)
    print(f"\nTime taken: {(time.time() - start_time)/60:.1f} minutes")

if __name__ == "__main__":
    main()