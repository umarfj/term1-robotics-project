import pandas as pd
import numpy as np
import joblib
import sys
import os

from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report

# --- CONFIGURATION ---
DATA_PATH = "data/grasp_dataset.csv"  # Ensure this matches your file location
MODEL_SAVE_PATH = "best_grasp_classifier.pkl"
RANDOM_STATE = 42  # Ensures results are consistent every time you run it

def load_and_balance_data(filepath):
    """
    Loads the CSV and balances the dataset by downsampling the majority class.
    Returns a balanced DataFrame.
    """
    if not os.path.exists(filepath):
        print(f"Error: File '{filepath}' not found.")
        sys.exit(1)
        
    df = pd.read_csv(filepath)
    print(f"Total samples loaded: {len(df)}")
    print(f"Initial Distribution:\n{df['label'].value_counts()}")

    # Separate classes
    success_df = df[df['label'] == 1]
    fail_df = df[df['label'] == 0]

    # Find the count of the smaller class
    min_count = min(len(success_df), len(fail_df))

    # Downsample both to match the minority count
    success_balanced = success_df.sample(n=min_count, random_state=RANDOM_STATE)
    fail_balanced = fail_df.sample(n=min_count, random_state=RANDOM_STATE)

    # Combine and shuffle
    balanced_df = pd.concat([success_balanced, fail_balanced])
    balanced_df = balanced_df.sample(frac=1, random_state=RANDOM_STATE).reset_index(drop=True)
    
    print(f"\nBalanced Distribution:\n{balanced_df['label'].value_counts()}")
    return balanced_df

def main():
    # 1. Load and Balance
    df = load_and_balance_data(DATA_PATH)

    # 2. Prepare Features (X) and Labels (y)
    # Using 7 features: Position (x,y,z) + Orientation (qx,qy,qz,qw)
    feature_cols = ['pos_x', 'pos_y', 'pos_z', 'orn_x', 'orn_y', 'orn_z', 'orn_w']
    X = df[feature_cols].values
    y = df['label'].values

    # 3. Split Data (80% Train, 20% Validation)
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=RANDOM_STATE)

    # 4. Define the Hyperparameter Grid (The "Tournament" Bracket)
    # We will test all combinations of these settings
    param_grid = {
        'n_estimators': [100, 200, 300],        # Number of trees
        'max_depth': [None, 10, 20],            # Maximum depth of tree
        'min_samples_split': [2, 5, 10],        # Minimum samples to split a node
        'max_features': ['sqrt', None]          # Features to consider at each split
    }

    print("\n" + "="*40)
    print("STARTING HYPERPARAMETER TUNING (GridSearchCV)")
    print("Testing different Random Forest settings...")
    print("="*40)

    # 5. Initialize Grid Search
    # n_jobs=-1 uses all CPU cores to make it faster
    rf = RandomForestClassifier(random_state=RANDOM_STATE)
    grid_search = GridSearchCV(
        estimator=rf, 
        param_grid=param_grid, 
        cv=5,               # 5-fold Cross-Validation (robust checking)
        n_jobs=-1, 
        verbose=1,
        scoring='accuracy'
    )

    # 6. Run the Search (Train)
    grid_search.fit(X_train, y_train)

    # 7. Report Best Results
    best_clf = grid_search.best_estimator_
    print(f"\nBest Parameters Found: {grid_search.best_params_}")
    print(f"Best Cross-Validation Score: {grid_search.best_score_:.2%}")

    # 8. Final Validation on the Test Set (The unseen 20%)
    y_pred = best_clf.predict(X_val)
    final_accuracy = accuracy_score(y_val, y_pred)

    print("\n" + "="*40)
    print(f"FINAL VALIDATION ACCURACY: {final_accuracy:.2%}")
    print("="*40)
    print("\nClassification Report:")
    print(classification_report(y_val, y_pred, target_names=["Fail (0)", "Success (1)"]))

    # 9. Feature Importance (Why did it decide that?)
    print("\nFeature Importances (What mattered most?):")
    importances = best_clf.feature_importances_
    sorted_indices = np.argsort(importances)[::-1] # Sort high to low
    
    for i in sorted_indices:
        print(f"{feature_cols[i]}: {importances[i]:.4f}")

    # 10. Save the Winner
    joblib.dump(best_clf, MODEL_SAVE_PATH)
    print(f"\n>> Best model saved to '{MODEL_SAVE_PATH}'")
    print(">> You can now run your test_robot.py script.")

if __name__ == "__main__":
    main()