# Robotic Grasp Learning Project

A machine learning-based robotic grasp prediction system using PyBullet physics simulation. The project implements object-oriented grasp planning, data generation, and classifier training (SVM and Random Forest) to predict grasp success with 70%+ accuracy.

## 📋 Table of Contents
- [What This Project Does](#what-this-project-does)
- [Project Structure](#project-structure)
- [Installation](#installation)
- [Quick Start](#quick-start)
- [Usage Guide](#usage-guide)
  - [Method 1: Using main.py (Recommended)](#method-1-using-mainpy-recommended)
  - [Method 2: Using Individual Modules](#method-2-using-individual-modules)
- [Features](#features)
- [Dataset Format](#dataset-format)
- [Results](#results)

---

## 🎯 What This Project Does

This project trains machine learning classifiers to predict whether a robotic grasp will succeed based on 7D pose features (position + orientation quaternion). The system:

1. **Generates training data** by simulating thousands of grasp attempts in PyBullet with:
   - Multiple objects (Cube, Duck)
   - Multiple grippers (TwoFinger PR2, ThreeFinger SDH, FrankaPanda 7-DOF)
   - Realistic noise injection (σ=0.015m position, σ=0.08rad orientation)

2. **Trains classifiers** using:
   - **SVM** with RBF kernel and grid search (71.29%±0.31% accuracy)
   - **Random Forest** with 100 trees (69.92%±0.47% accuracy)

3. **Tests models** in live simulation by:
   - Predicting grasp success probability
   - Executing predicted grasps
   - Tracking true/false positives/negatives

---

## 📁 Project Structure

```
term1-robotics-project/
├── main.py                          # Main CLI interface (USE THIS!)
├── requirements.txt                 # Python dependencies
├── README.md                        # This file
│
├── data/                            # Generated datasets
│   └── grasp_dataset.csv            # Training data (30k samples)
│
├── assets/                          # URDF files for objects/grippers
│   └── gripper_files/
│       └── threeFingers/
│           └── sdh.urdf
│
└── src/                             # Source code (OOP design)
    ├── __init__.py
    │
    ├── simulation/
    │   └── sim_manager.py           # PyBullet simulation manager
    │
    ├── hardware/
    │   ├── objects.py               # Cube, Duck (GraspableObject base)
    │   └── gripper.py               # TwoFinger, ThreeFinger, FrankaPanda
    │
    ├── planning/
    │   └── sampler.py               # Grasp pose sampling with noise
    │
    ├── learning/
    │   ├── data_generator.py        # Dataset generation
    │   ├── classifier.py            # SVM & RF classifiers
    │   ├── main_train_svm.py        # Legacy SVM training script
    │   └── main_train.py            # Legacy RF training script
    │
    └── testing/
        └── testing.py               # Live evaluation framework
```

---

## 🔧 Installation

### Prerequisites
- Python 3.8 or higher
- Windows/Linux/macOS

### Step 1: Clone the repository
```bash
git clone https://github.com/umarfj/term1-robotics-project.git
cd term1-robotics-project
```

### Step 2: Install dependencies
```bash
pip install -r requirements.txt
```

**Dependencies:**
- `numpy>=1.21.0` - Numerical computing
- `pandas>=1.3.0` - Data manipulation
- `scikit-learn>=1.0.0` - Machine learning (SVM, Random Forest)
- `pybullet>=3.2.0` - Physics simulation
- `joblib>=1.1.0` - Model serialization

---

## 🚀 Quick Start

### 1. Generate Training Data (1000 samples, ~10 minutes)
```bash
python main.py --mode generate_data --num_samples 1000
```

### 2. Train SVM Classifier
```bash
python main.py --mode train_svm --data_path data/grasp_dataset.csv
```

### 3. Test Trained Model (50 trials)
```bash
python main.py --mode test --model_name svm --num_trials 50
```

---

## 📖 Usage Guide

### Method 1: Using main.py (Recommended)

The `main.py` script provides a unified CLI interface with `argparse` for all operations.

#### **Generate Dataset**
```bash
# Generate 1000 samples (headless mode, fast)
python main.py --mode generate_data --num_samples 1000

# Generate 500 samples with GUI visualization (slower)
python main.py --mode generate_data --num_samples 500 --gui

# Custom output filename
python main.py --mode generate_data --num_samples 2000 --output_name my_dataset.csv

# With random seed for reproducibility
python main.py --mode generate_data --num_samples 1000 --seed 42
```

**Required Arguments:**
- `--mode generate_data`: Activates data generation mode

**Optional Arguments:**
- `--num_samples N`: Number of grasp attempts (default: 1000)
- `--output_name FILE`: Output CSV filename (default: grasp_dataset.csv)
- `--seed N`: Random seed for reproducibility (default: None)
- `--gui`: Show PyBullet GUI (useful for debugging)

**Output:** CSV file saved to `data/grasp_dataset.csv` with columns:
```
pos_x, pos_y, pos_z, orn_x, orn_y, orn_z, orn_w, label
```

---

#### **Train SVM Classifier**
```bash
# Train SVM with default settings
python main.py --mode train_svm --data_path data/grasp_dataset.csv

# Custom model filename
python main.py --mode train_svm --data_path data/grasp_dataset.csv --model_name my_svm.pkl

# Adjust validation split and CV folds
python main.py --mode train_svm --data_path data/grasp_dataset.csv --test_size 0.3 --cv_folds 10
```

**Required Arguments:**
- `--mode train_svm`: Activates SVM training mode

**Optional Arguments:**
- `--data_path PATH`: Training data CSV (default: data/grasp_dataset.csv)
- `--model_name FILE`: Output model filename (default: best_grasp_classifier_svm.pkl)
- `--test_size FLOAT`: Validation split ratio (default: 0.2)
- `--cv_folds N`: Cross-validation folds (default: 5)

**Training Process:**
1. Loads CSV and balances classes (removes excess negatives)
2. Splits into train/validation (80/20 by default)
3. Grid search over hyperparameters:
   - C: [0.1, 1, 10, 100]
   - gamma: ['scale', 'auto', 0.1, 1]
   - kernel: ['rbf', 'linear']
4. Evaluates best model on validation set
5. Saves trained model to `src/learning/best_grasp_classifier_svm.pkl`

**Output:** Trained model saved to `src/learning/` directory

---

#### **Train Random Forest Classifier**
```bash
# Train RF with default settings
python main.py --mode train_rf --data_path data/grasp_dataset.csv

# Custom number of trees and max depth
python main.py --mode train_rf --data_path data/grasp_dataset.csv --n_estimators 200 --max_depth 20

# Custom model filename
python main.py --mode train_rf --data_path data/grasp_dataset.csv --model_name my_rf.pkl
```

**Required Arguments:**
- `--mode train_rf`: Activates Random Forest training mode

**Optional Arguments:**
- `--data_path PATH`: Training data CSV (default: data/grasp_dataset.csv)
- `--model_name FILE`: Output model filename (default: best_grasp_classifier_rf.pkl)
- `--test_size FLOAT`: Validation split ratio (default: 0.2)
- `--n_estimators N`: Number of trees (default: 100)
- `--max_depth N`: Maximum tree depth (default: None/unlimited)

**Output:** Trained model saved to `src/learning/` directory

---

#### **Test Trained Model**
```bash
# Test SVM model (50 trials, headless)
python main.py --mode test --model_name svm --num_trials 50

# Test RF model with GUI visualization
python main.py --mode test --model_name rf --num_trials 30 --gui

# Test custom model by filename
python main.py --mode test --model_name src/learning/my_model.pkl --num_trials 100
```

**Required Arguments:**
- `--mode test`: Activates testing mode

**Optional Arguments:**
- `--model_name NAME`: Model to test ('svm', 'rf', or path to .pkl file) (default: svm)
- `--num_trials N`: Number of test attempts (default: 50)
- `--gui`: Show PyBullet GUI visualization

**Testing Process:**
1. Loads trained classifier from `src/learning/`
2. For each trial:
   - Randomly selects object (Cube/Duck) and gripper (TwoFinger/FrankaPanda)
   - Samples grasp pose with noise
   - Predicts success probability (threshold=0.7)
   - Executes grasp: approach → grasp → lift
   - Checks actual outcome (object height > 0.2m)
   - Records prediction accuracy
3. Reports metrics:
   - Overall accuracy
   - True/false positives/negatives

**Friendly Model Names:**
- `svm` → resolves to `src/learning/best_grasp_classifier_svm.pkl`
- `rf` → resolves to `src/learning/best_grasp_classifier_rf_30k.pkl`

---

### Method 2: Using Individual Modules

For advanced users who want direct access to OOP interfaces.

#### **Data Generator (data_generator.py)**

```python
from src.learning.data_generator import DataGenerator

# Initialize generator (headless mode)
generator = DataGenerator(gui_mode=False, seed=None)

# Generate 1000 samples and save to data/grasp_dataset.csv
output_path = generator.run(output_name="grasp_dataset.csv", num_samples=1000)
print(f"Dataset saved to: {output_path}")
```

**Constructor Parameters:**
- `gui_mode` (bool): Show PyBullet GUI (default: False)
- `seed` (int|None): Random seed for reproducibility (default: None)

**Methods:**
- `run(output_name, num_samples)`: Generate dataset and save to CSV
  - Returns: Path to generated CSV file

---

#### **Classifier (classifier.py)**

**Train SVM:**
```python
from src.learning.classifier import SVMClassifier

# Initialize SVM with hyperparameters
svm = SVMClassifier(probability=True, C=10.0, gamma='scale', kernel='rbf')

# Train with grid search
results = svm.train_from_csv_with_grid(
    data_path="data/grasp_dataset.csv",
    model_filename="best_grasp_classifier_svm.pkl",
    test_size=0.2,
    random_state=42,
    param_grid={
        'svc__C': [0.1, 1, 10, 100],
        'svc__gamma': ['scale', 'auto', 0.1, 1],
        'svc__kernel': ['rbf', 'linear']
    },
    cv=5,
    n_jobs=-1,
    verbose=1
)

print(f"Best params: {results['best_params']}")
print(f"Validation accuracy: {results['val_accuracy']:.4f}")
print(results['classification_report'])
```

**Train Random Forest:**
```python
from src.learning.classifier import RFClassifier

# Initialize RF
rf = RFClassifier(n_estimators=100, max_depth=None, n_jobs=-1, random_state=42)

# Train on CSV data
results = rf.train_from_csv(
    data_path="data/grasp_dataset.csv",
    model_filename="best_grasp_classifier_rf.pkl",
    test_size=0.2,
    random_state=42
)

print(f"Validation accuracy: {results['val_accuracy']:.4f}")
print(results['classification_report'])
```

**Predict with Trained Model:**
```python
import numpy as np

# Load saved model
svm = SVMClassifier()
svm.load("best_grasp_classifier_svm.pkl")

# Predict single grasp
features = np.array([[0.1, 0.05, 0.3, 0.0, 0.0, 0.707, 0.707]])  # [x,y,z,qx,qy,qz,qw]
prediction = svm.predict(features)
probabilities = svm.predict_proba(features)

print(f"Prediction: {prediction[0]}")  # 0 or 1
print(f"Success probability: {probabilities[0][1]:.2%}")
```

---

#### **Testing (testing.py)**

```python
from src.testing.testing import Test

# Initialize tester
tester = Test()

# Run 50 test trials with SVM model (GUI enabled)
tester.test_simulation(
    model_name="svm",  # or "rf", or direct path to .pkl
    num_trials=50,
    gui_mode=True
)
```

**Method Parameters:**
- `model_name` (str): Model to test ('svm', 'rf', or path to .pkl file)
- `num_trials` (int): Number of grasp attempts
- `gui_mode` (bool): Show PyBullet GUI visualization

**Output Metrics:**
- Overall accuracy
- True positives (correctly predicted success)
- True negatives (correctly predicted failure)
- False positives (predicted success but failed) ⚠️ Dangerous
- False negatives (predicted failure but succeeded) ℹ️ Missed opportunity

---

## ✨ Features

### **Object-Oriented Design**
- **Abstract base classes**: `GraspableObject`, `Gripper`, `BaseClassifier`
- **Concrete implementations**: Cube, Duck, TwoFinger, ThreeFinger, FrankaPanda, SVMClassifier, RFClassifier
- **Modular architecture**: Separate modules for simulation, hardware, planning, learning, testing

### **Realistic Simulation**
- **Physics engine**: PyBullet with gravity, friction, constraint-based grasping
- **Noise injection**: Gaussian noise on position (σ=0.015m) and orientation (σ=0.08rad ≈ 4.6°)
- **Multiple grippers**: 
  - TwoFinger (PR2): 2-finger parallel jaw (300N force)
  - ThreeFinger (SDH): 3-finger adaptive (300N force)
  - FrankaPanda: 7-DOF arm with IK (null-space optimization)

### **Machine Learning**
- **SVM**: RBF kernel, grid search over C/gamma/kernel
- **Random Forest**: 100 trees, unlimited depth
- **Preprocessing**: StandardScaler for SVM, class balancing
- **Evaluation**: Cross-validation, train/val split, classification reports

### **Comprehensive Testing**
- **Live evaluation**: Real-time grasp execution and outcome checking
- **Success criteria**: Object lifted > 0.2m height
- **Prediction threshold**: 0.7 probability for success classification
- **Visual debugging**: Color-coded approach lines (green=success, red=fail)

---

## 📊 Dataset Format

Generated CSV files have 8 columns:

| Column   | Description                          | Type  | Range       |
|----------|--------------------------------------|-------|-------------|
| `pos_x`  | Grasp X position (meters)            | float | [-0.15, 0.15] |
| `pos_y`  | Grasp Y position (meters)            | float | [-0.15, 0.15] |
| `pos_z`  | Grasp Z position (meters)            | float | [0.0, 0.5]  |
| `orn_x`  | Orientation quaternion X             | float | [-1.0, 1.0] |
| `orn_y`  | Orientation quaternion Y             | float | [-1.0, 1.0] |
| `orn_z`  | Orientation quaternion Z             | float | [-1.0, 1.0] |
| `orn_w`  | Orientation quaternion W             | float | [-1.0, 1.0] |
| `label`  | Success label (1=success, 0=failure) | int   | {0, 1}      |

**Example rows:**
```csv
pos_x,pos_y,pos_z,orn_x,orn_y,orn_z,orn_w,label
0.0234,-0.0145,0.3012,0.0123,0.0456,0.7234,0.6890,1
-0.0567,0.0890,0.2567,-0.0234,-0.0123,0.6890,0.7234,0
```

---

## 📈 Results

### Dataset Statistics
- **Total samples**: 30,000 grasp attempts
- **Balanced dataset**: 23,290 samples (50/50 success/fail)
- **Generation time**: ~8 hours (headless mode)

### Classifier Performance
| Model         | Accuracy (CV) | Accuracy (Val) | False Positive Rate |
|---------------|---------------|----------------|---------------------|
| **SVM (RBF)** | 71.29%±0.31%  | 71.29%         | ~29%                |
| **Random Forest** | N/A       | 69.92%±0.47%   | ~30%                |

### Baseline Comparison
- **FrankaPanda (no noise)**: 90% success rate
- **With noise (σ=0.015m, σ=0.08rad)**: ~50% success rate

### Key Findings
✅ **Strengths:**
- 70%+ accuracy despite noisy inputs
- Successfully learned grasp patterns
- Fast inference (<1ms per prediction)

⚠️ **Weaknesses:**
- 30% false positive rate (dangerous for real robots)
- Performance ceiling ~90% (FrankaPanda baseline)
- Limited object diversity (2 objects only)

---

## 🛠️ Troubleshooting

### "Module not found" errors
Make sure you're running commands from the **project root** directory (where `main.py` is located).

### PyBullet GUI not showing
- On Linux: Install `mesa-utils` (`sudo apt install mesa-utils`)
- On Windows: Update graphics drivers
- Try `--gui` flag or set `gui_mode=True`

### Training is slow
- Use `--gui` flag only for debugging (slows simulation 10x)
- SVM grid search takes ~10-30 minutes with 5-fold CV
- Random Forest is faster (~2-5 minutes)

### Model file not found during testing
- Friendly names 'svm'/'rf' look in `src/learning/` directory
- Check that training completed successfully
- Use absolute path if model is elsewhere

---

## 📝 Citation

This project was developed for **UCL Term 1 Robotics Coursework (2024-2025)**.

**Author:** Umar Farooq Javed  
**Repository:** [github.com/umarfj/term1-robotics-project](https://github.com/umarfj/term1-robotics-project)

---

## 📄 License

This project is for academic purposes only.
