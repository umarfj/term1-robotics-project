"""
Main entry point for the robotic grasp learning project.

Provides command-line interface for:
- Data generation: Generate training datasets with PyBullet simulation
- Classifier training: Train SVM or Random Forest models
- Model testing: Evaluate trained models in live simulation

Usage:
    python main.py --mode generate_data --num_samples 1000
    python main.py --mode train_svm --data_path data/grasp_dataset.csv
    python main.py --mode train_rf --data_path data/grasp_dataset.csv
    python main.py --mode test --model_name svm --num_trials 50
"""

import argparse
import sys
import os

# Ensure Python can see the 'src' folder
sys.path.append(os.getcwd())

from src.learning.data_generator import DataGenerator
from src.learning.classifier import SVMClassifier, RFClassifier
from src.testing.testing import Test


def generate_data(args):
    """Generate grasp dataset using PyBullet simulation."""
    print(f"\n{'='*60}")
    print(f"GENERATING GRASP DATASET")
    print(f"{'='*60}")
    print(f"Number of samples: {args.num_samples}")
    print(f"Output file: {args.output_name}")
    print(f"GUI mode: {args.gui}")
    print(f"Random seed: {args.seed if args.seed else 'None (non-deterministic)'}")
    print(f"{'='*60}\n")
    
    generator = DataGenerator(gui_mode=args.gui, seed=args.seed)
    output_path = generator.run(output_name=args.output_name, num_samples=args.num_samples)
    
    print(f"\n{'='*60}")
    print(f"✓ Dataset generation complete!")
    print(f"  Saved to: {output_path}")
    print(f"{'='*60}\n")


def train_svm(args):
    """Train SVM classifier with grid search."""
    print(f"\n{'='*60}")
    print(f"TRAINING SVM CLASSIFIER")
    print(f"{'='*60}")
    print(f"Data path: {args.data_path}")
    print(f"Output model: {args.model_name}")
    print(f"Test size: {args.test_size}")
    print(f"CV folds: {args.cv_folds}")
    print(f"{'='*60}\n")
    
    classifier = SVMClassifier(probability=True)
    
    # Default parameter grid for SVM
    param_grid = {
        'svc__C': [0.1, 1, 10, 100],
        'svc__gamma': ['scale', 'auto', 0.1, 1],
        'svc__kernel': ['rbf', 'linear']
    }
    
    results = classifier.train_from_csv_with_grid(
        data_path=args.data_path,
        model_filename=args.model_name,
        test_size=args.test_size,
        random_state=42,
        param_grid=param_grid,
        cv=args.cv_folds,
        n_jobs=-1,
        verbose=2
    )
    
    print(f"\n{'='*60}")
    print(f"✓ SVM Training Complete!")
    print(f"  Best parameters: {results['best_params']}")
    print(f"  CV accuracy: {results['cv_score']:.4f}")
    print(f"  Validation accuracy: {results['val_accuracy']:.4f}")
    print(f"  Model saved to: src/learning/{args.model_name}")
    print(f"{'='*60}\n")
    print("Classification Report:")
    print(results['classification_report'])


def train_rf(args):
    """Train Random Forest classifier."""
    print(f"\n{'='*60}")
    print(f"TRAINING RANDOM FOREST CLASSIFIER")
    print(f"{'='*60}")
    print(f"Data path: {args.data_path}")
    print(f"Output model: {args.model_name}")
    print(f"Test size: {args.test_size}")
    print(f"N estimators: {args.n_estimators}")
    print(f"Max depth: {args.max_depth if args.max_depth else 'None (unlimited)'}")
    print(f"{'='*60}\n")
    
    classifier = RFClassifier(
        n_estimators=args.n_estimators,
        max_depth=args.max_depth,
        n_jobs=-1,
        random_state=42
    )
    
    results = classifier.train_from_csv(
        data_path=args.data_path,
        model_filename=args.model_name,
        test_size=args.test_size,
        random_state=42
    )
    
    print(f"\n{'='*60}")
    print(f"✓ Random Forest Training Complete!")
    print(f"  Validation accuracy: {results['val_accuracy']:.4f}")
    print(f"  Model saved to: src/learning/{args.model_name}")
    print(f"{'='*60}\n")
    print("Classification Report:")
    print(results['classification_report'])


def test_model(args):
    """Test trained classifier in live simulation."""
    print(f"\n{'='*60}")
    print(f"TESTING CLASSIFIER IN LIVE SIMULATION")
    print(f"{'='*60}")
    print(f"Model: {args.model_name}")
    print(f"Number of trials: {args.num_trials}")
    print(f"GUI mode: {args.gui}")
    print(f"{'='*60}\n")
    
    tester = Test()
    tester.test_simulation(
        model_name=args.model_name,
        num_trials=args.num_trials,
        gui_mode=args.gui
    )


def main():
    """Parse arguments and execute requested mode."""
    parser = argparse.ArgumentParser(
        description="Robotic Grasp Learning Project - Main Interface",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Generate 1000 training samples
  python main.py --mode generate_data --num_samples 1000
  
  # Generate data with GUI visualization
  python main.py --mode generate_data --num_samples 500 --gui
  
  # Train SVM classifier
  python main.py --mode train_svm --data_path data/grasp_dataset.csv
  
  # Train Random Forest classifier
  python main.py --mode train_rf --data_path data/grasp_dataset.csv --n_estimators 200
  
  # Test SVM model with 50 trials
  python main.py --mode test --model_name svm --num_trials 50
  
  # Test RF model with GUI visualization
  python main.py --mode test --model_name rf --num_trials 30 --gui
        """
    )
    
    parser.add_argument(
        '--mode',
        type=str,
        required=True,
        choices=['generate_data', 'train_svm', 'train_rf', 'test'],
        help='Execution mode: generate_data, train_svm, train_rf, or test'
    )
    
    # Data generation arguments
    parser.add_argument(
        '--num_samples',
        type=int,
        default=1000,
        help='Number of grasp samples to generate (default: 1000)'
    )
    
    parser.add_argument(
        '--output_name',
        type=str,
        default='grasp_dataset.csv',
        help='Output CSV filename for generated data (default: grasp_dataset.csv)'
    )
    
    parser.add_argument(
        '--seed',
        type=int,
        default=None,
        help='Random seed for reproducibility (default: None)'
    )
    
    # Training arguments
    parser.add_argument(
        '--data_path',
        type=str,
        default='data/grasp_dataset.csv',
        help='Path to training data CSV (default: data/grasp_dataset.csv)'
    )
    
    parser.add_argument(
        '--model_name',
        type=str,
        default=None,
        help='Model filename or friendly name (svm, rf) for saving/loading'
    )
    
    parser.add_argument(
        '--test_size',
        type=float,
        default=0.2,
        help='Validation split ratio (default: 0.2)'
    )
    
    parser.add_argument(
        '--cv_folds',
        type=int,
        default=5,
        help='Number of cross-validation folds for SVM (default: 5)'
    )
    
    parser.add_argument(
        '--n_estimators',
        type=int,
        default=100,
        help='Number of trees for Random Forest (default: 100)'
    )
    
    parser.add_argument(
        '--max_depth',
        type=int,
        default=None,
        help='Max depth for Random Forest trees (default: None/unlimited)'
    )
    
    # Testing arguments
    parser.add_argument(
        '--num_trials',
        type=int,
        default=50,
        help='Number of test trials for evaluation (default: 50)'
    )
    
    # Common arguments
    parser.add_argument(
        '--gui',
        action='store_true',
        help='Enable PyBullet GUI visualization'
    )
    
    args = parser.parse_args()
    
    # Set default model names based on mode
    if args.mode == 'train_svm' and args.model_name is None:
        args.model_name = 'best_grasp_classifier_svm.pkl'
    elif args.mode == 'train_rf' and args.model_name is None:
        args.model_name = 'best_grasp_classifier_rf.pkl'
    elif args.mode == 'test' and args.model_name is None:
        args.model_name = 'svm'  # Default to SVM for testing
    
    # Execute requested mode
    try:
        if args.mode == 'generate_data':
            generate_data(args)
        elif args.mode == 'train_svm':
            train_svm(args)
        elif args.mode == 'train_rf':
            train_rf(args)
        elif args.mode == 'test':
            test_model(args)
    except Exception as e:
        print(f"\n[ERROR] {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
