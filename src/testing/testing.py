"""
OOP version of main_test.py.

Class Test provides method test_simulation(model_name, num_trials, gui_mode)
and preserves the original testing logic (sampling, prediction, execution, scoring).
"""

import os
import numpy as np
import joblib
import pybullet as p

from src.simulation.sim_manager import SimulationManager
from src.hardware.objects import Cube, Duck
from src.hardware.gripper import TwoFinger, FrankaPanda
from src.planning.sampler import GraspSampler


class Test:
    def __init__(self):
        pass

    def _resolve_model_path(self, model_name: str) -> str:
        """Resolve a friendly model name or direct path.

        Preference order:
        1) src/learning/<expected_file>.pkl
        2) workspace root <expected_file>.pkl
        3) treat model_name as a direct path
        """
        name = (model_name or "").lower()
        root = os.getcwd()
        src_learning = os.path.join(root, "src", "learning")

        if name in ("rf", "random_forest"):
            candidate = os.path.join(src_learning, "best_grasp_classifier_rf_30k.pkl")
            if os.path.exists(candidate):
                return candidate
            fallback = os.path.join(root, "best_grasp_classifier_rf_30k.pkl")
            return fallback

        if name in ("svm", "support_vector_machine"):
            candidate = os.path.join(src_learning, "best_grasp_classifier_svm.pkl")
            if os.path.exists(candidate):
                return candidate
            fallback = os.path.join(root, "best_grasp_classifier_svm.pkl")
            return fallback

        # If a direct filepath was provided
        return model_name

    def test_simulation(self, model_name: str, num_trials: int, gui_mode: bool):
        model_path = self._resolve_model_path(model_name)
        if not os.path.exists(model_path):
            print(f"Error: Model file '{model_path}' not found.")
            return

        clf = joblib.load(model_path)
        print(f">> Model loaded: {model_path}")

        # Setup simulation and sampler
        sim = SimulationManager(gui_mode=gui_mode)
        sampler = GraspSampler()

        print(f"\nStarting Live Evaluation: {num_trials} attempts...")
        correct_preds = 0
        true_positives = 0
        false_positives = 0
        true_negatives = 0
        false_negatives = 0

        for i in range(num_trials):
            print(f"\n--- Test Attempt {i+1}/{num_trials} ---")

            # Random object
            if np.random.rand() > 0.5:
                current_obj = Cube(start_pos=[0, 0, 0])
                obj_name = "Cube"
            else:
                current_obj = Duck(start_pos=[0, 0, 0])
                obj_name = "Duck"

            sim.run_simulation(1.0)

            # Random gripper (TwoFinger or FrankaPanda)
            if np.random.rand() > 0.5:
                current_gripper = TwoFinger(sim.client_id)
                grip_name = "TwoFinger"
            else:
                current_gripper = FrankaPanda(sim.client_id)
                grip_name = "FrankaPanda"

            target_pos = current_obj.get_position()

            # Candidate grasp with noise
            grasp_pos, grasp_orn = sampler.sample_grasp_pose(target_pos, radius=0.10)
            approach_pos, _ = sampler.calculate_approach_pose(grasp_pos, grasp_orn, distance=0.1)

            # Predict
            features = [
                grasp_pos[0], grasp_pos[1], grasp_pos[2],
                grasp_orn[0], grasp_orn[1], grasp_orn[2], grasp_orn[3]
            ]
            probs = clf.predict_proba([features])[0]
            success_probability = probs[1]
            prediction = 1 if success_probability > 0.7 else 0
            pred_label = "SUCCESS" if prediction == 1 else "FAIL"
            confidence = probs[prediction] * 100
            print(f"Model Prediction: {pred_label} ({confidence:.1f}% confidence)")

            # Execute & visualize
            line_color = [0, 1, 0] if prediction == 1 else [1, 0, 0]
            p.addUserDebugLine(approach_pos, grasp_pos, line_color, lineWidth=4, lifeTime=5)

            current_gripper.open()
            current_gripper.move_to(approach_pos, grasp_orn)
            sim.run_simulation(0.5)

            current_gripper.move_to(grasp_pos, grasp_orn)
            sim.step()

            current_gripper.close()
            sim.run_simulation(1.0)

            lift_pos = [grasp_pos[0], grasp_pos[1], grasp_pos[2] + 0.3]
            current_gripper.move_to(lift_pos, grasp_orn)

            sim.run_simulation(2.0)
            is_success = sim.check_success(current_obj)
            actual_label = "SUCCESS" if is_success else "FAIL"
            print(f"Actual Outcome:   {actual_label}")

            # Score
            if prediction == int(is_success):
                print(">> Result: CORRECT")
                correct_preds += 1
                if is_success:
                    true_positives += 1
                else:
                    true_negatives += 1
            else:
                print(">> Result: WRONG")
                if is_success:
                    false_negatives += 1
                else:
                    false_positives += 1

            # Cleanup
            current_gripper.cleanup()
            current_obj.cleanup()
            sim.setup()

        # Final report
        print("\n" + "="*40)
        print("FINAL TEST RESULTS")
        print("="*40)
        print(f"Total Attempts:     {num_trials}")
        print(f"Correct Predictions: {correct_preds}")
        print(f"Overall Accuracy:    {correct_preds / num_trials:.2%}")
        print("-" * 20)
        print(f"True Positives (Predicted Success & Was Success): {true_positives}")
        print(f"True Negatives (Predicted Fail & Was Fail):       {true_negatives}")
        print(f"False Positives (Predicted Success & Failed):     {false_positives} (Dangerous)")
        print(f"False Negatives (Predicted Fail & Succeeded):     {false_negatives} (Missed Opportunity)")
        print("="*40)

        sim.cleanup()
