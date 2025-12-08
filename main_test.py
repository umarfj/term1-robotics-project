import pybullet as p
import pandas as pd
import numpy as np
import joblib
import os
import sys
import time

# Ensure python looks in the current directory for modules
sys.path.append(os.getcwd())

from src.simulation.sim_manager import SimulationManager
from src.hardware.objects import Cube, Duck
from src.hardware.gripper import TwoFinger, FrankaPanda, ThreeFinger
from src.planning.sampler import GraspSampler

# --- CONFIGURATION ---
MODEL_PATH = "best_grasp_classifier_rf_30k.pkl" # The file you just trained
NUM_TESTS = 500  # Coursework requires "at least ten" 
GUI_MODE = False # Must be True to see the robot move

def main():
    # 1. Load the Trained Model
    if not os.path.exists(MODEL_PATH):
        print(f"Error: Model file '{MODEL_PATH}' not found.")
        print("Run 'train_optimized_model.py' first.")
        return
    
    clf = joblib.load(MODEL_PATH)
    print(f">> Model loaded: {MODEL_PATH}")

    # 2. Setup Simulation
    sim = SimulationManager(gui_mode=GUI_MODE)
    sampler = GraspSampler() # This now includes your noise logic
    
    print(f"\nStarting Live Evaluation: {NUM_TESTS} attempts...")
    
    correct_preds = 0
    true_positives = 0
    false_positives = 0
    true_negatives = 0
    false_negatives = 0

    for i in range(NUM_TESTS):
        print(f"\n--- Test Attempt {i+1}/{NUM_TESTS} ---")
        
        # Random Setup
        if np.random.rand() > 0.5:
            current_obj = Cube(start_pos=[0, 0, 0])
            obj_name = "Cube"
        else:
            current_obj = Duck(start_pos=[0, 0, 0])
            obj_name = "Duck"
            
        sim.run_simulation(1.0) # Let object settle

        if np.random.rand() > 0.5:
            current_gripper = TwoFinger(sim.client_id)
            grip_name = "TwoFinger"
        else:
            current_gripper = FrankaPanda(sim.client_id)
            grip_name = "FrankaPanda"
            
        target_pos = current_obj.get_position()
        
        # 3. GENERATE CANDIDATE (With Noise)
        grasp_pos, grasp_orn = sampler.sample_grasp_pose(target_pos, radius=0.10)
        approach_pos, _ = sampler.calculate_approach_pose(grasp_pos, grasp_orn, distance=0.1)

        # 4. PREDICT (Ask the AI)
        # We must format the input exactly like training: [x, y, z, qx, qy, qz, qw]
        features = [
            grasp_pos[0], grasp_pos[1], grasp_pos[2],
            grasp_orn[0], grasp_orn[1], grasp_orn[2], grasp_orn[3]
        ]
        
        # Reshape to 2D array (1 row)
        probs = clf.predict_proba([features])[0] # Returns [prob_fail, prob_success]
        success_probability = probs[1] # The probability of class 1

        # Only predict Success if the model is VERY confident (> 70%)
        if success_probability > 0.75:
            prediction = 1
        else:
            prediction = 0        
        pred_label = "SUCCESS" if prediction == 1 else "FAIL"
        confidence = probs[prediction] * 100
        
        print(f"Model Prediction: {pred_label} ({confidence:.1f}% confidence)")

        # 5. EXECUTE (Verify Physical Truth)
        # Visual Debug: Draw the target grasp (Green=Pred Success, Red=Pred Fail)
        line_color = [0, 1, 0] if prediction == 1 else [1, 0, 0]
        p.addUserDebugLine(approach_pos, grasp_pos, line_color, lineWidth=4, lifeTime=5)

        # Move to Approach
        current_gripper.open()
        current_gripper.move_to(approach_pos, grasp_orn)
        sim.run_simulation(0.5)
        
        # Move to Grasp
        current_gripper.move_to(grasp_pos, grasp_orn)
        sim.step()
        
        # Close Fingers
        current_gripper.close()
        sim.run_simulation(1.0)
        
        # Lift Object
        lift_pos = [grasp_pos[0], grasp_pos[1], grasp_pos[2] + 0.3]
        current_gripper.move_to(lift_pos, grasp_orn)
        
        # Hold and Check
        sim.run_simulation(2.0)
        is_success = sim.check_success(current_obj)
        actual_label = "SUCCESS" if is_success else "FAIL"
        
        print(f"Actual Outcome:   {actual_label}")
        
        # 6. SCORE
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
                false_negatives += 1 # Model said fail, but it worked
            else:
                false_positives += 1 # Model said success, but it failed

        # Cleanup
        current_gripper.cleanup()
        current_obj.cleanup()
        sim.setup()

    # --- FINAL REPORT ---
    print("\n" + "="*40)
    print("FINAL TEST RESULTS")
    print("="*40)
    print(f"Total Attempts:     {NUM_TESTS}")
    print(f"Correct Predictions: {correct_preds}")
    print(f"Overall Accuracy:    {correct_preds / NUM_TESTS:.2%}")
    print("-" * 20)
    print(f"True Positives (Predicted Success & Was Success): {true_positives}")
    print(f"True Negatives (Predicted Fail & Was Fail):       {true_negatives}")
    print(f"False Positives (Predicted Success & Failed):     {false_positives} (Dangerous)")
    print(f"False Negatives (Predicted Fail & Succeeded):     {false_negatives} (Missed Opportunity)")
    print("="*40)
    
    sim.cleanup()

if __name__ == "__main__":
    main()