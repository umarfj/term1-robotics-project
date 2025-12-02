import pybullet as p
import pandas as pd
import numpy as np
import time
import os
import sys

sys.path.append(os.getcwd())

from src.simulation.sim_manager import SimulationManager
from src.hardware.objects import Cube, Duck
from src.hardware.gripper import TwoFinger, ThreeFinger
from src.planning.sampler import GraspSampler

# --- CONFIGURATION ---
NUM_SAMPLES = 10        # 10 for dry-run
GUI_MODE = True
DATA_DIR = "data"
FILENAME = "grasp_dataset.csv"

def interpolate_pose(start_pos, end_pos, steps=50):
    path = []
    for t in np.linspace(0, 1, steps):
        pos = (1 - t) * np.array(start_pos) + t * np.array(end_pos)
        path.append(list(pos))
    return path

def main():
    if not os.path.exists(DATA_DIR):
        os.makedirs(DATA_DIR)
        
    sim = SimulationManager(gui_mode=GUI_MODE)
    sampler = GraspSampler()
    dataset = [] 
    
    print(f"Starting Data Generation: {NUM_SAMPLES} samples...")

    for i in range(NUM_SAMPLES):
        print(f"\n--- Sample {i+1}/{NUM_SAMPLES} ---")
        
        if np.random.rand() > 0.5:
            current_obj = Cube(start_pos=[0, 0, 0.1])
            obj_name = "Cube"
        else:
            current_obj = Duck(start_pos=[0, 0, 0.1])
            obj_name = "Duck"
            
        # if np.random.rand() > 0.5:
        #     current_gripper = TwoFinger(sim.client_id)
        #     grip_name = "TwoFinger"
        # else:
        #     current_gripper = ThreeFinger(sim.client_id)
        #     grip_name = "ThreeFinger"
        current_gripper = TwoFinger(sim.client_id)
        grip_name = "TwoFinger"
        print(f"Target: {obj_name} | Gripper: {grip_name}")

        target_pos = current_obj.get_position()
        grasp_pos, grasp_orn = sampler.sample_grasp_pose(target_pos, radius=0.10)
        approach_pos, _ = sampler.calculate_approach_pose(grasp_pos, grasp_orn, distance=0.1)

        # EXECUTION
        current_gripper.open()
        current_gripper.move_to(approach_pos, grasp_orn)
        sim.run_simulation(0.5)
        
        # Approach
        path_points = interpolate_pose(approach_pos, grasp_pos, steps=20)
        for point in path_points:
            current_gripper.move_to(point, grasp_orn)
            sim.step()
            
        # Grasp
        current_gripper.close()
        sim.run_simulation(1.0)
        
        # Lift (Slower & Continuous Grip)
        lift_pos = [grasp_pos[0], grasp_pos[1], grasp_pos[2] + 0.3]
        lift_path = interpolate_pose(grasp_pos, lift_pos, steps=100) # Slower steps
        
        for point in lift_path:
            current_gripper.move_to(point, grasp_orn)
            current_gripper.close() # Re-apply grip force!
            sim.step()
            
        # Hold
        sim.run_simulation(3.0)
        
        is_success = sim.check_success(current_obj)
        label = 1 if is_success else 0
        print(f"Result: {'SUCCESS' if is_success else 'FAIL'}")
        
        data_row = {
            "pos_x": grasp_pos[0], "pos_y": grasp_pos[1], "pos_z": grasp_pos[2],
            "orn_x": grasp_orn[0], "orn_y": grasp_orn[1], "orn_z": grasp_orn[2], "orn_w": grasp_orn[3],
            "gripper_type": grip_name, "object_type": obj_name, "label": label
        }
        dataset.append(data_row)
        
        current_gripper.cleanup()
        current_obj.cleanup()
        sim.setup()
        
    df = pd.DataFrame(dataset)
    full_path = os.path.join(DATA_DIR, FILENAME)
    df.to_csv(full_path, index=False)
    
    print("\nData Generation Complete!")
    print(f"Total Samples: {len(df)}")
    print(f"Success Rate: {df['label'].mean():.2%}")
    
    sim.cleanup()

if __name__ == "__main__":
    main()