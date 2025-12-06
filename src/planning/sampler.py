import numpy as np
import pybullet as p

class GraspSampler:
    def __init__(self, rng_seed=None):
        if rng_seed is not None:
            np.random.seed(rng_seed)

    def sample_grasp_pose(self, target_pos, radius):
        # 1. Calculate the Perfect (Ideal) Geometry
        # Restrict to upper hemisphere
        phi = np.random.uniform(0, 2 * np.pi)
        
        # FIX: Restrict phi for Franka Panda if needed (as discussed before)
        # phi = np.random.uniform(-np.pi/2 + 0.2, np.pi/2 - 0.2) 

        theta = np.random.uniform(0, np.pi / 3) 

        x_offset = radius * np.sin(theta) * np.cos(phi)
        y_offset = radius * np.sin(theta) * np.sin(phi)
        z_offset = radius * np.cos(theta) 

        # The "Perfect" Position
        ideal_pos = [
            target_pos[0] + x_offset,
            target_pos[1] + y_offset,
            target_pos[2] + z_offset 
        ]

        # Calculate "Perfect" Orientation (Looking at object)
        direction = np.array(target_pos) - np.array(ideal_pos)
        direction = direction / np.linalg.norm(direction)
        
        yaw = np.arctan2(direction[1], direction[0])
        xy_dist = np.sqrt(direction[0]**2 + direction[1]**2)
        pitch = np.arctan2(xy_dist, direction[2])
        roll = 0.0
        
        # ---------------------------------------------------------
        # 2. ADD NOISE (THE JITTER)
        # ---------------------------------------------------------
        
        # Position Noise: Gaussian noise with Standard Deviation of 1.5cm (0.015)
        # This means most grasps will be within +/- 3cm of the target
        pos_noise = np.random.normal(0, 0.015, 3) 
        noisy_grasp_pos = [
            ideal_pos[0] + pos_noise[0],
            ideal_pos[1] + pos_noise[1],
            ideal_pos[2] + pos_noise[2]
        ]

        # Orientation Noise: +/- 5 degrees (approx 0.08 radians)
        r_noise = np.random.normal(0, 0.08)
        p_noise = np.random.normal(0, 0.08)
        y_noise = np.random.normal(0, 0.08)

        # Apply noise to the Euler angles
        noisy_yaw = yaw + y_noise
        noisy_pitch = pitch + p_noise
        noisy_roll = roll + r_noise

        # Convert the NOISY angles to Quaternion
        # Note: We apply roll noise here, effectively rotating around the approach vector randomly
        base_orn = p.getQuaternionFromEuler([0, noisy_pitch, noisy_yaw])
        grasp_orn = p.multiplyTransforms([0,0,0], base_orn, [0,0,0], p.getQuaternionFromEuler([0, 0, noisy_roll]))[1]

        # Return the NOISY coordinates. 
        # main.py will record these and the robot will try to execute them.
        return noisy_grasp_pos, grasp_orn

    def calculate_approach_pose(self, grasp_pos, grasp_orn, distance=0.1):
        rot_matrix = p.getMatrixFromQuaternion(grasp_orn)
        z_axis = np.array([rot_matrix[2], rot_matrix[5], rot_matrix[8]])
        approach_pos = np.array(grasp_pos) - (z_axis * distance)
        return list(approach_pos), grasp_orn