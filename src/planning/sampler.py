import numpy as np
import pybullet as p

class GraspSampler:
    def __init__(self, rng_seed=None):
        if rng_seed is not None:
            np.random.seed(rng_seed)

    def sample_grasp_pose(self, target_pos, radius):
        phi = np.random.uniform(0, 2 * np.pi)
        theta = np.random.uniform(0, np.pi / 2) 

        x_offset = radius * np.sin(theta) * np.cos(phi)
        y_offset = radius * np.sin(theta) * np.sin(phi)
        z_offset = radius * np.cos(theta) 

        grasp_pos = [
            target_pos[0] + x_offset,
            target_pos[1] + y_offset,
            target_pos[2] + z_offset # Slightly below to account for gripper approach
        ]

        direction = np.array(target_pos) - np.array(grasp_pos)
        direction = direction / np.linalg.norm(direction)
        
        yaw = np.arctan2(direction[1], direction[0])
        xy_dist = np.sqrt(direction[0]**2 + direction[1]**2)
        pitch = np.arctan2(xy_dist, direction[2])
        
        grasp_orn = p.getQuaternionFromEuler([0, pitch, yaw]) 

        return grasp_pos, grasp_orn

    def calculate_approach_pose(self, grasp_pos, grasp_orn, distance=0.1):
        rot_matrix = p.getMatrixFromQuaternion(grasp_orn)
        z_axis = np.array([rot_matrix[2], rot_matrix[5], rot_matrix[8]])
        approach_pos = np.array(grasp_pos) - (z_axis * distance)
        return list(approach_pos), grasp_orn