"""
Grasp pose sampling module with noise injection.

This module provides a GraspSampler class that generates candidate grasp
poses with realistic position and orientation noise to simulate sensor
uncertainty and motor inaccuracies in robotic manipulation.
"""

import numpy as np
import pybullet as p


class GraspSampler:
    """
    Samples candidate grasp poses with configurable noise injection.

    Generates grasp poses on a sphere around target objects with Gaussian
    noise applied to position (σ=0.015m) and orientation (σ=0.08 rad) to
    reflect real-world robotic system uncertainties.

    Attributes:
        rng_seed: Random seed for reproducibility (optional).
    """

    def __init__(self, rng_seed=None):
        """
        Initialize grasp sampler.

        Args:
            rng_seed: Random seed for NumPy random number generator.
                     If None, uses non-deterministic random state.
        """
        if rng_seed is not None:
            np.random.seed(rng_seed)

    def sample_grasp_pose(self, target_pos, radius):
        """
        Sample a noisy grasp pose around a target object.

        Generates an ideal grasp position on a sphere of specified radius
        centered on the target, aligns gripper orientation toward the object,
        then adds Gaussian noise to simulate real-world uncertainty.

        Args:
            target_pos: Target object center position as [x, y, z] list.
            radius: Sampling radius in meters (typical: 0.10m).

        Returns:
            tuple: (grasp_pos, grasp_orn) where:
                - grasp_pos: Noisy grasp position as [x, y, z] list (meters)
                - grasp_orn: Noisy grasp orientation as quaternion [x,y,z,w]

        Noise Parameters:
            - Position noise: σ=0.015m (1.5cm standard deviation)
            - Orientation noise: σ=0.08 rad (≈4.6° standard deviation)
        """
        # 1. Calculate ideal geometry (spherical sampling on upper hemisphere)
        phi = np.random.uniform(0, 2 * np.pi)
        theta = np.random.uniform(0, np.pi / 3)

        x_offset = radius * np.sin(theta) * np.cos(phi)
        y_offset = radius * np.sin(theta) * np.sin(phi)
        z_offset = radius * np.cos(theta)

        # Ideal position on sphere
        ideal_pos = [
            target_pos[0] + x_offset,
            target_pos[1] + y_offset,
            target_pos[2] + z_offset
        ]

        # Calculate ideal orientation (gripper pointing toward object)
        direction = np.array(target_pos) - np.array(ideal_pos)
        direction = direction / np.linalg.norm(direction)

        yaw = np.arctan2(direction[1], direction[0])
        xy_dist = np.sqrt(direction[0]**2 + direction[1]**2)
        pitch = np.arctan2(xy_dist, direction[2])
        roll = 0.0

        # 2. Add position noise (Gaussian, σ=1.5cm per axis)
        pos_noise = np.random.normal(0, 0.015, 3)
        noisy_grasp_pos = [
            ideal_pos[0] + pos_noise[0],
            ideal_pos[1] + pos_noise[1],
            ideal_pos[2] + pos_noise[2]
        ]

        # 3. Add orientation noise (Gaussian, σ=0.08 rad ≈ 4.6°)
        r_noise = np.random.normal(0, 0.08)
        p_noise = np.random.normal(0, 0.08)
        y_noise = np.random.normal(0, 0.08)

        noisy_yaw = yaw + y_noise
        noisy_pitch = pitch + p_noise
        noisy_roll = roll + r_noise

        # Convert noisy Euler angles to quaternion
        base_orn = p.getQuaternionFromEuler([0, noisy_pitch, noisy_yaw])
        grasp_orn = p.multiplyTransforms(
            [0, 0, 0], base_orn,
            [0, 0, 0], p.getQuaternionFromEuler([0, 0, noisy_roll])
        )[1]

        return noisy_grasp_pos, grasp_orn

    def calculate_approach_pose(self, grasp_pos, grasp_orn, distance=0.1):
        """
        Calculate pre-grasp approach pose offset along gripper's approach axis.

        Moves the gripper backwards along its z-axis by the specified distance
        to create a collision-free approach trajectory.

        Args:
            grasp_pos: Final grasp position as [x, y, z] list (meters).
            grasp_orn: Final grasp orientation as quaternion [x, y, z, w].
            distance: Approach distance in meters. Default is 0.1m.

        Returns:
            tuple: (approach_pos, approach_orn) where:
                - approach_pos: Position offset by distance along -z axis
                - approach_orn: Same orientation as grasp_orn
        """
        # Extract z-axis (approach direction) from rotation matrix
        rot_matrix = p.getMatrixFromQuaternion(grasp_orn)
        z_axis = np.array([rot_matrix[2], rot_matrix[5], rot_matrix[8]])

        # Move backwards along approach axis
        approach_pos = np.array(grasp_pos) - (z_axis * distance)
        return list(approach_pos), grasp_orn