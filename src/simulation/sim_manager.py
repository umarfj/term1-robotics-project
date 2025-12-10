"""
Simulation manager module for PyBullet physics simulation.

This module provides a SimulationManager class that handles PyBullet
initialization, gravity setup, time-stepping, and grasp success evaluation.
"""

import pybullet as p
import pybullet_data
import time


class SimulationManager:
    """
    Manages PyBullet physics simulation environment.

    Handles PyBullet connection, world setup, physics stepping,
    and grasp success evaluation based on object height thresholds.

    Attributes:
        connection_mode: PyBullet connection mode (GUI or DIRECT).
        client_id: PyBullet physics client ID.
        plane_id: ID of the ground plane object.
    """

    def __init__(self, gui_mode=True):
        """
        Initialize simulation manager and connect to PyBullet.

        Args:
            gui_mode: If True, uses PyBullet GUI mode. If False, uses
                     DIRECT (headless) mode for faster execution.
        """
        self.connection_mode = p.GUI if gui_mode else p.DIRECT
        self.client_id = p.connect(self.connection_mode)
        p.setAdditionalSearchPath(pybullet_data.getDataPath())
        self.setup()

    def setup(self):
        """
        Reset simulation and initialize world with gravity and ground plane.

        Loads a ground plane and sets gravity to -9.8 m/s² in the z-direction.
        """
        p.resetSimulation()
        p.setGravity(0, 0, -9.8)
        self.plane_id = p.loadURDF("plane.urdf")

    def step(self):
        """
        Execute one physics simulation step.

        In GUI mode, adds a sleep delay to match real-time 240 Hz visualization.
        """
        p.stepSimulation()
        if self.connection_mode == p.GUI:
            time.sleep(1./240.)

    def run_simulation(self, duration_sec):
        """
        Run simulation for a specified duration.

        Args:
            duration_sec: Simulation duration in seconds. Steps are calculated
                         assuming 240 Hz physics frequency.
        """
        steps = int(duration_sec * 240)
        for _ in range(steps):
            self.step()

    def check_success(self, object_instance, height_threshold=0.2):
        """
        Evaluate grasp success based on object height.

        Args:
            object_instance: GraspableObject instance to check.
            height_threshold: Minimum height (meters) above ground plane
                             for successful grasp. Default is 0.2m.

        Returns:
            bool: True if object height exceeds threshold, False otherwise.
        """
        current_height = object_instance.get_height()
        return current_height > height_threshold

    def cleanup(self):
        """Disconnect from PyBullet physics server."""
        p.disconnect()