"""
Graspable object classes for robotic manipulation simulation.

This module defines abstract and concrete graspable object classes
with physics properties optimized for stable grasping in PyBullet.
"""

import pybullet as p
import pybullet_data
from abc import ABC


class GraspableObject(ABC):
    """
    Abstract base class for objects that can be grasped.

    Provides common interface for loading, positioning, querying,
    and managing object lifecycle in PyBullet simulation.

    Attributes:
        urdf_name: Name of URDF file to load.
        start_pos: Initial position [x, y, z] in meters.
        scale: Global scaling factor for object size.
        body_id: PyBullet body unique ID (None if not loaded).
    """

    def __init__(self, urdf_name, start_pos, scale=1.0):
        """
        Initialize graspable object.

        Args:
            urdf_name: URDF filename (e.g., "cube_small.urdf").
            start_pos: Initial position as [x, y, z] list in meters.
            scale: Scaling factor for object size. Default is 1.0.
        """
        p.setAdditionalSearchPath(pybullet_data.getDataPath())
        self.urdf_name = urdf_name
        self.start_pos = start_pos
        self.scale = scale
        self.body_id = None
        self.load()

    def load(self):
        """
        Load object URDF into simulation with physics properties.

        Sets mass to 0.1 kg and applies friction coefficients
        (lateral, rolling, spinning) for stable contact dynamics.
        """
        self.body_id = p.loadURDF(
            self.urdf_name,
            self.start_pos,
            globalScaling=self.scale
        )
        # Physics configuration for stable grasping
        p.changeDynamics(
            self.body_id,
            -1,  # Base link
            mass=0.1,
            lateralFriction=2.0,
            rollingFriction=0.1,
            spinningFriction=0.1
        )

    def get_position(self):
        """
        Query object's current position in world coordinates.

        Returns:
            list: [x, y, z] position in meters, or start_pos if not loaded.
        """
        if self.body_id is not None:
            pos, _ = p.getBasePositionAndOrientation(self.body_id)
            return pos
        return self.start_pos

    def get_height(self):
        """
        Get object's current height (z-coordinate) above ground plane.

        Returns:
            float: Height in meters.
        """
        pos = self.get_position()
        return pos[2]

    def reset(self):
        """Reset object to initial position and orientation."""
        if self.body_id is not None:
            p.resetBasePositionAndOrientation(
                self.body_id,
                self.start_pos,
                [0, 0, 0, 1]
            )

    def cleanup(self):
        """Remove object from simulation and clear body ID."""
        if self.body_id is not None:
            p.removeBody(self.body_id)
            self.body_id = None


class Cube(GraspableObject):
    """
    Cube object for grasping experiments.

    Uses PyBullet's cube_small.urdf with regular geometry
    suitable for testing basic grasp strategies.
    """

    def __init__(self, start_pos=[0, 0, 0.05]):
        """
        Initialize cube object.

        Args:
            start_pos: Initial position [x, y, z]. Default is [0, 0, 0.05].
        """
        super().__init__("cube_small.urdf", start_pos, scale=1.0)


class Duck(GraspableObject):
    """
    Duck object for grasping experiments.

    Uses PyBullet's duck_vhacd.urdf with irregular geometry
    to test grasp robustness on non-trivial shapes.
    """

    def __init__(self, start_pos=[0, 0, 0.05]):
        """
        Initialize duck object.

        Args:
            start_pos: Initial position [x, y, z]. Default is [0, 0, 0.05].
        """
        super().__init__("duck_vhacd.urdf", start_pos, scale=0.8)