"""
Abstract gripper class and concrete implementations for robotic manipulation.

This module defines an abstract Gripper base class and three concrete
implementations: TwoFinger (PR2 gripper), ThreeFinger (SDH gripper),
and FrankaPanda (7-DOF manipulator with IK control).

All grippers support constraint-based positioning for realistic physics simulation.
"""

from abc import ABC, abstractmethod
import pybullet as p
import pybullet_data
import os
import numpy as np
import math


class Gripper(ABC):
    """
    Abstract base class for robotic grippers.

    Defines common interface for gripper control including opening,
    closing, positioning, and constraint-based attachment to world frame.

    Attributes:
        physics_client: PyBullet physics client ID.
        body_id: PyBullet body unique ID (None if not loaded).
        constraint_id: Fixed constraint ID for world attachment (None if not attached).
        urdf_path: Path to gripper URDF file.
        manual_offset: Orientation offset as quaternion [x, y, z, w].
        tool_tip_offset: TCP offset distance in meters.
    """

    def __init__(self, physics_client):
        """
        Initialize gripper base class.

        Args:
            physics_client: PyBullet physics client ID for simulation context.
        """
        self.physics_client = physics_client
        self.body_id = None
        self.constraint_id = None
        self.urdf_path = None
        self.manual_offset = [0, 0, 0, 1]
        self.tool_tip_offset = 0.0

    @abstractmethod
    def open(self):
        """Open gripper fingers to maximum width. Must be implemented by subclasses."""
        pass

    @abstractmethod
    def close(self):
        """Close gripper fingers to grasp objects. Must be implemented by subclasses."""
        pass

    def attach_to_world(self, offset=[0, 0, 0]):
        """
        Create a fixed constraint to attach gripper base to world frame.

        This enables smooth movement via constraint updates instead of
        teleportation, providing more realistic physics simulation.

        Args:
            offset: Position offset [x, y, z] for constraint parent frame.
        """
        if self.body_id is not None and self.constraint_id is None:
            self.constraint_id = p.createConstraint(
                parentBodyUniqueId=self.body_id,
                parentLinkIndex=-1,
                childBodyUniqueId=-1,
                childLinkIndex=-1,
                jointType=p.JOINT_FIXED,
                jointAxis=[0, 0, 0],
                parentFramePosition=offset,
                childFramePosition=[0, 0, 0],
                physicsClientId=self.physics_client
            )

    def move_to(self, position, orientation):
        """
        Move gripper to target pose using constraint update or teleportation.

        Applies manual orientation offset and TCP offset before positioning.
        If constraint exists, updates constraint for smooth motion. Otherwise
        teleports gripper directly (fallback for unconstrained grippers).

        Args:
            position: Target position as [x, y, z] list in meters.
            orientation: Target orientation as quaternion [x, y, z, w].
        """
        if self.body_id is not None:
            # 1. Apply manual rotation offset (gripper-specific correction)
            if self.manual_offset != [0, 0, 0, 1]:
                final_orn = p.multiplyTransforms(
                    [0, 0, 0], orientation,
                    [0, 0, 0], self.manual_offset
                )[1]
            else:
                final_orn = orientation

            # 2. Apply TCP offset (tool center point correction)
            rot_mat = p.getMatrixFromQuaternion(orientation)
            z_axis = np.array([rot_mat[2], rot_mat[5], rot_mat[8]])
            final_pos = np.array(position) - (z_axis * self.tool_tip_offset)

            # 3. Move using constraint update OR direct teleportation
            if self.constraint_id is not None:
                # Smooth constraint-based movement
                p.changeConstraint(
                    self.constraint_id,
                    jointChildPivot=final_pos,
                    jointChildFrameOrientation=final_orn,
                    maxForce=500,
                    physicsClientId=self.physics_client
                )
            else:
                # Fallback: direct teleportation (no constraint)
                p.resetBasePositionAndOrientation(
                    self.body_id,
                    final_pos,
                    final_orn,
                    physicsClientId=self.physics_client
                )

    def cleanup(self):
        """Remove constraint and body from simulation."""
        if self.constraint_id is not None:
            p.removeConstraint(self.constraint_id, physicsClientId=self.physics_client)
            self.constraint_id = None
        if self.body_id is not None:
            p.removeBody(self.body_id, physicsClientId=self.physics_client)
            self.body_id = None

    def _set_friction(self, friction=2.0):
        """
        Set friction coefficients on all gripper links.

        Args:
            friction: Lateral friction coefficient. Default is 2.0.
                     Also sets rolling (0.1) and spinning (0.1) friction.
        """
        if self.body_id is not None:
            # Base link
            # Base link
            p.changeDynamics(self.body_id, -1,
                           lateralFriction=friction,
                           rollingFriction=0.1,
                           spinningFriction=0.1,
                           physicsClientId=self.physics_client)
            # All joint links
            num_joints = p.getNumJoints(self.body_id, physicsClientId=self.physics_client)
            for i in range(num_joints):
                p.changeDynamics(self.body_id, i,
                               lateralFriction=friction,
                               rollingFriction=0.1,
                               spinningFriction=0.1,
                               physicsClientId=self.physics_client)


class TwoFinger(Gripper):
    """
    Two-finger PR2 gripper implementation.

    Parallel jaw gripper with constraint-based positioning.
    Uses PR2 gripper URDF from PyBullet data.

    Attributes:
        urdf_path: Path to pr2_gripper.urdf.
        manual_offset: 90° pitch rotation to align gripper orientation.
        tool_tip_offset: TCP offset distance (0.205m).
        vertical_bias: Additional downward offset (0.08m) for contact.
    """

    def __init__(self, physics_client):
        """
        Initialize PR2 two-finger gripper.

        Args:
            physics_client: PyBullet physics client ID.
        """
        super().__init__(physics_client)
        self.urdf_path = os.path.join(pybullet_data.getDataPath(), "pr2_gripper.urdf")
        self.manual_offset = p.getQuaternionFromEuler([0, -1.57, 0])
        self.tool_tip_offset = 0.205
        self.vertical_bias = 0.08
        self._load_gripper()

    def _load_gripper(self):
        """Load PR2 gripper URDF and attach to world with constraint."""
        self.body_id = p.loadURDF(
            self.urdf_path,
            basePosition=[0, 0, 0.5],
            useFixedBase=False,
            physicsClientId=self.physics_client
        )
        self._set_friction(2.0)
        self.attach_to_world()

    def open(self):
        """Open gripper to maximum width (0.54 rad for each jaw)."""
        if self.body_id is not None:
            for j in [0, 2]:
                p.setJointMotorControl2(
                    self.body_id, j, p.POSITION_CONTROL,
                    targetPosition=0.54,
                    force=20,
                    physicsClientId=self.physics_client
                )

    def close(self):
        """
        Close gripper with high force for secure grasping.

        Applies 300N force to each jaw (joints 0 and 2) for stable contact.
        """
        if self.body_id is not None:
            for j in [0, 2]:
                p.setJointMotorControl2(
                    self.body_id, j, p.POSITION_CONTROL,
                    targetPosition=0.0,
                    force=300,
                    maxVelocity=1.0,
                    physicsClientId=self.physics_client
                )


class ThreeFinger(Gripper):
    """
    Three-finger SDH gripper implementation.

    Schunk Dexterous Hand with three articulated fingers.
    Uses custom SDH URDF from assets/gripper_files.

    Attributes:
        urdf_path: Path to sdh.urdf in project assets.
        manual_offset: 180° yaw rotation for gripper alignment.
        tool_tip_offset: TCP offset distance (0m for SDH).
    """

    def __init__(self, physics_client):
        """
        Initialize SDH three-finger gripper.

        Args:
            physics_client: PyBullet physics client ID.
        """
        super().__init__(physics_client)

        current_dir = os.path.dirname(os.path.abspath(__file__))
        project_root = os.path.dirname(os.path.dirname(current_dir))
        self.urdf_path = os.path.join(project_root, "assets", "gripper_files", "threeFingers", "sdh.urdf")

        self.manual_offset = p.getQuaternionFromEuler([0, 0, 3.14159])
        self.tool_tip_offset = 0

        self._load_gripper()

    def _load_gripper(self):
        """Load SDH gripper URDF and attach to world with constraint."""
        if not os.path.exists(self.urdf_path):
            print(f"Warning: URDF file not found at {self.urdf_path}")
            return

        self.body_id = p.loadURDF(
            self.urdf_path,
            basePosition=[0, 0, 0.5],
            useFixedBase=False,
            physicsClientId=self.physics_client
        )
        self._set_friction(2.0)
        self.attach_to_world()

    def open(self):
        """Open all fingers to fully extended position (0.0 rad)."""
        if self.body_id is not None:
            num_joints = p.getNumJoints(self.body_id, physicsClientId=self.physics_client)
            for joint_index in range(num_joints):
                info = p.getJointInfo(self.body_id, joint_index, physicsClientId=self.physics_client)
                if info[2] in [p.JOINT_REVOLUTE, p.JOINT_PRISMATIC]:
                    p.setJointMotorControl2(
                        self.body_id, joint_index, p.POSITION_CONTROL,
                        targetPosition=0.0,
                        force=20,
                        physicsClientId=self.physics_client
                    )

    def close(self):
        """
        Close fingers with high force for secure grasping.

        Flexion joints (finger_12, finger_13, etc.) close to 1.0 rad.
        Knuckle/spread joints maintain 0.0 rad for straight configuration.
        Applies 300N force per joint.
        """
        if self.body_id is not None:
            num_joints = p.getNumJoints(self.body_id, physicsClientId=self.physics_client)
            for joint_index in range(num_joints):
                info = p.getJointInfo(self.body_id, joint_index, physicsClientId=self.physics_client)
                joint_name = info[1].decode('utf-8')

                if info[2] in [p.JOINT_REVOLUTE, p.JOINT_PRISMATIC]:
                    target_pos = 0.0

                    # Flexion joints (fingers closing)
                    if any(x in joint_name for x in ["12_joint", "13_joint", "22_joint", "23_joint", "thumb_2_joint", "thumb_3_joint"]):
                        target_pos = 1.0

                    # Knuckle/Spread joints (keep straight)
                    elif any(x in joint_name for x in ["knuckle_joint", "21_joint"]):
                        target_pos = 0.0

                    p.setJointMotorControl2(
                        self.body_id, joint_index, p.POSITION_CONTROL,
                        targetPosition=target_pos,
                        force=300.0,
                        maxVelocity=1.0,
                        physicsClientId=self.physics_client
                    )


class FrankaPanda(Gripper):
    """
    Franka Panda 7-DOF manipulator with inverse kinematics control.

    Advanced robotic arm with redundant DOF enabling null-space optimization
    for collision avoidance and singularity handling. Uses PyBullet IK solver.

    Attributes:
        urdf_path: Path to panda.urdf in PyBullet data.
        num_dofs: Number of arm degrees of freedom (7).
        end_effector_index: Link index for IK target (11).
        tool_tip_offset_xyz: TCP offset in local gripper frame [x, y, z].
        ll: Lower joint limits (radians).
        ul: Upper joint limits (radians).
        jr: Joint ranges for null-space optimization (radians).
        rp: Rest pose configuration for null-space bias (radians).
    """

    def __init__(self, physics_client):
        """
        Initialize Franka Panda manipulator.

        Args:
            physics_client: PyBullet physics client ID.
        """
        super().__init__(physics_client)
        self.urdf_path = os.path.join(pybullet_data.getDataPath(), "franka_panda/panda.urdf")
        self.num_dofs = 7
        self.end_effector_index = 11
        self.tool_tip_offset_xyz = np.array([-0.01, 0, -0.105], dtype=float)

        # IK solver parameters
        self.ll = [-7] * self.num_dofs  # Lower limits
        self.ul = [7] * self.num_dofs   # Upper limits
        self.jr = [7] * self.num_dofs   # Joint ranges
        self.rp = [0.98, 0.458, 0.31, -2.24, -0.30, 2.66, 2.32, 0.02, 0.02]  # Rest pose

        self._load_gripper()

    def _load_gripper(self):
        """
        Load Franka Panda URDF and initialize to rest pose.

        Robot base positioned at [-0.6, 0, 0] to reach workspace origin.
        """
        self.body_id = p.loadURDF(
            self.urdf_path,
            basePosition=[-0.6, 0, 0],
            useFixedBase=True,
            physicsClientId=self.physics_client
        )

        # Reset to rest pose and disable damping
        index = 0
        for j in range(p.getNumJoints(self.body_id, physicsClientId=self.physics_client)):
            p.changeDynamics(self.body_id, j, linearDamping=0, angularDamping=0, physicsClientId=self.physics_client)
            info = p.getJointInfo(self.body_id, j, physicsClientId=self.physics_client)
            jointType = info[2]
            if jointType in [p.JOINT_PRISMATIC, p.JOINT_REVOLUTE]:
                if index < len(self.rp):
                    p.resetJointState(self.body_id, j, self.rp[index], physicsClientId=self.physics_client)
                    index += 1

    def attach_to_world(self, offset=[0, 0, 0]):
        """
        Franka Panda is fixed base - no constraint needed.

        Args:
            offset: Ignored for Franka Panda (fixed base).
        """
        pass

    def set_tcp_offset(self, dx=0.0, dy=0.0, dz=0.0):
        """
        Set tool center point offset in local gripper frame.

        Args:
            dx: Offset along local x-axis (meters).
            dy: Offset along local y-axis (meters).
            dz: Offset along local z-axis (meters).
        """
        self.tool_tip_offset_xyz = np.array([dx, dy, dz], dtype=float)

    def move_to(self, position, orientation):
        """
        Move end-effector to target pose using inverse kinematics.

        Computes IK solution with null-space optimization biased toward
        rest pose. Applies TCP offset in world frame before IK calculation.

        Args:
            position: Target position as [x, y, z] list (meters).
            orientation: Target orientation as quaternion [x, y, z, w].
        """
        if self.body_id is not None:
            # Apply TCP offset in world frame
            rot_mat = p.getMatrixFromQuaternion(orientation)
            x_axis = np.array([rot_mat[0], rot_mat[3], rot_mat[6]])
            y_axis = np.array([rot_mat[1], rot_mat[4], rot_mat[7]])
            z_axis = np.array([rot_mat[2], rot_mat[5], rot_mat[8]])

            tcp_world_offset = (
                x_axis * self.tool_tip_offset_xyz[0]
                + y_axis * self.tool_tip_offset_xyz[1]
                + z_axis * self.tool_tip_offset_xyz[2]
            )
            final_pos = np.array(position) - tcp_world_offset

            # Stage 1: Hover above target (optional staging for smoother motion)
            hover_height = 0
            hover_pos = final_pos.copy()
            hover_pos[2] = final_pos[2] + hover_height

            hover_joint_poses = p.calculateInverseKinematics(
                self.body_id,
                self.end_effector_index,
                hover_pos,
                orientation,
                self.ll,
                self.ul,
                self.jr,
                self.rp,
                maxNumIterations=20,
                physicsClientId=self.physics_client
            )
            for i in range(self.num_dofs):
                p.setJointMotorControl2(
                    self.body_id, i, p.POSITION_CONTROL,
                    hover_joint_poses[i], force=5 * 240.,
                    physicsClientId=self.physics_client
                )

            # Stage 2: Descend to final position
            final_joint_poses = p.calculateInverseKinematics(
                self.body_id,
                self.end_effector_index,
                final_pos,
                orientation,
                self.ll,
                self.ul,
                self.jr,
                self.rp,
                maxNumIterations=20,
                physicsClientId=self.physics_client
            )
            for i in range(self.num_dofs):
                p.setJointMotorControl2(
                    self.body_id, i, p.POSITION_CONTROL,
                    final_joint_poses[i], force=5 * 240.,
                    physicsClientId=self.physics_client
                )

    def open(self):
        """Open gripper fingers to maximum width (0.04m per finger)."""
        if self.body_id is not None:
            for i in [9, 10]:  # Finger joints
                p.setJointMotorControl2(
                    self.body_id,
                    i,
                    p.POSITION_CONTROL,
                    0.04,
                    force=100,
                    physicsClientId=self.physics_client
                )

    def close(self):
        """Close gripper fingers to grasp objects (0.0m - fully closed)."""
        if self.body_id is not None:
            for i in [9, 10]:  # Finger joints
                p.setJointMotorControl2(
                    self.body_id,
                    i,
                    p.POSITION_CONTROL,
                    0.0,
                    force=100,
                    physicsClientId=self.physics_client
                )

