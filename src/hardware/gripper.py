"""
Abstract Gripper class and subclasses.
UPDATES: Using constraint-based positioning (like professor's code) instead of useFixedBase.
"""

from abc import ABC, abstractmethod
import pybullet as p
import pybullet_data
import os
import numpy as np

class Gripper(ABC):
    def __init__(self, physics_client):
        self.physics_client = physics_client
        self.body_id = None
        self.constraint_id = None  # NEW: For constraint-based positioning
        self.urdf_path = None
        self.manual_offset = [0, 0, 0, 1] 
        self.tool_tip_offset = 0.0 
        self.vertical_bias = 1
    
    @abstractmethod
    def open(self):
        pass
    
    @abstractmethod
    def close(self):
        pass
    
    def attach_to_world(self, offset=[0, 0, 0]):
        """
        Create a FIXED constraint to attach gripper to world.
        This allows movement via constraint updates (professor's approach).
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
        Move gripper by updating constraint (if attached) or teleporting.
        Applies TCP offset calculation.
        """
        if self.body_id is not None:
            # 1. Apply Rotation Fix
            if self.manual_offset != [0, 0, 0, 1]:
                final_orn = p.multiplyTransforms(
                    [0,0,0], orientation, 
                    [0,0,0], self.manual_offset
                )[1]
            else:
                final_orn = orientation

            # 2. Apply TCP Offset
            rot_mat = p.getMatrixFromQuaternion(orientation)
            z_axis = np.array([rot_mat[2], rot_mat[5], rot_mat[8]])
            final_pos = np.array(position) - (z_axis * self.tool_tip_offset)
            
            # Apply Vertical Bias (World Z)
            final_pos[2] -= self.vertical_bias

            # 3. Move using constraint OR teleport
            if self.constraint_id is not None:
                # Use constraint for smooth movement (professor's method)
                p.changeConstraint(
                    self.constraint_id,
                    jointChildPivot=final_pos,
                    jointChildFrameOrientation=final_orn,
                    maxForce=500,
                    physicsClientId=self.physics_client
                )
            else:
                # Fallback to teleport if no constraint
                p.resetBasePositionAndOrientation(
                    self.body_id, 
                    final_pos, 
                    final_orn,
                    physicsClientId=self.physics_client
                )

    def cleanup(self):
        """Remove constraint and body."""
        if self.constraint_id is not None:
            p.removeConstraint(self.constraint_id, physicsClientId=self.physics_client)
            self.constraint_id = None
        if self.body_id is not None:
            p.removeBody(self.body_id, physicsClientId=self.physics_client)
            self.body_id = None

    def _set_friction(self, friction=2.0):
        """Set friction on all gripper links."""
        if self.body_id is not None:
            # Base link
            p.changeDynamics(self.body_id, -1,
                           lateralFriction=friction,
                           rollingFriction=0.1,
                           spinningFriction=0.1,
                           physicsClientId=self.physics_client)
            # All joints
            num_joints = p.getNumJoints(self.body_id, physicsClientId=self.physics_client)
            for i in range(num_joints):
                p.changeDynamics(self.body_id, i, 
                               lateralFriction=friction,
                               rollingFriction=0.1,
                               spinningFriction=0.1,
                               physicsClientId=self.physics_client)


class TwoFinger(Gripper):
    def __init__(self, physics_client):
        super().__init__(physics_client)
        self.urdf_path = os.path.join(pybullet_data.getDataPath(), "pr2_gripper.urdf")
        self.manual_offset = p.getQuaternionFromEuler([0, -1.57, 0])
        self.tool_tip_offset = 0.20  # Measured via measure_tip_offset.py
        self.vertical_bias = 0.08    # Drop 2cm in world -Z to ensure contact
        self._load_gripper()
    
    def _load_gripper(self):
        self.body_id = p.loadURDF(
            self.urdf_path,
            basePosition=[0, 0, 0.5],
            useFixedBase=False,  # FIXED: Changed to False (will use constraints)
            physicsClientId=self.physics_client
        )
        self._set_friction(2.0)
        # Auto-attach to world with constraint
        self.attach_to_world()
    
    def open(self):
        if self.body_id is not None:
            for j in [0, 2]:
                p.setJointMotorControl2(
                    self.body_id, j, p.POSITION_CONTROL,
                    targetPosition=0.54, 
                    force=20,
                    physicsClientId=self.physics_client
                )
    
    def close(self):
        """Close with HIGH force like professor's code (force=300-400)."""
        if self.body_id is not None:
            for j in [0, 2]:
                p.setJointMotorControl2(
                    self.body_id, j, p.POSITION_CONTROL,
                    targetPosition=0.0, 
                    force=300,          # INCREASED: Professor uses 300-400
                    maxVelocity=1.0,    # Moderate speed
                    physicsClientId=self.physics_client
                )


class ThreeFinger(Gripper):
    def __init__(self, physics_client):
        super().__init__(physics_client)
        
        current_dir = os.path.dirname(os.path.abspath(__file__))
        project_root = os.path.dirname(os.path.dirname(current_dir))
        self.urdf_path = os.path.join(project_root, "assets", "gripper_files", "threeFingers", "sdh.urdf")
        
        self.manual_offset = p.getQuaternionFromEuler([0, -1.57, 0]) 
        self.tool_tip_offset = 0  # ADJUSTED: Reduced from 0.22 to fix offset issue
        
        self._load_gripper()
    
    def _load_gripper(self):
        if not os.path.exists(self.urdf_path):
            print(f"Warning: URDF file not found at {self.urdf_path}")
            return
        
        self.body_id = p.loadURDF(
            self.urdf_path,
            basePosition=[0, 0, 0.5],
            useFixedBase=False,  # FIXED: Changed to False (will use constraints)
            physicsClientId=self.physics_client
        )
        self._set_friction(2.0)
        # Auto-attach to world with constraint
        self.attach_to_world()
    
    def open(self):
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
        """Close with HIGH force like professor's code (force=300-400)."""
        if self.body_id is not None:
            num_joints = p.getNumJoints(self.body_id, physicsClientId=self.physics_client)
            for joint_index in range(num_joints):
                info = p.getJointInfo(self.body_id, joint_index, physicsClientId=self.physics_client)
                if info[2] in [p.JOINT_REVOLUTE, p.JOINT_PRISMATIC]:
                    p.setJointMotorControl2(
                        self.body_id, joint_index, p.POSITION_CONTROL,
                        targetPosition=1.0, 
                        force=300.0,      # INCREASED: Professor uses 300-400
                        maxVelocity=1.0,  # Moderate speed
                        physicsClientId=self.physics_client
                    )