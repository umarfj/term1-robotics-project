import pybullet as p
import pybullet_data
from abc import ABC

class GraspableObject(ABC):
    def __init__(self, urdf_name, start_pos, scale=1.0):
        p.setAdditionalSearchPath(pybullet_data.getDataPath())
        self.urdf_name = urdf_name
        self.start_pos = start_pos
        self.scale = scale
        self.body_id = None
        self.load()

    def load(self):
        self.body_id = p.loadURDF(
            self.urdf_name, 
            self.start_pos, 
            globalScaling=self.scale
        )
        # PROFESSOR'S PHYSICS FIX:
        # 1. Mass: Keep it light (0.1kg)
        # 2. Friction: Add rolling and spinning friction!
        p.changeDynamics(
            self.body_id, 
            -1, # Base link
            mass=0.1, 
            lateralFriction=2.0,
            rollingFriction=0.1,
            spinningFriction=0.1
        )

    def get_position(self):
        if self.body_id is not None:
            pos, _ = p.getBasePositionAndOrientation(self.body_id)
            return pos
        return self.start_pos

    def get_height(self):
        pos = self.get_position()
        return pos[2]

    def reset(self):
        if self.body_id is not None:
            p.resetBasePositionAndOrientation(self.body_id, self.start_pos, [0,0,0,1])
    
    def cleanup(self):
        if self.body_id is not None:
            p.removeBody(self.body_id)
            self.body_id = None


class Cube(GraspableObject):
    def __init__(self, start_pos=[0, 0, 0.05]):
        super().__init__("cube_small.urdf", start_pos, scale=1.0)


class Duck(GraspableObject):
    def __init__(self, start_pos=[0, 0, 0.05]):
        super().__init__("duck_vhacd.urdf", start_pos, scale=0.8)