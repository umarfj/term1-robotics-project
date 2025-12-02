import pybullet as p
import pybullet_data
import time

class SimulationManager:
    def __init__(self, gui_mode=True):
        self.connection_mode = p.GUI if gui_mode else p.DIRECT
        self.client_id = p.connect(self.connection_mode)
        p.setAdditionalSearchPath(pybullet_data.getDataPath())
        self.setup()

    def setup(self):
        p.resetSimulation()
        p.setGravity(0, 0, -9.8)
        self.plane_id = p.loadURDF("plane.urdf")

    def step(self):
        p.stepSimulation()
        if self.connection_mode == p.GUI:
            time.sleep(1./240.)

    def run_simulation(self, duration_sec):
        steps = int(duration_sec * 240)
        for _ in range(steps):
            self.step()

    def check_success(self, object_instance, height_threshold=0.2):
        current_height = object_instance.get_height()
        return current_height > height_threshold

    def cleanup(self):
        p.disconnect()