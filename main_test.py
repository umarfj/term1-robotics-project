import pybullet as p
import pybullet_data
import time
import os

# --- CONFIGURATION ---
# Path to the suspect gripper (SDH)
# Adjust this path if your folder structure is different
GRIPPER_PATH = r"./assets/gripper_files/threeFingers/sdh.urdf"

def main():
    p.connect(p.GUI)
    p.setAdditionalSearchPath(pybullet_data.getDataPath())
    
    # 1. Load Environment
    p.setGravity(0, 0, 0) # No gravity so it floats
    p.loadURDF("plane.urdf")
    
    # 2. Load the Gripper at exactly (0, 0, 1) in the world
    start_pos = [0, 0, 1.0]
    start_orn = p.getQuaternionFromEuler([0, 0, 0])
    
    if not os.path.exists(GRIPPER_PATH):
        print(f"Error: File not found at {GRIPPER_PATH}")
        return

    gripper_id = p.loadURDF(GRIPPER_PATH, start_pos, start_orn, useFixedBase=True)
    
    # 3. Draw the Origin of the URDF
    # We draw lines starting from the robot's base position
    # RED = X, GREEN = Y, BLUE = Z
    line_len = 0.2
    p.addUserDebugLine(start_pos, [start_pos[0]+line_len, start_pos[1], start_pos[2]], [1, 0, 0], lineWidth=5)
    p.addUserDebugLine(start_pos, [start_pos[0], start_pos[1]+line_len, start_pos[2]], [0, 1, 0], lineWidth=5)
    p.addUserDebugLine(start_pos, [start_pos[0], start_pos[1], start_pos[2]+line_len], [0, 0, 1], lineWidth=5)
    
    # 4. Draw a "Target" ball at the same location
    # If the wrist is inside this ball, the Origin is at the wrist.
    visual_shape = p.createVisualShape(p.GEOM_SPHERE, radius=0.1, rgbaColor=[1, 1, 0, 0.5])
    p.createMultiBody(baseVisualShapeIndex=visual_shape, basePosition=start_pos)

    print("------------------------------------------------")
    print("LOOK AT THE YELLOW BALL IN THE SIMULATION.")
    print("1. If the WRIST is inside the ball -> Standard URDF (Needs Offset in Code)")
    print("2. If the FINGERTIPS are inside the ball -> Grasp-Ready URDF (No Offset Needed)")
    print("------------------------------------------------")

    p.resetDebugVisualizerCamera(cameraDistance=0.5, cameraYaw=45, cameraPitch=-30, cameraTargetPosition=start_pos)
    
    while p.isConnected():
        p.stepSimulation()
        time.sleep(1./240.)

if __name__ == "__main__":
    main()