import pybullet as p
import pybullet_data
import time
import os

# 1. Start the Simulation in GUI mode (opens a window)
physicsClient = p.connect(p.GUI)

# 2. Add standard assets (like the floor) to the search path
p.setAdditionalSearchPath(pybullet_data.getDataPath())

# 3. Setup the environment
p.setGravity(0, 0, -9.8)
planeId = p.loadURDF("plane.urdf")

# 4. Define the path to your URDF
# NOTE: Update this string to match exactly where your URDF file is located!
# Based on your tree, it's likely one of these two:
gripper_urdf_path = ".\\assets\\gripper_files\\robotiq_2f_85\\robotiq.urdf"

# Check if file exists before trying to load
if not os.path.exists(gripper_urdf_path):
    print(f"Error: Could not find file at {gripper_urdf_path}")
    print("Please check inside your folders and update the 'gripper_urdf_path' variable.")
else:
    # 5. Load the Gripper
    # useFixedBase=True keeps it floating in air so it doesn't fall over
    startPos = [0, 0, 0.5] # Suspended 0.5 meters in the air
    startOrientation = p.getQuaternionFromEuler([0, 0, 0])
    
    gripperId = p.loadURDF(gripper_urdf_path, startPos, startOrientation, useFixedBase=True)
    
    # Optional: Reset camera to look at the gripper
    p.resetDebugVisualizerCamera(cameraDistance=0.5, cameraYaw=45, cameraPitch=-30, cameraTargetPosition=startPos)

    print("Gripper loaded successfully! Press Ctrl+C in terminal to exit.")

    # 6. Keep the window open
    while p.isConnected():
        p.stepSimulation()
        time.sleep(1./240.) # Run at 240Hz