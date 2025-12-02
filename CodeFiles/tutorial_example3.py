import pybullet as p
import pybullet_data
import time

# Connect to physics server
cid = p.connect(p.SHARED_MEMORY)
if cid < 0:
    p.connect(p.GUI)

# Set up environment
p.setAdditionalSearchPath(pybullet_data.getDataPath())
p.resetSimulation()
p.setGravity(0, 0, -10)
p.setRealTimeSimulation(0)  # manual stepping

# Load plane
plane_id = p.loadURDF("plane.urdf")

# Load cube
cube_start_pos = [0.6, 0.3, 0.025]  # cube on the plane
cube_start_orientation = p.getQuaternionFromEuler([0, 0, 0])
cube_id = p.loadURDF("cube_small.urdf", cube_start_pos, cube_start_orientation, globalScaling=1.0)
#cube_id = p.loadURDF("cylinder.urdf", cube_start_pos, cube_start_orientation, globalScaling=1.0)
#cube_id = p.loadURDF("sphere.urdf", cube_start_pos, cube_start_orientation, globalScaling=1.0)

# Load PR2 gripper
pr2_gripper = p.loadURDF("pr2_gripper.urdf", 0.5, 0.3, 0.7, 0, 0, 0, 1)

# Initialize gripper joints (open)
joint_positions = [0.550569, 0.0, 0.549657, 0.0]
for i, pos in enumerate(joint_positions):
    p.resetJointState(pr2_gripper, i, pos)

# Fix gripper in place with a constraint
pr2_cid = p.createConstraint(
    parentBodyUniqueId=pr2_gripper,
    parentLinkIndex=-1,
    childBodyUniqueId=-1,
    childLinkIndex=-1,
    jointType=p.JOINT_FIXED,
    jointAxis=[0, 0, 0],
    parentFramePosition=[0.2, 0, 0],
    childFramePosition=[0.5, 0.3, 0.7]
)

# Helper functions
def close_gripper():
    for joint in [0, 2]:
        p.setJointMotorControl2(pr2_gripper, joint, p.POSITION_CONTROL,
                                targetPosition=0.1, maxVelocity=1, force=10)

def open_gripper():
    for joint in [0, 2]:
        p.setJointMotorControl2(pr2_gripper, joint, p.POSITION_CONTROL,
                                targetPosition=0.0, maxVelocity=1, force=10)

def move_gripper(z, yaw):
    p.configureDebugVisualizer(p.COV_ENABLE_RENDERING, 0)
    p.changeConstraint(
        pr2_cid,
        jointChildPivot=[0.5, 0.3, z],
        jointChildFrameOrientation=p.getQuaternionFromEuler([0, 0, yaw]),
        maxForce=50
    )
    p.configureDebugVisualizer(p.COV_ENABLE_RENDERING, 1)

def update_camera(z, yaw_angle):
    p.resetDebugVisualizerCamera(
        cameraDistance=0.5,
        cameraYaw=50 + (yaw_angle * 180 / 3.1416),
        cameraPitch=-60,
        cameraTargetPosition=[0.5, 0.3, z]
    )

# Initial camera and gripper state
z_pos = 0.7
yaw_angle = 0.0
update_camera(z_pos, yaw_angle)

# Step 1: Move gripper above cube
move_gripper(0.15 + 0.1, yaw_angle)  # slightly above cube (~0.025 cube height + gripper offset)
for _ in range(100):
    p.stepSimulation()
    time.sleep(1./240.)

# Step 2: Lower gripper onto cube
move_gripper(0.025 + 0.03, yaw_angle)  # touch cube (~cube height/2 + small offset)
for _ in range(100):
    p.stepSimulation()
    time.sleep(1./240.)

# Step 3: Close gripper to grasp cube
close_gripper()
for _ in range(100):
    p.stepSimulation()
    time.sleep(1./240.)

# Step 4: Lift cube
move_gripper(0.4, yaw_angle)  # lift to 0.4m
for _ in range(200):
    p.stepSimulation()
    time.sleep(1./240.)

# Keep GUI open
for _ in range(240):
    p.stepSimulation()
    time.sleep(1./240.)

p.disconnect()
