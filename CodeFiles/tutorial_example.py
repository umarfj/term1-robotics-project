import pybullet as p
import time
import pybullet_data


physicsClient = p.connect(p.GUI)

p.resetDebugVisualizerCamera(
    cameraDistance=2.0,        # zoom in/out (smaller = closer)
    cameraYaw=50,              # left/right rotation
    cameraPitch=-35,           # up/down angle
    cameraTargetPosition=[0,0,0]  # where the camera points (e.g., the object or plane)
)

p.setGravity(0,0,-10)
p.setAdditionalSearchPath(pybullet_data.getDataPath())
planeId = p.loadURDF("plane.urdf")
cubeStartPos = [0,0,1]
cubeStartOrientation = p.getQuaternionFromEuler([0,0,-90])
boxId = p.loadURDF('r2d2.urdf',cubeStartPos,cubeStartOrientation)

for i in range(1000):
	p.stepSimulation()
	time.sleep(1./240.)


cubePos, cubeOrn = p.getBasePositionAndOrientation(boxId)
print(cubePos, cubeOrn)

