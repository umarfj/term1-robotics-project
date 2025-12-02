import pybullet as p
import pybullet_data
import time
import random

class SceneObject:
    '''
    Define a Class attribute: counts how many objects exist
    Write the __init__() that takes as arguments, name of the object, 
    name of the urdf file, position and/or orientation. Using these
    arguments, setup the instance attributes, load the object URDF file
    and increment the class attribute so it can be printed in line 60
    '''


    def move_up(self, amount):
        """Move the object up by the given amount"""
        self.position[2] += amount
        p.resetBasePositionAndOrientation(self.body_id, self.position, self.orientation)
        print(f"{self.name} moved up by {amount}. New z-position: {self.position[2]}")


if __name__ == "__main__":
    # Connect to physics server
    cid = p.connect(p.SHARED_MEMORY)
    if cid < 0:
        p.connect(p.GUI)

    p.setAdditionalSearchPath(pybullet_data.getDataPath())
    p.resetSimulation()
    p.setRealTimeSimulation(0)
    p.setGravity(0, 0, -10)

    # Create a floor
    plane_id = p.loadURDF("plane.urdf")

    objects = []

    # Random number of boxes and cylinders (up to 10 each)
    num_boxes = random.randint(1, 10)
    num_cylinders = random.randint(1, 10)

    # Create boxes above the floor
    for i in range(num_boxes):
        pos = (random.uniform(-1, 1), random.uniform(-1, 1), random.uniform(0.5, 1.5))
        obj = SceneObject(f"Box_{i+1}", "cube_small.urdf", position=pos)
        objects.append(obj)

    # Create cylinders above the floor
    for i in range(num_cylinders):
        pos = (random.uniform(-1, 1), random.uniform(-1, 1), random.uniform(0.5, 1.5))
        obj = SceneObject(f"Cylinder_{i+1}", "cylinder.urdf", position=pos)
        objects.append(obj)

    # Example: move all objects up by a small random amount
    for obj in objects:
        obj.move_up(random.uniform(0.1, 0.5))

    # Print total objects using the class variable
    print("Total objects in scene:", ******FILL_IN_HERE****)

    # Run simulation for a while so objects settle on the floor
    for _ in range(480):
        p.stepSimulation()
        time.sleep(1./240.)

    p.disconnect()
