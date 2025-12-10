"""
OOP data generation module that mirrors the existing generate-data workflow
but encapsulates it in a reusable class. Saves output CSV into the `data/` folder.

Keeps logic consistent with main_generate_data.py: random object + gripper,
sample candidate grasp (with noise), execute, label, and persist features.
"""

import os
import csv
from typing import Optional

import numpy as np
import pybullet as p

from src.simulation.sim_manager import SimulationManager
from src.hardware.objects import Cube, Duck
from src.hardware.gripper import TwoFinger, FrankaPanda
from src.planning.sampler import GraspSampler


class DataGenerator:
    """Generate grasp dataset rows and save to CSV under `data/`.

    Features per row: [x, y, z, qx, qy, qz, qw, label]
    - label: 1 for success, 0 for fail
    
    Attributes:
        gui_mode: If True, show PyBullet GUI during generation.
        rng: NumPy random number generator for reproducibility.
        sim: SimulationManager instance for PyBullet control.
        sampler: GraspSampler for pose sampling with noise.
        root: Workspace root directory.
        data_dir: Output directory for CSV files (data/).
    """

    def __init__(self, gui_mode: bool = False, seed: Optional[int] = None):
        """Initialize data generator.
        
        Args:
            gui_mode: If True, show PyBullet GUI (default False).
            seed: Random seed for reproducibility, None for non-deterministic (default None).
        """
        self.gui_mode = gui_mode
        # Use a local RNG to avoid locking global numpy randomness across runs
        # When seed is None, the generator is non-deterministic.
        self.rng = np.random.default_rng(seed)

        # Setup simulation manager and grasp sampler
        self.sim = SimulationManager(gui_mode=self.gui_mode)
        self.sampler = GraspSampler()

        # Ensure data directory exists
        self.root = os.getcwd()
        self.data_dir = os.path.join(self.root, "data")
        os.makedirs(self.data_dir, exist_ok=True)

    def _choose_random_object(self):
        # Match main_generate_data.py: start z at 0.1
        if self.rng.random() > 0.5:
            return Cube(start_pos=[0, 0, 0.1]), "Cube"
        return Duck(start_pos=[0, 0, 0.1]), "Duck"

    def _choose_random_gripper(self):
        if self.rng.random() > 0.5:
            return TwoFinger(self.sim.client_id), "TwoFinger"
        return FrankaPanda(self.sim.client_id), "FrankaPanda"

    def _interpolate_pose(self, start_pos, end_pos, steps=50):
        path = []
        for t in np.linspace(0, 1, steps):
            pos = (1 - t) * np.array(start_pos) + t * np.array(end_pos)
            path.append(list(pos))
        return path

    def _generate_one(self, attempt_idx: int | None = None):
        """Generate one sample: sample grasp, execute, label and return features+label.

        Prints per-attempt outcome to mirror main_generate_data.py style.
        
        Args:
            attempt_idx: Attempt number for logging (optional).
            
        Returns:
            Tuple of (features, label) where features is [x, y, z, qx, qy, qz, qw]
            and label is 1 (success) or 0 (fail).
        """
        # Random world
        current_obj, _ = self._choose_random_object()
        self.sim.run_simulation(1.0)  # settle
        current_gripper, _ = self._choose_random_gripper()

        target_pos = current_obj.get_position()

        # Sample candidate grasp + approach (with noise via sampler)
        grasp_pos, grasp_orn = self.sampler.sample_grasp_pose(target_pos, radius=0.10)
        approach_pos, _ = self.sampler.calculate_approach_pose(grasp_pos, grasp_orn, distance=0.1)

        # Visual (optional; harmless in DIRECT)
        p.addUserDebugLine(approach_pos, grasp_pos, [0, 0, 1], lineWidth=2, lifeTime=1)

        # Execute grasp (match main_generate_data.py pacing and paths)
        current_gripper.open()
        current_gripper.move_to(approach_pos, grasp_orn)
        self.sim.run_simulation(0.5)

        # Approach via interpolated path (20 steps)
        approach_path = self._interpolate_pose(approach_pos, grasp_pos, steps=20)
        for point in approach_path:
            current_gripper.move_to(point, grasp_orn)
            self.sim.step()

        # Grasp
        current_gripper.close()
        self.sim.run_simulation(1.0)

        # Lift via interpolated path (100 steps) with continuous grip re-application
        lift_pos = [grasp_pos[0], grasp_pos[1], grasp_pos[2] + 0.3]
        lift_path = self._interpolate_pose(grasp_pos, lift_pos, steps=100)
        for point in lift_path:
            current_gripper.move_to(point, grasp_orn)
            current_gripper.close()  # re-apply grip force
            self.sim.step()

        # Hold
        self.sim.run_simulation(3.0)

        # Label success
        is_success = self.sim.check_success(current_obj)
        label = 1 if is_success else 0

        # Per-run outcome reporting
        if attempt_idx is not None:
            outcome = "SUCCESS" if is_success else "FAIL"
            print(f"Attempt {attempt_idx}: {outcome}")

        # Cleanup
        current_gripper.cleanup()
        current_obj.cleanup()
        self.sim.setup()

        features = [
            grasp_pos[0], grasp_pos[1], grasp_pos[2],
            grasp_orn[0], grasp_orn[1], grasp_orn[2], grasp_orn[3],
            label,
        ]
        return features

    def run(self, output_name: str = "grasp_dataset.csv", num_samples: int = 1000):
        """Generate num_samples and save to data/<output_name>.

        Prints each attempt's outcome and a final success rate summary.
        
        Args:
            output_name: Output CSV filename (default 'grasp_dataset.csv').
            num_samples: Number of grasp samples to generate (default 1000).
            
        Returns:
            Path to generated CSV file.
        """
        out_path = os.path.join(self.data_dir, output_name)

        # Write header if creating new file
        write_header = not os.path.exists(out_path)

        with open(out_path, mode="a", newline="") as f:
            writer = csv.writer(f)
            if write_header:
                writer.writerow(["x", "y", "z", "qx", "qy", "qz", "qw", "label"])

            successes = 0
            for i in range(num_samples):
                sample = self._generate_one(attempt_idx=i + 1)
                writer.writerow(sample)
                if sample[-1] == 1:
                    successes += 1
                if (i + 1) % 50 == 0:
                    print(f"Generated {i+1}/{num_samples} samples -> {out_path}")

        success_rate = successes / num_samples if num_samples else 0.0
        print(f"\nDone. Wrote {num_samples} samples to: {out_path}")
        print(f"Successes: {successes}/{num_samples} ({success_rate:.2%})")
