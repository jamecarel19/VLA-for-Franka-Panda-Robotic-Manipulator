"""
Webots Dataset Recorder for Imitation Learning
Franka Panda + Dual Camera Setup

Records synchronized multi-modal data for vision-language-action models.
"""

import os
import sys
import json
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass, asdict
from PIL import Image
import numpy as np

from controller import Robot, Camera, Supervisor, Node


@dataclass
class RobotState:
    """Robot state snapshot."""
    t: int
    joints: List[float]
    ee_pose: List[float]  # [x, y, z, qx, qy, qz, qw]
    gripper: float
    object_positions: Dict[str, List[float]]


@dataclass
class Action:
    """Action command."""
    t: int
    delta_xyz: List[float]
    delta_rpy: List[float]
    gripper_cmd: int  # 0=open, 1=close


@dataclass
class Timestamp:
    """Timestep timing."""
    t: int
    sim_time: float
    wall_time: float


@dataclass
class EpisodeMetadata:
    """Episode-level metadata."""
    episode_id: int
    task: str
    difficulty: str
    success: bool
    num_steps: int
    operator: str
    date: str
    simulator: str
    robot: str
    notes: str


class DatasetRecorder:
    """
    Manages dataset recording for imitation learning.
    
    Directory structure:
        dataset/
            episode_0001/
                frames/
                    000001_grip.png
                    000001_work.png
                    ...
                states.jsonl
                actions.jsonl
                timestamps.jsonl
                meta.json
    """
    
    def __init__(
        self,
        robot: Supervisor,
        gripper_camera: Camera,
        workspace_camera: Camera,
        dataset_root: str = "dataset",
        operator: str = "jame"
    ):
        self.robot = robot
        self.gripper_camera = gripper_camera
        self.workspace_camera = workspace_camera
        self.dataset_root = Path(dataset_root)
        self.operator = operator
        
        # Episode state
        self.recording = False
        self.episode_dir: Optional[Path] = None
        self.episode_id: Optional[int] = None
        self.current_step = 0
        
        # Data buffers
        self.states_buffer: List[RobotState] = []
        self.actions_buffer: List[Action] = []
        self.timestamps_buffer: List[Timestamp] = []
        
        # Task metadata
        self.current_task = ""
        self.current_difficulty = "easy"
        
        # Previous state for delta calculation
        self.prev_ee_pose: Optional[np.ndarray] = None
        self.prev_gripper: Optional[float] = None
        
        # Object tracking nodes
        self.tracked_objects: Dict[str, Node] = {}
        
        # Create root directory
        self.dataset_root.mkdir(parents=True, exist_ok=True)

        # Paths for inter-controller workspace camera sync
        controllers_dir = Path(__file__).resolve().parent.parent
        self.capture_state_path = controllers_dir / "capture_state.txt"
        self.workspace_output_dir = controllers_dir / "workspace_camera_logger" / "output"
        self.workspace_output_dir.mkdir(parents=True, exist_ok=True)

        # Cache and enable joint sensors (defensive: enable even if already on)
        self.joint_sensors: List[Any] = []
        timestep_ms = int(self.robot.getBasicTimeStep()) if hasattr(self.robot, "getBasicTimeStep") else 32
        for i in range(7):
            sensor = self.robot.getDevice(f"panda_joint{i+1}_sensor")
            if sensor:
                sensor.enable(timestep_ms)
                self.joint_sensors.append(sensor)
            else:
                self.joint_sensors.append(None)

        # Enable gripper finger sensor if available
        self.finger_sensor = self.robot.getDevice("panda_finger::right_sensor")
        if self.finger_sensor:
            self.finger_sensor.enable(timestep_ms)
        
        print(f"[DatasetRecorder] Initialized. Root: {self.dataset_root}")
    
    def add_tracked_object(self, name: str, node: Node) -> None:
        """Add object to track in state logs."""
        self.tracked_objects[name] = node
        print(f"[DatasetRecorder] Tracking object: {name}")
    
    def _get_next_episode_id(self) -> int:
        """Find next available episode ID."""
        existing = [
            int(d.name.split('_')[1]) 
            for d in self.dataset_root.iterdir() 
            if d.is_dir() and d.name.startswith('episode_')
        ]
        return max(existing, default=0) + 1
    
    def _create_episode_directory(self, episode_id: int) -> Path:
        """Create episode directory structure."""
        ep_dir = self.dataset_root / f"episode_{episode_id:04d}"
        ep_dir.mkdir(exist_ok=True)
        (ep_dir / "frames").mkdir(exist_ok=True)
        return ep_dir
    
    def start_episode(self, task: Optional[str] = None, difficulty: Optional[str] = None) -> None:
        """Start recording a new episode."""
        if self.recording:
            print("[DatasetRecorder] WARNING: Already recording. Call end_episode() first.")
            return

        # Prompt for task if not provided using task_prompt.txt
        if not task:
            prompt_file = Path(__file__).parent / "task_prompt.txt"
            print("\n" + "="*50)
            print("PAUSED: Waiting for task command")
            print(f"Edit {prompt_file.name} and save to continue...")
            print("="*50 + "\n", flush=True)
            
            # Clear any existing prompt file
            prompt_file.write_text("")
            
            # Wait for user to write task
            while True:
                self.robot.step(int(self.robot.getBasicTimeStep()))
                if prompt_file.exists():
                    task = prompt_file.read_text().strip()
                    if task:
                        prompt_file.unlink()  # Delete after reading
                        break
        
        self.current_task = task

        self.episode_id = self._get_next_episode_id()
        self.episode_dir = self._create_episode_directory(self.episode_id)
        self.current_step = 0
        
        # Clear buffers
        self.states_buffer.clear()
        self.actions_buffer.clear()
        self.timestamps_buffer.clear()
        
        # Reset previous state
        self.prev_ee_pose = None
        self.prev_gripper = None
        
        # Set task metadata
        if difficulty:
            self.current_difficulty = difficulty
        
        # Clean up old workspace camera frames before starting new episode
        for old_ppm in self.workspace_output_dir.glob("frame_*.ppm"):
            old_ppm.unlink()
        
        self.recording = True

        # Signal workspace camera logger to start capturing
        self._write_capture_state(capturing=True, frame_id=0)
        
        print(f"[DatasetRecorder] Started episode {self.episode_id:04d}")
        print(f"  Task: {self.current_task}")
        print(f"  Directory: {self.episode_dir}")
    
    def _get_joint_positions(self) -> List[float]:
        """Get current joint positions."""
        joints = []
        for sensor in self.joint_sensors:
            if sensor:
                joints.append(sensor.getValue())
            else:
                joints.append(0.0)
        return joints
    
    def _get_ee_pose(self) -> List[float]:
        """
        Get end-effector pose [x, y, z, qx, qy, qz, qw].
        Uses forward kinematics from joint positions.
        """
        # Use FK fallback only (avoid Supervisor APIs in Robot controller)
        joints = self._get_joint_positions()
        return self._fk_to_pose(joints)
    
    def _rotation_matrix_to_quaternion(self, R: List[float]) -> List[float]:
        """Convert 3x3 rotation matrix to quaternion [qx, qy, qz, qw]."""
        # R is flattened row-major: [R00, R01, R02, R10, R11, R12, R20, R21, R22]
        trace = R[0] + R[4] + R[8]
        
        if trace > 0:
            s = 0.5 / np.sqrt(trace + 1.0)
            qw = 0.25 / s
            qx = (R[7] - R[5]) * s
            qy = (R[2] - R[6]) * s
            qz = (R[3] - R[1]) * s
        elif R[0] > R[4] and R[0] > R[8]:
            s = 2.0 * np.sqrt(1.0 + R[0] - R[4] - R[8])
            qw = (R[7] - R[5]) / s
            qx = 0.25 * s
            qy = (R[1] + R[3]) / s
            qz = (R[2] + R[6]) / s
        elif R[4] > R[8]:
            s = 2.0 * np.sqrt(1.0 + R[4] - R[0] - R[8])
            qw = (R[2] - R[6]) / s
            qx = (R[1] + R[3]) / s
            qy = 0.25 * s
            qz = (R[5] + R[7]) / s
        else:
            s = 2.0 * np.sqrt(1.0 + R[8] - R[0] - R[4])
            qw = (R[3] - R[1]) / s
            qx = (R[2] + R[6]) / s
            qy = (R[5] + R[7]) / s
            qz = 0.25 * s
        
        return [qx, qy, qz, qw]
    
    def _fk_to_pose(self, joints: List[float]) -> List[float]:
        """Compute forward kinematics using DH parameters."""
        # Franka Panda DH parameters
        a = [0.0, 0.0, 0.0, 0.0825, -0.0825, 0.0, 0.088]
        d = [0.333, 0.0, 0.316, 0.0, 0.384, 0.0, 0.107]
        alpha = [-np.pi/2, np.pi/2, np.pi/2, np.pi/2, -np.pi/2, np.pi/2, 0.0]
        
        # Build transformation matrix
        T = np.eye(4)
        for i in range(7):
            cos_q = np.cos(joints[i])
            sin_q = np.sin(joints[i])
            cos_a = np.cos(alpha[i])
            sin_a = np.sin(alpha[i])
            
            # DH transformation matrix
            Ti = np.array([
                [cos_q, -sin_q*cos_a,  sin_q*sin_a, a[i]*cos_q],
                [sin_q,  cos_q*cos_a, -cos_q*sin_a, a[i]*sin_q],
                [0,      sin_a,        cos_a,       d[i]],
                [0,      0,            0,           1]
            ])
            T = T @ Ti
        
        # Extract position
        pos = T[:3, 3].tolist()
        
        # Convert rotation matrix to quaternion
        R = T[:3, :3]
        trace = np.trace(R)
        
        if trace > 0:
            s = 0.5 / np.sqrt(trace + 1.0)
            qw = 0.25 / s
            qx = (R[2, 1] - R[1, 2]) * s
            qy = (R[0, 2] - R[2, 0]) * s
            qz = (R[1, 0] - R[0, 1]) * s
        elif R[0, 0] > R[1, 1] and R[0, 0] > R[2, 2]:
            s = 2.0 * np.sqrt(1.0 + R[0, 0] - R[1, 1] - R[2, 2])
            qw = (R[2, 1] - R[1, 2]) / s
            qx = 0.25 * s
            qy = (R[0, 1] + R[1, 0]) / s
            qz = (R[0, 2] + R[2, 0]) / s
        elif R[1, 1] > R[2, 2]:
            s = 2.0 * np.sqrt(1.0 + R[1, 1] - R[0, 0] - R[2, 2])
            qw = (R[0, 2] - R[2, 0]) / s
            qx = (R[0, 1] + R[1, 0]) / s
            qy = 0.25 * s
            qz = (R[1, 2] + R[2, 1]) / s
        else:
            s = 2.0 * np.sqrt(1.0 + R[2, 2] - R[0, 0] - R[1, 1])
            qw = (R[1, 0] - R[0, 1]) / s
            qx = (R[0, 2] + R[2, 0]) / s
            qy = (R[1, 2] + R[2, 1]) / s
            qz = 0.25 * s
        
        return pos + [qx, qy, qz, qw]
    
    def _get_gripper_position(self) -> float:
        """Get gripper opening (0=closed, 0.04=fully open)."""
        if hasattr(self, "finger_sensor") and self.finger_sensor:
            return self.finger_sensor.getValue()
        return 0.02  # Default mid-position
    
    def _get_object_positions(self) -> Dict[str, List[float]]:
        """Get positions of tracked objects."""
        # Object tracking disabled (Supervisor-only APIs)
        return {}

    def _write_capture_state(self, capturing: bool, frame_id: int) -> None:
        """Notify workspace camera controller of capture state."""
        try:
            with open(self.capture_state_path, "w", encoding="utf-8") as f:
                f.write(f"capturing={1 if capturing else 0}\n")
                f.write(f"frame={frame_id}\n")
        except Exception as e:
            print(f"[DatasetRecorder] Failed to write capture_state: {e}")
    
    def _compute_action(self, current_ee: np.ndarray, current_gripper: float) -> Action:
        """
        Compute action as delta from previous state.
        Delta XYZ in meters, delta RPY in radians.
        """
        if self.prev_ee_pose is None:
            # First step: zero action
            delta_xyz = [0.0, 0.0, 0.0]
            delta_rpy = [0.0, 0.0, 0.0]
            gripper_cmd = 0
        else:
            # Position delta
            delta_xyz = (current_ee[:3] - self.prev_ee_pose[:3]).tolist()
            
            # Orientation delta (simplified: just use quaternion difference)
            # For proper delta_rpy, would need quaternion to euler conversion
            delta_rpy = [0.0, 0.0, 0.0]  # Placeholder
            
            # Gripper command: threshold at 0.03
            if current_gripper < 0.01:
                gripper_cmd = 1  # Closed
            elif current_gripper > 0.03:
                gripper_cmd = 0  # Open
            else:
                gripper_cmd = 1 if self.prev_gripper and self.prev_gripper < 0.02 else 0
        
        return Action(
            t=self.current_step,
            delta_xyz=delta_xyz,
            delta_rpy=delta_rpy,
            gripper_cmd=gripper_cmd
        )
    
    def _save_camera_images(self, frame_id: int) -> None:
        """Save synchronized camera images."""
        frames_dir = self.episode_dir / "frames"
        
        # Gripper camera
        grip_path = frames_dir / f"{frame_id:06d}_grip.png"
        self.gripper_camera.saveImage(str(grip_path), 100)
        
        # Workspace camera from external controller output (PPM -> PNG)
        work_path = frames_dir / f"{frame_id:06d}_work.png"
        ppm_path = self.workspace_output_dir / f"frame_{frame_id:06d}.ppm"

        # Request workspace frame
        self._write_capture_state(capturing=True, frame_id=frame_id)

        selected_ppm = None
        if ppm_path.exists():
            selected_ppm = ppm_path
        else:
            # Fallback: use latest available frame <= current id
            candidates = sorted(self.workspace_output_dir.glob("frame_*.ppm"))
            if candidates:
                selected_ppm = candidates[-1]

        if selected_ppm and selected_ppm.exists():
            try:
                loaded = False
                # Retry a few times to allow file to finish writing
                for _ in range(6):  # up to ~120ms total
                    with open(selected_ppm, "rb") as f:
                        header = f.readline().strip()  # P6
                        if not header.startswith(b"P6"):
                            raise ValueError("Not a P6 PPM")
                        dims = f.readline().strip()
                        maxv = f.readline().strip()
                        width, height = map(int, dims.split())
                        if maxv != b"255":
                            raise ValueError("Unsupported max value")
                        data = f.read()
                    expected_bytes = width * height * 3
                    if len(data) >= expected_bytes:
                        img = Image.frombytes("RGB", (width, height), data[:expected_bytes])
                        img.save(work_path)
                        loaded = True
                        break
                    time.sleep(0.02)
                if not loaded:
                    raise ValueError("not enough image data")
            except Exception as e:
                print(f"[DatasetRecorder] Workspace frame fallback to gripper: {e}")
                self.gripper_camera.saveImage(str(work_path), 100)
        else:
            # If workspace frame not ready yet, fallback to gripper image
            self.gripper_camera.saveImage(str(work_path), 100)
    
    def log_step(self) -> None:
        """Log current timestep data."""
        if not self.recording:
            return
        
        self.current_step += 1
        
        # Get robot state
        joints = self._get_joint_positions()
        ee_pose = self._get_ee_pose()
        gripper = self._get_gripper_position()
        objects = self._get_object_positions()
        
        state = RobotState(
            t=self.current_step,
            joints=joints,
            ee_pose=ee_pose,
            gripper=gripper,
            object_positions=objects
        )
        
        # Compute action
        ee_array = np.array(ee_pose)
        action = self._compute_action(ee_array, gripper)
        
        # Record timestamp
        timestamp = Timestamp(
            t=self.current_step,
            sim_time=self.robot.getTime(),
            wall_time=time.time()
        )
        
        # Buffer data
        self.states_buffer.append(state)
        self.actions_buffer.append(action)
        self.timestamps_buffer.append(timestamp)
        
        # Save images
        self._save_camera_images(self.current_step)
        
        # Update previous state
        self.prev_ee_pose = ee_array
        self.prev_gripper = gripper
        
        if self.current_step % 100 == 0:
            print(f"[DatasetRecorder] Logged step {self.current_step}")
    
    def _write_jsonl(self, filepath: Path, data: List[Any]) -> None:
        """Write list of dataclass objects to JSONL."""
        with open(filepath, 'w') as f:
            for item in data:
                json.dump(asdict(item), f)
                f.write('\n')
    
    def _write_json(self, filepath: Path, data: Any) -> None:
        """Write data to JSON file."""
        with open(filepath, 'w') as f:
            json.dump(asdict(data) if hasattr(data, '__dataclass_fields__') else data, f, indent=2)
    
    def end_episode(self, success: bool = True, notes: str = "") -> None:
        """End recording and save episode."""
        if not self.recording:
            print("[DatasetRecorder] WARNING: Not currently recording.")
            return
        
        print(f"[DatasetRecorder] Ending episode {self.episode_id:04d}...")
        
        # Write states
        self._write_jsonl(self.episode_dir / "states.jsonl", self.states_buffer)
        
        # Write actions
        self._write_jsonl(self.episode_dir / "actions.jsonl", self.actions_buffer)
        
        # Write timestamps
        self._write_jsonl(self.episode_dir / "timestamps.jsonl", self.timestamps_buffer)
        
        # Write metadata
        metadata = EpisodeMetadata(
            episode_id=self.episode_id,
            task=self.current_task,
            difficulty=self.current_difficulty,
            success=success,
            num_steps=self.current_step,
            operator=self.operator,
            date=datetime.now().strftime("%Y-%m-%d"),
            simulator="Webots",
            robot="Franka Panda",
            notes=notes
        )
        self._write_json(self.episode_dir / "meta.json", metadata)
        
        print(f"[DatasetRecorder] Saved episode {self.episode_id:04d}")
        print(f"  Steps: {self.current_step}")
        print(f"  Success: {success}")
        print(f"  Directory: {self.episode_dir}")
        
        self.recording = False
        self.episode_dir = None
        self.episode_id = None
        self.current_step = 0

        # Stop workspace camera capture
        self._write_capture_state(capturing=False, frame_id=-1)


class TeleopRecorderController:
    """
    Webots controller with keyboard teleoperation and dataset recording.
    
    Controls:
        R - Start recording episode
        S - Stop and save episode (success)
        F - Stop and save episode (failure)
        SPACE - Return to home
        Arrows/Z/X/A/S - Robot control (as defined in panda_arm_demo)
    """
    
    def __init__(self):
        self.robot = Supervisor()
        self.timestep = int(self.robot.getBasicTimeStep())
        
        # Initialize devices
        self._init_motors()
        self._init_sensors()
        self._init_cameras()
        
        # Dataset recorder
        self.recorder = DatasetRecorder(
            robot=self.robot,
            gripper_camera=self.wrist_camera,
            workspace_camera=self.workspace_camera,
            dataset_root="../../../dataset",
            operator="jame"
        )
        
        # Add tracked objects (if available)
        self._setup_object_tracking()
        
        # Control state
        self.recording_active = False
        self.last_r_press = -1.0
        self.last_s_press = -1.0
        
        print("[TeleopRecorder] Controller initialized")
        print("  Press R to start recording")
        print("  Press S to save (success) or F to save (failure)")
    
    def _init_motors(self):
        """Initialize motor devices."""
        self.motors = []
        for i in range(7):
            motor = self.robot.getDevice(f"panda_joint{i+1}")
            motor.setVelocity(1.0)
            self.motors.append(motor)
        
        self.gripper_motor = self.robot.getDevice("panda_finger::right")
    
    def _init_sensors(self):
        """Initialize position sensors."""
        self.sensors = []
        for i in range(7):
            sensor = self.robot.getDevice(f"panda_joint{i+1}_sensor")
            sensor.enable(self.timestep)
            self.sensors.append(sensor)
        
        # Add gripper sensor if exists
        try:
            self.gripper_sensor = self.robot.getDevice("panda_finger::right_sensor")
            if self.gripper_sensor:
                self.gripper_sensor.enable(self.timestep)
        except:
            self.gripper_sensor = None
    
    def _init_cameras(self):
        """Initialize cameras."""
        self.wrist_camera = self.robot.getDevice("wrist_camera")
        self.wrist_camera.enable(self.timestep)
        
        # Workspace camera (from separate robot)
        self.workspace_camera = self.robot.getDevice("workspace_camera")
        if self.workspace_camera:
            self.workspace_camera.enable(self.timestep)
        else:
            # Try to get from another robot node
            print("[TeleopRecorder] WARNING: workspace_camera not found in main robot")
    
    def _setup_object_tracking(self):
        """Setup object tracking for state logging."""
        # Try to find common objects in scene
        objects_to_track = [
            "RED_CUBE",
            "BLUE_CUBE", 
            "BOX",
            "TARGET_BOX"
        ]
        
        for obj_name in objects_to_track:
            node = self.robot.getFromDef(obj_name)
            if node:
                self.recorder.add_tracked_object(obj_name.lower(), node)
    
    def run(self):
        """Main control loop."""
        self.robot.keyboardEnable(self.timestep)
        
        while self.robot.step(self.timestep) != -1:
            self._handle_keyboard()
            
            # Log step if recording
            if self.recording_active:
                self.recorder.log_step()
    
    def _handle_keyboard(self):
        """Process keyboard input."""
        key = self.robot.getKeyboard()
        current_time = self.robot.getTime()
        
        # Recording controls
        if key == ord('R'):
            if current_time - self.last_r_press > 0.5:
                if not self.recording_active:
                    self.recorder.start_episode()
                    self.recording_active = True
                self.last_r_press = current_time
        
        elif key == ord('S'):
            if current_time - self.last_s_press > 0.5:
                if self.recording_active:
                    self.recorder.end_episode(success=True)
                    self.recording_active = False
                self.last_s_press = current_time
        
        elif key == ord('F'):
            if current_time - self.last_s_press > 0.5:
                if self.recording_active:
                    self.recorder.end_episode(success=False)
                    self.recording_active = False
                self.last_s_press = current_time
        
        # TODO: Add robot control handling (arrows, etc.)
        # This would integrate with your existing panda_arm_demo controller


def main():
    """Entry point."""
    controller = TeleopRecorderController()
    controller.run()


if __name__ == "__main__":
    main()
