#!/Users/srisanvi/miniforge/bin/python
"""
Franka Panda Teleoperation with Dataset Recording
Combines robot control with synchronized multi-modal data recording
"""

import numpy as np
from controller import Supervisor, Camera
import time
import sys
from pathlib import Path
import torch

from recorder import DatasetRecorder

# Check if running in model mode
MODEL_MODE = '--model' in sys.argv
if MODEL_MODE:
    from model_policy import ModelPolicy


class PandaTeleopRecorder:
    """
    Panda robot teleoperation controller with integrated dataset recording
    """
    
    def __init__(self):
        self.robot = Supervisor()
        self.timestep = int(self.robot.getBasicTimeStep())
        
        # Joint limits
        self.joint_min = [-2.8973, -1.7628, -2.8973, -3.0718, -2.8973, -0.0175, -2.8973]
        self.joint_max = [2.8973, 1.7628, 2.8973, -0.0698, 2.8973, 3.7525, 2.8973]
        
        # Initialize motors and sensors
        self.motors = []
        self.sensors = []
        motor_names = [
            "panda_joint1", "panda_joint2", "panda_joint3", "panda_joint4",
            "panda_joint5", "panda_joint6", "panda_joint7", "panda_finger::left"
        ]
        sensor_names = [
            "panda_joint1_sensor", "panda_joint2_sensor", "panda_joint3_sensor",
            "panda_joint4_sensor", "panda_joint5_sensor", "panda_joint6_sensor",
            "panda_joint7_sensor"
        ]
        
        for idx, name in enumerate(motor_names):
            motor = self.robot.getDevice(name)
            # Limit finger max velocity to avoid warnings
            if idx == 7:
                motor.setVelocity(0.2)
            else:
                motor.setVelocity(1.0)
            self.motors.append(motor)
        
        for name in sensor_names:
            sensor = self.robot.getDevice(name)
            sensor.enable(self.timestep)
            self.sensors.append(sensor)
        
        # Enable keyboard
        self.keyboard = self.robot.getKeyboard()
        self.keyboard.enable(self.timestep)
        
        # Enable cameras
        self.wrist_camera = self.robot.getDevice("wrist_camera")
        self.wrist_camera.enable(self.timestep)
        
        # Use wrist camera for both views for now (workspace camera needs separate Robot node)
        self.workspace_camera = self.wrist_camera
        
        # Control state
        self.wrist_pitch_target = 1.571  # Joint 6 target
        self.locked_joint5 = 0.0
        self.locked_joint7 = 0.785
        self.gripper_target = 0.04  # Open
        
        # Home position
        self.home_position = [0.0, -0.785, 0.0, -2.356, 0.0, 1.571, 0.785]
        self.returning_home = False
        self.home_start_time = 0.0
        
        # Control parameters (quadrupled teleop speed from original, homing unchanged)
        self.ee_vel = 0.32  # end-effector velocity (m/s)
        self.wrist_step = 0.02  # Wrist pitch step
        self.ik_damping = 0.05  # damping
        self.max_joint_vel = 2.0  # max joint velocity (rad/s) for teleop
        self.desired_ee_vel = np.zeros(3)  # Current desired velocity
        
        # Initialize dataset recorder
        self.recorder = DatasetRecorder(
            robot=self.robot,
            gripper_camera=self.wrist_camera,
            workspace_camera=self.workspace_camera,
            dataset_root='dataset',
            operator='jame'
        )

        # Task entry state
        self.episode_counter = 0
        self.current_difficulty = "medium"
        
        # Model policy (if enabled)
        self.model_policy = None
        if MODEL_MODE:
            checkpoint_path = Path(__file__).parent / "checkpoints" / "best_model.pth"
            if checkpoint_path.exists():
                # Use MPS if available (Apple Silicon GPU), otherwise CPU
                device = 'mps' if torch.backends.mps.is_available() else 'cpu'
                self.model_policy = ModelPolicy(str(checkpoint_path), device=device)
                print("\n🤖 MODEL POLICY MODE ENABLED")
                print("   Press SPACE to run one autonomous episode\n")
            else:
                print(f"\n⚠️  Model not found: {checkpoint_path}")
                print("   Falling back to teleoperation mode\n")
        
        # Track objects (add your scene objects here)
        self._setup_tracked_objects()
        
        # Recording state
        self.is_recording = False
        self.current_task = "pick and place task"
        self.current_difficulty = "medium"
        
        print("\n=== Panda Teleoperation with Dataset Recording ===")
        print("\nRobot Controls:")
        print("  Arrow Keys    - Move in X/Y plane")
        print("  Z/X           - Move up/down")
        print("  A/S           - Wrist pitch up/down")
        print("  O/P           - Gripper open/close")
        print("  SPACE         - Return to home position")
        print("\nRecording Controls:")
        print("  R             - Start recording episode")
        print("  T (while R)   - Stop and save (success)")
        print("  F (while R)   - Stop and save (failure)")
        print("  Q/ESC         - Quit")
        print("=" * 50 + "\n")
    
    def _setup_tracked_objects(self):
        """Setup object tracking for dataset recording"""
        # Add DEF names from your world file here
        # Example:
        # try:
    def _setup_tracked_objects(self):
        """Object tracking disabled for now (requires Supervisor mode fixes)"""
        # You can add objects later when needed
        pass
    
    def get_joint_positions(self):
        """Read current joint positions"""
        return np.array([s.getValue() for s in self.sensors])
    
    def dh_transform(self, a, alpha, d, theta):
        """DH transformation matrix"""
        ct = np.cos(theta)
        st = np.sin(theta)
        ca = np.cos(alpha)
        sa = np.sin(alpha)
        
        return np.array([
            [ct, -st * ca,  st * sa, a * ct],
            [st,  ct * ca, -ct * sa, a * st],
            [0,   sa,       ca,      d],
            [0,   0,        0,       1]
        ])
    
    def forward_kinematics(self, q):
        """
        Full FK using DH parameters (exact match to C code)
        Returns end-effector transformation matrix
        """
        # Franka Panda DH parameters (exact from C code)
        a = [0.0, 0.0, 0.0, 0.0825, -0.0825, 0.0, 0.088]
        d = [0.333, 0.0, 0.316, 0.0, 0.384, 0.0, 0.107]
        alpha = [-np.pi/2, np.pi/2, np.pi/2, np.pi/2, -np.pi/2, np.pi/2, 0.0]
        
        T = np.eye(4)
        for i in range(7):
            T = T @ self.dh_transform(a[i], alpha[i], d[i], q[i])
        
        return T
    
    def get_ee_pose_7d(self, q):
        """Get EE pose as [x, y, z, qx, qy, qz, qw] from joint positions."""
        T = self.forward_kinematics(q)
        pos = T[:3, 3]
        
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
        
        return np.concatenate([pos, [qx, qy, qz, qw]])
    
    def jacobian_4dof(self, q):
        """
        Compute 3x4 Jacobian for position control using first 4 joints (exact from C)
        """
        eps = 1e-4  # Same as C code
        J = np.zeros((3, 4))
        
        T0 = self.forward_kinematics(q)
        p0 = T0[:3, 3]
        
        for i in range(4):  # Only joints 0-3
            q_pert = q.copy()
            q_pert[i] += eps
            T1 = self.forward_kinematics(q_pert)
            p1 = T1[:3, 3]
            J[:, i] = (p1 - p0) / eps
        
        return J
    
    def inverse_kinematics_velocity(self, desired_vel):
        """
        Velocity-based IK for 4 DOF (exact match to C code)
        Takes desired EE velocity, returns joint velocities
        """
        # Get current joint positions
        q = self.get_joint_positions()
        
        # Compute 3x4 Jacobian
        J = self.jacobian_4dof(q)
        
        # Compute J * J^T (3x3) with damping
        JJt = J @ J.T
        JJt += (self.ik_damping ** 2) * np.eye(3)
        
        # Invert 3x3 matrix
        try:
            JJt_inv = np.linalg.inv(JJt)
        except:
            return np.zeros(4)
        
        # Compute J^T * (J*J^T)^-1 * desired_vel
        temp = JJt_inv @ desired_vel
        joint_vels = J.T @ temp
        
        # Clamp joint velocities
        max_vel = 0.5  # Same as C code
        joint_vels = np.clip(joint_vels, -max_vel, max_vel)
        
        return joint_vels
    
    def clamp(self, value, min_val, max_val):
        """Clamp value to range"""
        return max(min_val, min(max_val, value))
    
    def get_model_action(self, task=""):
        """Get action from trained model."""
        # Capture current images
        grip_img = np.frombuffer(self.wrist_camera.getImage(), dtype=np.uint8)
        grip_img = grip_img.reshape((self.wrist_camera.getHeight(), 
                                     self.wrist_camera.getWidth(), 4))[:, :, :3]
        
        work_img = np.frombuffer(self.workspace_camera.getImage(), dtype=np.uint8)
        work_img = work_img.reshape((self.workspace_camera.getHeight(),
                                     self.workspace_camera.getWidth(), 4))[:, :, :3]
        
        # Get robot state - match recorder format exactly
        q = self.get_joint_positions()  # 7 joints
        ee_pose = self.get_ee_pose_7d(q)  # [x, y, z, qx, qy, qz, qw] = 7D
        gripper_pos = self.gripper_target  # Use current target
        
        # Build state vector [joints(7), ee_pose(7), gripper(1)] = 15D
        state = np.concatenate([q, ee_pose, [gripper_pos]])
        
        # Get action from model with task
        action = self.model_policy.predict_action(grip_img, work_img, state, task=task)
        
        # action format: [delta_x, delta_y, delta_z, gripper_cmd, pad, pad, pad]
        return action[:4]  # Return [delta_xyz, gripper_cmd]
    
    def handle_keyboard(self):
        """Process keyboard input"""
        # Reset desired velocity each cycle
        self.desired_ee_vel = np.zeros(3)
        key = self.keyboard.getKey()
        
        # Process all queued keys this step (like the C controller)
        while key != -1:
            # Recording controls
            if key == ord('R'):
                if not self.is_recording:
                    self.recorder.start_episode(difficulty=self.current_difficulty)
                    self.is_recording = True
                    print(f"\n🔴 RECORDING: {self.recorder.current_task}")
                    print("Press T (success) or F (failure) to stop\n")
            elif key == ord('T'):
                if self.is_recording:
                    self.recorder.end_episode(success=True, notes="")
                    self.is_recording = False
                    print("\n✅ Episode saved (SUCCESS)\n")
            elif key == ord('F'):
                if self.is_recording:
                    self.recorder.end_episode(success=False, notes="")
                    self.is_recording = False
                    print("\n❌ Episode saved (FAILURE)\n")
            
            # Robot controls (only when not returning home)
            if not self.returning_home:
                # X/Y movement (arrows) - velocity control
                if key == self.keyboard.UP:
                    self.desired_ee_vel[0] = self.ee_vel  # Forward (X+)
                elif key == self.keyboard.DOWN:
                    self.desired_ee_vel[0] = -self.ee_vel  # Backward (X-)
                elif key == self.keyboard.LEFT:
                    self.desired_ee_vel[1] = -self.ee_vel  # Left (Y-)
                elif key == self.keyboard.RIGHT:
                    self.desired_ee_vel[1] = self.ee_vel  # Right (Y+)
                
                # Z movement - velocity control
                elif key == ord('Z'):
                    self.desired_ee_vel[2] = self.ee_vel  # Up
                elif key == ord('X'):
                    self.desired_ee_vel[2] = -self.ee_vel  # Down
                
                # Wrist pitch
                elif key == ord('A'):
                    self.wrist_pitch_target += self.wrist_step
                    self.wrist_pitch_target = self.clamp(
                        self.wrist_pitch_target,
                        self.joint_min[5],
                        self.joint_max[5]
                    )
                elif key == ord('S'):
                    self.wrist_pitch_target -= self.wrist_step
                    self.wrist_pitch_target = self.clamp(
                        self.wrist_pitch_target,
                        self.joint_min[5],
                        self.joint_max[5]
                    )
                
                # Gripper
                elif key == ord('O'):
                    self.gripper_target = 0.04  # Open
                elif key == ord('P'):
                    self.gripper_target = 0.0  # Close
            
            # Home position / Model execution
            if key == ord(' '):  # SPACE
                if self.model_policy and not self.is_recording:
                    # Run autonomous episode with model
                    print("\n🤖 Starting autonomous episode with model...")
                    self.run_model_episode()
                elif not self.returning_home:
                    print("Returning to home position...")
                    self.returning_home = True
                    self.home_start_time = self.robot.getTime()
                    # Slower return home speed
                    for idx, m in enumerate(self.motors):
                        m.setVelocity(0.2 if idx == 7 else 0.35)
                    
                    # Set all joints to home
                    for i in range(7):
                        self.motors[i].setPosition(self.home_position[i])
            
            # Quit
            if key == ord('Q') or key == 27:  # ESC
                if self.is_recording:
                    print("\nWarning: Still recording! Saving episode...")
                    self.recorder.end_episode(success=False, notes="Interrupted")
                return False

            key = self.keyboard.getKey()
        
        return True
    
    def update_motors(self):
        """Update motor positions based on IK (exact match to C code)"""
        # Check if returning home is complete
        if self.returning_home:
            if self.robot.getTime() - self.home_start_time > 3.0:
                self.returning_home = False
                print("Home position reached")
                self.wrist_pitch_target = self.home_position[5]
                # Restore nominal velocity limits
                for idx, m in enumerate(self.motors):
                    m.setVelocity(0.2 if idx == 7 else self.max_joint_vel)
            return
        
        # Get current joint positions
        q = self.get_joint_positions()
        
        # Compute joint velocities from desired EE velocity
        joint_vels = self.inverse_kinematics_velocity(self.desired_ee_vel)
        
        # Set motor velocities and compute target positions
        dt = self.timestep / 1000.0  # Convert ms to seconds
        
        for i in range(7):
            # Set velocity limit (finger motor capped at 0.2)
            if i == 7:
                self.motors[i].setVelocity(0.2)
            else:
                self.motors[i].setVelocity(self.max_joint_vel)
            
            if i < 4:
                # Joints 0-3: use IK velocities
                new_pos = q[i] + joint_vels[i] * dt
                new_pos = self.clamp(new_pos, self.joint_min[i], self.joint_max[i])
            elif i == 4:
                # Joint 4 (index 4): locked
                new_pos = self.locked_joint5
            elif i == 5:
                # Joint 5 (index 5): manual wrist pitch control
                new_pos = self.wrist_pitch_target
            else:
                # Joint 6 (index 6): locked
                new_pos = self.locked_joint7
            
            self.motors[i].setPosition(new_pos)
        
        # Gripper
        self.motors[7].setPosition(self.gripper_target)
    
    def run(self):
        """Main control loop"""
        # Step twice to ensure sensors are fully initialized
        self.robot.step(self.timestep)
        self.robot.step(self.timestep)
        
        # Store initial joint positions as home
        q_init = self.get_joint_positions()
        self.home_position = q_init.tolist()
        self.locked_joint5 = q_init[4]
        self.locked_joint7 = q_init[6]
        self.wrist_pitch_target = q_init[5]
        
        # Main loop
        while self.robot.step(self.timestep) != -1:
            # Process keyboard
            if not self.handle_keyboard():
                break
            
            # Update motor commands
            self.update_motors()
            
            # Record data if recording
            if self.is_recording:
                self.recorder.log_step()
        
        # Cleanup
        if self.is_recording:
            print("\nSaving current episode...")
            self.recorder.end_episode(success=False, notes="Simulation ended")
    
    def run_model_episode(self):
        """Run one autonomous episode using the trained model."""
        # Prompt for task if model uses language
        if self.model_policy.use_language:
            task_file = Path(__file__).parent / "model_task.txt"
            print("\n" + "="*50)
            print("TASK REQUIRED")
            print(f"Edit {task_file.name} with your task command and save")
            print("Waiting for task input...")
            print("="*50 + "\n")
            
            # Create blank task file
            task_file.write_text("")
            
            # Wait for user to write and save task
            while True:
                if self.robot.step(self.timestep) == -1:
                    return
                
                # Check if file has been modified with content
                if task_file.exists():
                    content = task_file.read_text().strip()
                    if content:
                        self.current_task = content
                        print(f"[Model] Task received: {self.current_task}\n")
                        break
        else:
            self.current_task = ""
        
        print("="*50)
        print("AUTONOMOUS EXECUTION")
        print("="*50)
        print("Press Q to stop early\n")
        
        max_steps = 3000  # Increased to allow full episode completion
        step_count = 0
        
        while step_count < max_steps:
            # Check for early stop
            key = self.keyboard.getKey()
            if key == ord('Q') or key == 27:
                print("\n⏹️  Stopped by user")
                break
            
            # Get action from model (now with task)
            action = self.get_model_action(self.current_task)
            delta_xyz = action[:3]
            gripper_cmd = action[3]
            
            # Scale delta to velocity (delta is per timestep, convert to m/s)
            # During recording: delta ~= vel * timestep, so vel ~= delta / timestep
            self.desired_ee_vel = delta_xyz / (self.timestep / 1000.0)  # timestep is in ms
            self.gripper_target = 0.0 if gripper_cmd > 0.5 else 0.04
            
            # Update motors
            self.update_motors()
            
            # Step simulation
            if self.robot.step(self.timestep) == -1:
                break
            
            step_count += 1
            
            if step_count % 100 == 0:
                print(f"Step {step_count}/{max_steps}")
        
        print(f"\n✅ Completed {step_count} steps\n")
        
        # Return to home
        print("Returning to home position...")
        self.returning_home = True
        for i in range(7):
            self.motors[i].setPosition(self.home_position[i])
        
        # Wait for home
        for _ in range(100):
            if self.robot.step(self.timestep) == -1:
                break
        
        self.returning_home = False


if __name__ == "__main__":
    controller = PandaTeleopRecorder()
    controller.run()
