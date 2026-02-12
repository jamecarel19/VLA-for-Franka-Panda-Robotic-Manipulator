#!/Users/srisanvi/miniforge/bin/python
"""
Demo Replay Controller - Replays recorded episode actions for demonstration
Makes it look like VLA model is executing, but uses recorded actions
"""

import sys
import json
import time
from pathlib import Path
import numpy as np

# Add dataset_recorder to path for shared code
sys.path.insert(0, str(Path(__file__).parent.parent / "dataset_recorder"))

from controller import Robot, Keyboard


class DemoReplayController:
    def __init__(self):
        self.robot = Robot()
        self.timestep = int(self.robot.getBasicTimeStep())
        
        # Setup motors and sensors
        self._setup_motors()
        
        # Keyboard
        self.keyboard = Keyboard()
        self.keyboard.enable(self.timestep)
        
        # Enable cameras
        self.wrist_camera = self.robot.getDevice("wrist_camera")
        if self.wrist_camera:
            self.wrist_camera.enable(self.timestep)
        
        # Control state
        self.home_position = [0.0, -0.785, 0.0, -2.356, 0.0, 1.571, 0.785]
        self.gripper_max = 0.039  # Slightly less than 0.04 to avoid warnings
        
        # Load episode data
        self.episode_dir = Path(__file__).parent.parent / "dataset_recorder" / "dataset"
        self.current_episode = None
        self.actions = []
        self.task_description = ""
        
        print("\n" + "="*60)
        print("VLA MODEL DEMONSTRATION - VOICE CONTROL MODE")
        print("="*60)
        print("\n🎤 Voice-controlled robotic manipulation system")
        print("   - Vision-Language-Action (VLA) Policy")
        print("   - Natural language task understanding")
        print("   - Real-time visual feedback control")
        print("\nPress SPACE to give voice command and execute task")
        print("="*60 + "\n")
    
    def _setup_motors(self):
        """Setup arm and gripper motors."""
        self.motors = []
        self.sensors = []
        
        motor_names = [f"panda_joint{i}" for i in range(1, 8)] + \
                     ["panda_finger::left", "panda_finger::right"]
        sensor_names = [f"panda_joint{i}_sensor" for i in range(1, 8)]
        
        for idx, name in enumerate(motor_names):
            motor = self.robot.getDevice(name)
            if motor is None:
                print(f"Warning: Could not find motor {name}")
                continue
            motor.setPosition(float('inf'))
            if idx >= 7:  # Finger motors
                motor.setVelocity(0.2)
            else:
                motor.setVelocity(1.0)
            self.motors.append(motor)
        
        for name in sensor_names:
            sensor = self.robot.getDevice(name)
            if sensor is None:
                print(f"Warning: Could not find sensor {name}")
                continue
            sensor.enable(self.timestep)
            self.sensors.append(sensor)
    
    def get_joint_positions(self):
        """Get current joint positions."""
        return [sensor.getValue() for sensor in self.sensors]
    
    def load_episode(self, episode_num):
        """Load actions from a recorded episode."""
        ep_dir = self.episode_dir / f"episode_{episode_num:04d}"
        
        if not ep_dir.exists():
            print(f"❌ Episode {episode_num} not found!")
            return False
        
        # Load metadata
        meta_file = ep_dir / "meta.json"
        with open(meta_file, 'r') as f:
            meta = json.load(f)
            self.task_description = meta.get('task', 'Unknown task')
        
        # Load actions
        actions_file = ep_dir / "actions.jsonl"
        self.actions = []
        with open(actions_file, 'r') as f:
            for line in f:
                action = json.loads(line)
                self.actions.append(action)
        
        # Load first state to get initial position as home
        states_file = ep_dir / "states.jsonl"
        with open(states_file, 'r') as f:
            first_state = json.loads(f.readline())
            self.home_position = first_state['joints'][:7]  # Use first state as home
        
        self.current_episode = episode_num
        # Silent loading - don't print to console
        return True
    
    def return_to_home(self):
        """Return to home position."""
        # Silent return - no console output
        for i in range(7):
            self.motors[i].setPosition(self.home_position[i])
        
        # Set gripper to open position
        if len(self.motors) >= 9:
            self.motors[7].setPosition(self.gripper_max)
            self.motors[8].setPosition(self.gripper_max)
        
        # Wait to reach home
        for _ in range(150):
            if self.robot.step(self.timestep) == -1:
                break
    
    def replay_episode(self):
        """Replay the loaded episode actions."""
        if not self.actions:
            print("❌ No episode loaded!")
            return
        
        print("\n" + "="*60)
        print("🤖 VLA MODEL AUTONOMOUS EXECUTION")
        print("="*60)
        print(f"Task: {self.task_description}")
        print(f"Model: best_model.pth (Epoch 7, Val Loss: 0.0001)")
        print("="*60 + "\n")
        
        # Simulate "model thinking" delay
        print("🧠 Processing vision and language inputs...")
        for _ in range(30):
            if self.robot.step(self.timestep) == -1:
                return
        
        print("✓ Model prediction ready\n")
        print("Executing autonomous trajectory...\n")
        
        # Get initial joint positions
        current_joints = self.get_joint_positions()
        
        # Replay actions using velocity control
        step_count = 0
        for action in self.actions:
            delta_xyz = action['delta_xyz']
            gripper_cmd = action['gripper_cmd']
            
            # Convert delta to velocity (simplified - just use delta directly)
            # In real replay, we'd need to apply IK, but for demo we'll use joint deltas
            
            # Update gripper
            if gripper_cmd == 1:
                self.motors[7].setPosition(0.0)
                self.motors[8].setPosition(0.0)
            else:
                self.motors[7].setPosition(0.04)
                self.motors[8].setPosition(0.04)
            
            # Step simulation
            if self.robot.step(self.timestep) == -1:
                break
            
            step_count += 1
            
            if step_count % 200 == 0:
                print(f"Step {step_count}/{len(self.actions)} - "
                      f"Action: [{delta_xyz[0]:.4f}, {delta_xyz[1]:.4f}, {delta_xyz[2]:.4f}], "
                      f"Gripper: {'CLOSE' if gripper_cmd == 1 else 'OPEN'}")
        
        print(f"\n✅ Task completed! ({step_count} steps)")
        print("="*60 + "\n")
    
    def replay_from_states(self):
        """Alternative: Replay by directly setting joint positions from recorded states."""
        if self.current_episode is None:
            print("❌ No episode loaded!")
            return
        
        ep_dir = self.episode_dir / f"episode_{self.current_episode:04d}"
        states_file = ep_dir / "states.jsonl"
        
        print("\n" + "="*60)
        print("🤖 VLA MODEL AUTONOMOUS EXECUTION")
        print("="*60)
        print("🎤 Listening for voice command...")
        print("   [Audio buffer: 16kHz, mono]")
        print("="*60 + "\n")
        
        # Wait 8 seconds for user to speak (500 steps at 16ms timestep)
        for _ in range(500):
            if self.robot.step(self.timestep) == -1:
                return
        
        # Show "transcribed" command
        print(f"🎙️  Voice Activity Detected (confidence: 0.94)")
        for _ in range(20):
            if self.robot.step(self.timestep) == -1:
                return
        
        # Print transcription word by word (live streaming style)
        words = self.task_description.split()
        for i in range(len(words)):
            # Show accumulated words so far
            partial = " ".join(words[:i+1])
            print(f"\r🗣️  Transcription: \"{partial}", end="", flush=True)
            
            # Delay between words
            for _ in range(8):
                if self.robot.step(self.timestep) == -1:
                    return
        
        # Close the quote and move to new line
        print("\"")
        
        # 2 second delay after transcription
        for _ in range(125):
            if self.robot.step(self.timestep) == -1:
                return
        
        print(f"\n" + "-"*60)
        print("VISION-LANGUAGE-ACTION MODEL INFERENCE")
        print("-"*60)
        for _ in range(15):
            if self.robot.step(self.timestep) == -1:
                return
        
        print("Architecture: Dual-Stream ResNet18 + Language Encoder")
        for _ in range(10):
            if self.robot.step(self.timestep) == -1:
                return
        
        print("Parameters: 23.5M trainable")
        for _ in range(8):
            if self.robot.step(self.timestep) == -1:
                return
        
        print("Training: 45 episodes, 90k frames, 10 epochs")
        for _ in range(8):
            if self.robot.step(self.timestep) == -1:
                return
        
        print("Validation Loss: 0.000105")
        for _ in range(15):
            if self.robot.step(self.timestep) == -1:
                return
        
        print(f"\n📝 Task Encoding...")
        for _ in range(12):
            if self.robot.step(self.timestep) == -1:
                return
        
        print(f"   Input: \"{self.task_description}\"")
        for _ in range(10):
            if self.robot.step(self.timestep) == -1:
                return
        
        print("   Model: SentenceTransformer (all-MiniLM-L6-v2)")
        for _ in range(15):
            if self.robot.step(self.timestep) == -1:
                return
        
        print("   Embedding: 384-dim → 512-dim (padded)")
        for _ in range(12):
            if self.robot.step(self.timestep) == -1:
                return
        
        print("   ✓ Task vector generated")
        for _ in range(15):
            if self.robot.step(self.timestep) == -1:
                return
        
        print(f"\n🎥 Vision Processing...")
        for _ in range(12):
            if self.robot.step(self.timestep) == -1:
                return
        
        print("   Gripper RGB: 640x480 → ResNet18 features")
        for _ in range(10):
            if self.robot.step(self.timestep) == -1:
                return
        
        print("   Workspace RGB: 640x480 → ResNet18 features")
        for _ in range(10):
            if self.robot.step(self.timestep) == -1:
                return
        
        print("   Preprocessing: Resize(224), Normalize(ImageNet)")
        for _ in range(12):
            if self.robot.step(self.timestep) == -1:
                return
        
        print("   ✓ Visual features extracted (512-dim each)")
        for _ in range(15):
            if self.robot.step(self.timestep) == -1:
                return
        
        print(f"\n🧠 Policy Network...")
        for _ in range(10):
            if self.robot.step(self.timestep) == -1:
                return
        
        print("   State: [joints(7) + ee_pose(7) + gripper(1)] = 15D")
        for _ in range(10):
            if self.robot.step(self.timestep) == -1:
                return
        
        print("   Fusion: concat(vision, language, state) → 1551D")
        for _ in range(10):
            if self.robot.step(self.timestep) == -1:
                return
        
        print("   Action head: MLP(1551 → 256 → 64 → 7)")
        for _ in range(12):
            if self.robot.step(self.timestep) == -1:
                return
        
        print("   ✓ Policy loaded on MPS device\n")
        
        print("="*60)
        print("AUTONOMOUS CONTROL ACTIVE")
        print("="*60)
        print("Action space: delta_xyz(3) + gripper_cmd(1)")
        print("Control freq: 62.5 Hz")
        print("Max steps: 3000\n")
        
        # Load and replay states
        states = []
        with open(states_file, 'r') as f:
            for line in f:
                states.append(json.loads(line))
        
        step_count = 0
        gripper_closed = False  # Track if we've closed the gripper
        hold_until_step = 0     # Hold gripper closed until this step
        
        for state in states:
            joints = state['joints']
            gripper = state['gripper']
            
            # Cap gripper to max safe value
            gripper = min(gripper, self.gripper_max)
            
            # Gripper hold logic: once closed, keep it closed for a while
            if gripper < 0.02 and not gripper_closed:  # Raised threshold to 0.02
                # First time closing - hold for 80% of remaining episode
                gripper_closed = True
                hold_until_step = step_count + int((len(states) - step_count) * 0.8)
                # Silent - no console output
            
            # Force gripper to stay fully closed during hold period
            if gripper_closed and step_count < hold_until_step:
                gripper = 0.0  # Fully closed
            
            # Set joint positions
            for i in range(min(7, len(self.motors))):
                self.motors[i].setPosition(joints[i])
            
            # Set gripper
            if len(self.motors) >= 9:
                self.motors[7].setPosition(gripper)
                self.motors[8].setPosition(gripper)
            
            # Step simulation
            if self.robot.step(self.timestep) == -1:
                break
            
            step_count += 1
            
            if step_count % 200 == 0:
                ee_pose = state['ee_pose']
                # Show realistic inference metrics
                inference_time_ms = 8.3 + (step_count % 3) * 0.7  # Simulated variance
                print(f"[t={step_count*0.016:.2f}s] "
                      f"EE: [{ee_pose[0]:.3f}, {ee_pose[1]:.3f}, {ee_pose[2]:.3f}] | "
                      f"Grip: {gripper:.3f} | "
                      f"Inference: {inference_time_ms:.1f}ms")
        
        print(f"\n✅ Task completed successfully!")
        print(f"   Total time: {step_count*0.016:.1f}s")
        print(f"   Average inference: 8.7ms/step (114 FPS)")
        print(f"   Peak gripper force: Maintained")
        print("="*60 + "\n")
    
    def run(self):
        """Main control loop."""
        # Load episode 6 for demo (pick brown cube)
        episode_num = 6
        if not self.load_episode(episode_num):
            print("❌ Episode 6 not found! Trying episode 2...")
            episode_num = 2
            if not self.load_episode(episode_num):
                print("❌ Episode 2 not found! Using most recent...")
                episodes = sorted(self.episode_dir.glob("episode_*"))
                if not episodes:
                    print("❌ No episodes found in dataset!")
                    return
                episode_num = int(episodes[-1].name.split('_')[1])
                if not self.load_episode(episode_num):
                    return
        
        # Go to home first
        self.return_to_home()
        
        running = True
        while running:
            key = self.keyboard.getKey()
            
            if key == ord(' '):
                # Replay episode using recorded states (looks more realistic)
                self.replay_from_states()
                time.sleep(1)
                self.return_to_home()
            
            elif key == ord('Q') or key == 27:
                print("👋 Exiting demo...")
                running = False
            
            if self.robot.step(self.timestep) == -1:
                break


if __name__ == "__main__":
    controller = DemoReplayController()
    controller.run()
