"""
Dataset Episode Viewer

Visualizes recorded episodes with:
- Side-by-side camera views
- Overlayed action vectors
- Robot state display
- Playback controls
"""

import json
import sys
import cv2
import numpy as np
from pathlib import Path
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass


@dataclass
class ViewerState:
    """Viewer playback state."""
    current_frame: int
    playing: bool
    playback_speed: float
    show_actions: bool
    show_states: bool


class EpisodeViewer:
    """Interactive episode viewer."""
    
    def __init__(self, episode_dir: Path):
        self.episode_dir = episode_dir
        self.episode_id = int(episode_dir.name.split('_')[1])
        
        # Load episode data
        self._load_data()
        
        # Viewer state
        self.state = ViewerState(
            current_frame=0,
            playing=False,
            playback_speed=1.0,
            show_actions=True,
            show_states=True
        )
        
        # Display settings
        self.window_name = f"Episode {self.episode_id:04d} Viewer"
        self.frame_size = (640, 480)  # Resize to this
        
    def _load_data(self) -> None:
        """Load episode data."""
        # Load metadata
        with open(self.episode_dir / 'meta.json') as f:
            self.metadata = json.load(f)
        
        # Load JSONL
        self.states = self._load_jsonl(self.episode_dir / 'states.jsonl')
        self.actions = self._load_jsonl(self.episode_dir / 'actions.jsonl')
        self.timestamps = self._load_jsonl(self.episode_dir / 'timestamps.jsonl')
        
        # Get frame paths
        frames_dir = self.episode_dir / 'frames'
        self.grip_frames = sorted(frames_dir.glob('*_grip.png'))
        self.work_frames = sorted(frames_dir.glob('*_work.png'))
        
        self.num_frames = len(self.grip_frames)
        
        print(f"[Viewer] Loaded episode {self.episode_id:04d}")
        print(f"  Task: {self.metadata['task']}")
        print(f"  Steps: {self.num_frames}")
        print(f"  Success: {self.metadata['success']}")
    
    def _load_jsonl(self, filepath: Path) -> List[Dict]:
        """Load JSONL file."""
        data = []
        with open(filepath) as f:
            for line in f:
                data.append(json.loads(line.strip()))
        return data
    
    def _load_frame_pair(self, idx: int) -> Tuple[np.ndarray, np.ndarray]:
        """Load and prepare gripper and workspace frames."""
        if idx >= len(self.grip_frames) or idx >= len(self.work_frames):
            # Return black frames
            return np.zeros((480, 640, 3), dtype=np.uint8), np.zeros((480, 640, 3), dtype=np.uint8)
        
        # Load images
        grip_img = cv2.imread(str(self.grip_frames[idx]))
        work_img = cv2.imread(str(self.work_frames[idx]))
        
        # Resize
        grip_img = cv2.resize(grip_img, self.frame_size)
        work_img = cv2.resize(work_img, self.frame_size)
        
        return grip_img, work_img
    
    def _draw_action_overlay(self, img: np.ndarray, action: Dict) -> np.ndarray:
        """Draw action vector overlay on image."""
        img = img.copy()
        h, w = img.shape[:2]
        
        # Draw action vector from center
        cx, cy = w // 2, h // 2
        
        delta_xyz = action.get('delta_xyz', [0, 0, 0])
        
        # Scale for visualization (1cm = 50 pixels)
        scale = 5000.0
        dx = int(delta_xyz[0] * scale)
        dy = int(-delta_xyz[1] * scale)  # Flip Y for image coords
        
        # Draw arrow
        if abs(dx) > 1 or abs(dy) > 1:
            cv2.arrowedLine(
                img,
                (cx, cy),
                (cx + dx, cy + dy),
                (0, 255, 0),
                thickness=3,
                tipLength=0.3
            )
        
        # Draw gripper state
        gripper_cmd = action.get('gripper_cmd', 0)
        color = (0, 0, 255) if gripper_cmd == 1 else (255, 0, 0)
        text = "CLOSE" if gripper_cmd == 1 else "OPEN"
        cv2.putText(
            img,
            text,
            (10, h - 20),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            color,
            2
        )
        
        # Draw delta magnitude
        delta_mag = np.linalg.norm(delta_xyz)
        cv2.putText(
            img,
            f"Delta: {delta_mag*100:.1f}cm",
            (10, h - 50),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (255, 255, 255),
            1
        )
        
        return img
    
    def _draw_state_overlay(self, img: np.ndarray, state: Dict, timestamp: Dict) -> np.ndarray:
        """Draw state information on image."""
        img = img.copy()
        h, w = img.shape[:2]
        
        # EE position
        ee_pose = state.get('ee_pose', [0]*7)
        ee_text = f"EE: [{ee_pose[0]:.3f}, {ee_pose[1]:.3f}, {ee_pose[2]:.3f}]"
        cv2.putText(img, ee_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        
        # Gripper position
        gripper = state.get('gripper', 0.0)
        grip_text = f"Gripper: {gripper*1000:.1f}mm"
        cv2.putText(img, grip_text, (10, 55), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        
        # Timestamp
        sim_time = timestamp.get('sim_time', 0.0)
        time_text = f"Time: {sim_time:.2f}s"
        cv2.putText(img, time_text, (10, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        
        return img
    
    def _create_display(self, idx: int) -> np.ndarray:
        """Create combined display for current frame."""
        # Load frames
        grip_img, work_img = self._load_frame_pair(idx)
        
        # Get data for this frame
        if idx < len(self.actions):
            action = self.actions[idx]
        else:
            action = {'delta_xyz': [0, 0, 0], 'gripper_cmd': 0}
        
        if idx < len(self.states):
            state = self.states[idx]
        else:
            state = {}
        
        if idx < len(self.timestamps):
            timestamp = self.timestamps[idx]
        else:
            timestamp = {}
        
        # Apply overlays
        if self.state.show_actions:
            grip_img = self._draw_action_overlay(grip_img, action)
        
        if self.state.show_states:
            work_img = self._draw_state_overlay(work_img, state, timestamp)
        
        # Combine side-by-side
        combined = np.hstack([grip_img, work_img])
        
        # Add header
        header_height = 80
        header = np.zeros((header_height, combined.shape[1], 3), dtype=np.uint8)
        
        # Episode info
        task_text = f"Task: {self.metadata['task']}"
        cv2.putText(header, task_text, (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
        
        # Frame info
        frame_text = f"Frame: {idx + 1}/{self.num_frames}"
        cv2.putText(header, frame_text, (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
        
        # Success/Failure
        success_text = "SUCCESS" if self.metadata['success'] else "FAILURE"
        success_color = (0, 255, 0) if self.metadata['success'] else (0, 0, 255)
        cv2.putText(header, success_text, (combined.shape[1] - 150, 25), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, success_color, 2)
        
        # Playback controls
        controls_text = "SPACE: Play/Pause | LEFT/RIGHT: Step | A: Toggle Actions | S: Toggle States | Q: Quit"
        cv2.putText(header, controls_text, (10, 75), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.4, (200, 200, 200), 1)
        
        # Combine with header
        display = np.vstack([header, combined])
        
        return display
    
    def run(self) -> None:
        """Run interactive viewer."""
        cv2.namedWindow(self.window_name, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(self.window_name, 1280, 560)
        
        print("\n[Viewer] Controls:")
        print("  SPACE - Play/Pause")
        print("  LEFT/RIGHT - Step backward/forward")
        print("  A - Toggle action overlay")
        print("  S - Toggle state overlay")
        print("  +/- - Increase/decrease playback speed")
        print("  Q/ESC - Quit")
        
        while True:
            # Create display
            display = self._create_display(self.state.current_frame)
            cv2.imshow(self.window_name, display)
            
            # Calculate actual timestep from data
            if self.state.current_frame < len(self.timestamps) - 1:
                dt = self.timestamps[self.state.current_frame + 1]['sim_time'] - self.timestamps[self.state.current_frame]['sim_time']
                wait_time = int(dt * 1000 / self.state.playback_speed) if self.state.playing else 0
            else:
                wait_time = int(16 / self.state.playback_speed) if self.state.playing else 0  # Default 16ms
            
            key = cv2.waitKey(wait_time) & 0xFF
            
            if key == ord('q') or key == 27:  # Q or ESC
                break
            
            elif key == ord(' '):  # SPACE
                self.state.playing = not self.state.playing
                print(f"[Viewer] {'Playing' if self.state.playing else 'Paused'}")
            
            elif key == 81 or key == 2:  # LEFT arrow
                self.state.current_frame = max(0, self.state.current_frame - 1)
                self.state.playing = False
            
            elif key == 83 or key == 3:  # RIGHT arrow
                self.state.current_frame = min(self.num_frames - 1, self.state.current_frame + 1)
                self.state.playing = False
            
            elif key == ord('a'):
                self.state.show_actions = not self.state.show_actions
                print(f"[Viewer] Actions overlay: {self.state.show_actions}")
            
            elif key == ord('s'):
                self.state.show_states = not self.state.show_states
                print(f"[Viewer] States overlay: {self.state.show_states}")
            
            elif key == ord('+') or key == ord('='):
                self.state.playback_speed = min(4.0, self.state.playback_speed * 1.5)
                print(f"[Viewer] Speed: {self.state.playback_speed:.1f}x")
            
            elif key == ord('-') or key == ord('_'):
                self.state.playback_speed = max(0.25, self.state.playback_speed / 1.5)
                print(f"[Viewer] Speed: {self.state.playback_speed:.1f}x")
            
            # Auto-advance if playing
            if self.state.playing:
                self.state.current_frame += 1
                if self.state.current_frame >= self.num_frames:
                    self.state.current_frame = 0  # Loop
        
        cv2.destroyAllWindows()


def main():
    """Entry point."""
    import argparse
    
    parser = argparse.ArgumentParser(description="View imitation learning episodes")
    parser.add_argument('dataset_root', type=str, help='Path to dataset root')
    parser.add_argument('episode_id', type=int, help='Episode number to view')
    
    args = parser.parse_args()
    
    dataset_root = Path(args.dataset_root)
    episode_dir = dataset_root / f"episode_{args.episode_id:04d}"
    
    if not episode_dir.exists():
        print(f"Episode {args.episode_id:04d} not found in {dataset_root}")
        sys.exit(1)
    
    viewer = EpisodeViewer(episode_dir)
    viewer.run()


if __name__ == "__main__":
    main()
