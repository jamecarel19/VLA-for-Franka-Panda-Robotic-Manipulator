"""
Dataset Validation Script

Validates dataset integrity:
- Episode structure completeness
- Frame synchronization
- Missing files
- Data corruption
- Temporal consistency
"""

import json
import sys
from pathlib import Path
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass
import numpy as np
from PIL import Image


@dataclass
class ValidationResult:
    """Validation result for an episode."""
    episode_id: int
    valid: bool
    errors: List[str]
    warnings: List[str]
    stats: Dict[str, any]


class DatasetValidator:
    """Validates dataset structure and integrity."""
    
    def __init__(self, dataset_root: str):
        self.dataset_root = Path(dataset_root)
        
        if not self.dataset_root.exists():
            raise FileNotFoundError(f"Dataset root not found: {dataset_root}")
    
    def validate_all(self, verbose: bool = True) -> List[ValidationResult]:
        """Validate all episodes in dataset."""
        episodes = sorted([
            d for d in self.dataset_root.iterdir()
            if d.is_dir() and d.name.startswith('episode_')
        ])
        
        if not episodes:
            print(f"[Validator] No episodes found in {self.dataset_root}")
            return []
        
        print(f"[Validator] Found {len(episodes)} episodes")
        print("=" * 70)
        
        results = []
        for ep_dir in episodes:
            result = self.validate_episode(ep_dir, verbose=verbose)
            results.append(result)
        
        self._print_summary(results)
        return results
    
    def validate_episode(self, episode_dir: Path, verbose: bool = True) -> ValidationResult:
        """Validate single episode."""
        episode_id = int(episode_dir.name.split('_')[1])
        errors = []
        warnings = []
        stats = {}
        
        if verbose:
            print(f"\n[Episode {episode_id:04d}] Validating...")
        
        # Check required files
        required_files = ['states.jsonl', 'actions.jsonl', 'timestamps.jsonl', 'meta.json']
        for fname in required_files:
            if not (episode_dir / fname).exists():
                errors.append(f"Missing required file: {fname}")
        
        # Check frames directory
        frames_dir = episode_dir / 'frames'
        if not frames_dir.exists():
            errors.append("Missing frames/ directory")
            return ValidationResult(episode_id, False, errors, warnings, stats)
        
        # Load metadata
        meta_path = episode_dir / 'meta.json'
        if meta_path.exists():
            with open(meta_path) as f:
                meta = json.load(f)
            stats['num_steps'] = meta.get('num_steps', 0)
            stats['task'] = meta.get('task', 'unknown')
            stats['success'] = meta.get('success', False)
        else:
            stats['num_steps'] = 0
        
        # Load JSONL files
        try:
            states = self._load_jsonl(episode_dir / 'states.jsonl')
            actions = self._load_jsonl(episode_dir / 'actions.jsonl')
            timestamps = self._load_jsonl(episode_dir / 'timestamps.jsonl')
        except Exception as e:
            errors.append(f"Failed to load JSONL: {e}")
            return ValidationResult(episode_id, False, errors, warnings, stats)
        
        # Check counts match
        num_states = len(states)
        num_actions = len(actions)
        num_timestamps = len(timestamps)
        
        if not (num_states == num_actions == num_timestamps):
            errors.append(
                f"Mismatched counts: states={num_states}, actions={num_actions}, timestamps={num_timestamps}"
            )
        
        stats['actual_steps'] = num_states
        
        # Check metadata consistency
        if stats.get('num_steps') != num_states:
            warnings.append(
                f"Metadata num_steps ({stats.get('num_steps')}) != actual ({num_states})"
            )
        
        # Validate frames
        frame_errors, frame_stats = self._validate_frames(frames_dir, num_states)
        errors.extend(frame_errors)
        stats.update(frame_stats)
        
        # Validate temporal consistency
        temporal_errors = self._validate_temporal(timestamps)
        errors.extend(temporal_errors)
        
        # Validate state ranges
        range_warnings = self._validate_state_ranges(states)
        warnings.extend(range_warnings)
        
        # Check action magnitudes
        action_warnings = self._validate_actions(actions)
        warnings.extend(action_warnings)
        
        valid = len(errors) == 0
        
        if verbose:
            status = "✓ VALID" if valid else "✗ INVALID"
            print(f"  Status: {status}")
            print(f"  Steps: {stats['actual_steps']}")
            print(f"  Errors: {len(errors)}")
            print(f"  Warnings: {len(warnings)}")
            
            if errors:
                for err in errors:
                    print(f"    ERROR: {err}")
            if warnings and len(warnings) <= 3:
                for warn in warnings[:3]:
                    print(f"    WARN: {warn}")
        
        return ValidationResult(episode_id, valid, errors, warnings, stats)
    
    def _load_jsonl(self, filepath: Path) -> List[Dict]:
        """Load JSONL file."""
        data = []
        with open(filepath) as f:
            for line in f:
                data.append(json.loads(line.strip()))
        return data
    
    def _validate_frames(self, frames_dir: Path, expected_steps: int) -> Tuple[List[str], Dict]:
        """Validate frame images."""
        errors = []
        stats = {}
        
        # Find all frames
        grip_frames = sorted(frames_dir.glob('*_grip.png'))
        work_frames = sorted(frames_dir.glob('*_work.png'))
        
        num_grip = len(grip_frames)
        num_work = len(work_frames)
        
        stats['num_grip_frames'] = num_grip
        stats['num_work_frames'] = num_work
        
        # Check counts
        if num_grip != expected_steps:
            errors.append(f"Expected {expected_steps} gripper frames, found {num_grip}")
        
        if num_work != expected_steps:
            errors.append(f"Expected {expected_steps} workspace frames, found {num_work}")
        
        # Check pairing
        if num_grip != num_work:
            errors.append(f"Mismatched frame counts: grip={num_grip}, work={num_work}")
        
        # Check frame IDs are sequential
        if grip_frames:
            grip_ids = [int(f.stem.split('_')[0]) for f in grip_frames]
            if grip_ids != list(range(1, len(grip_ids) + 1)):
                errors.append("Non-sequential gripper frame IDs")
        
        if work_frames:
            work_ids = [int(f.stem.split('_')[0]) for f in work_frames]
            if work_ids != list(range(1, len(work_ids) + 1)):
                errors.append("Non-sequential workspace frame IDs")
        
        # Check frame integrity (sample a few)
        sample_indices = [0, len(grip_frames) // 2, -1] if grip_frames else []
        for idx in sample_indices:
            if idx >= len(grip_frames):
                continue
            
            # Check gripper frame
            try:
                img = Image.open(grip_frames[idx])
                img.verify()
            except Exception as e:
                errors.append(f"Corrupted gripper frame {grip_frames[idx].name}: {e}")
            
            # Check workspace frame
            if idx < len(work_frames):
                try:
                    img = Image.open(work_frames[idx])
                    img.verify()
                except Exception as e:
                    errors.append(f"Corrupted workspace frame {work_frames[idx].name}: {e}")
        
        return errors, stats
    
    def _validate_temporal(self, timestamps: List[Dict]) -> List[str]:
        """Validate timestamp consistency."""
        errors = []
        
        if not timestamps:
            return errors
        
        # Check monotonic increase
        sim_times = [t['sim_time'] for t in timestamps]
        wall_times = [t['wall_time'] for t in timestamps]
        
        if not all(sim_times[i] <= sim_times[i+1] for i in range(len(sim_times) - 1)):
            errors.append("Non-monotonic simulation times")
        
        if not all(wall_times[i] <= wall_times[i+1] for i in range(len(wall_times) - 1)):
            errors.append("Non-monotonic wall clock times")
        
        # Check t field matches index
        for i, ts in enumerate(timestamps):
            if ts['t'] != i + 1:
                errors.append(f"Timestamp t field mismatch at index {i}: expected {i+1}, got {ts['t']}")
                break
        
        return errors
    
    def _validate_state_ranges(self, states: List[Dict]) -> List[str]:
        """Check state values are in reasonable ranges."""
        warnings = []
        
        if not states:
            return warnings
        
        # Check joint limits (approximate Panda limits)
        joint_limits = [
            (-2.8973, 2.8973),
            (-1.7628, 1.7628),
            (-2.8973, 2.8973),
            (-3.0718, -0.0698),
            (-2.8973, 2.8973),
            (-0.0175, 3.7525),
            (-2.8973, 2.8973)
        ]
        
        for state in states:
            joints = state.get('joints', [])
            for i, (q, (qmin, qmax)) in enumerate(zip(joints, joint_limits)):
                if not (qmin <= q <= qmax):
                    warnings.append(f"Joint {i+1} out of range at t={state['t']}: {q:.3f}")
                    break  # Only report first violation per state
        
        # Check gripper range
        for state in states:
            grip = state.get('gripper', 0.0)
            if not (0.0 <= grip <= 0.04):
                warnings.append(f"Gripper out of range [0, 0.04] at t={state['t']}: {grip:.4f}")
                break
        
        return warnings[:5]  # Limit warnings
    
    def _validate_actions(self, actions: List[Dict]) -> List[str]:
        """Check action magnitudes are reasonable."""
        warnings = []
        
        if not actions:
            return warnings
        
        # Check for excessively large deltas
        max_delta_xyz = 0.0
        max_delta_rpy = 0.0
        
        for action in actions:
            dxyz = action.get('delta_xyz', [0, 0, 0])
            drpy = action.get('delta_rpy', [0, 0, 0])
            
            delta_mag = np.linalg.norm(dxyz)
            rot_mag = np.linalg.norm(drpy)
            
            max_delta_xyz = max(max_delta_xyz, delta_mag)
            max_delta_rpy = max(max_delta_rpy, rot_mag)
            
            # Warn if very large (likely error)
            if delta_mag > 0.5:
                warnings.append(f"Large position delta at t={action['t']}: {delta_mag:.3f}m")
            
            if rot_mag > 1.0:
                warnings.append(f"Large rotation delta at t={action['t']}: {rot_mag:.3f}rad")
        
        # Report statistics
        if max_delta_xyz > 0.1:
            warnings.append(f"Max position delta: {max_delta_xyz:.3f}m (check for outliers)")
        
        return warnings[:3]
    
    def _print_summary(self, results: List[ValidationResult]) -> None:
        """Print validation summary."""
        print("\n" + "=" * 70)
        print("VALIDATION SUMMARY")
        print("=" * 70)
        
        total = len(results)
        valid = sum(1 for r in results if r.valid)
        invalid = total - valid
        
        total_errors = sum(len(r.errors) for r in results)
        total_warnings = sum(len(r.warnings) for r in results)
        
        print(f"Total episodes: {total}")
        print(f"Valid:          {valid} ({100*valid/total:.1f}%)")
        print(f"Invalid:        {invalid}")
        print(f"Total errors:   {total_errors}")
        print(f"Total warnings: {total_warnings}")
        
        # Stats across all episodes
        if results:
            total_steps = sum(r.stats.get('actual_steps', 0) for r in results)
            success_rate = sum(r.stats.get('success', False) for r in results) / len(results)
            
            print(f"\nTotal steps:    {total_steps}")
            print(f"Success rate:   {100*success_rate:.1f}%")
        
        print("=" * 70)


def main():
    """Entry point."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Validate imitation learning dataset")
    parser.add_argument('dataset_root', type=str, help='Path to dataset root directory')
    parser.add_argument('--episode', type=int, help='Validate specific episode only')
    parser.add_argument('--quiet', action='store_true', help='Minimal output')
    
    args = parser.parse_args()
    
    validator = DatasetValidator(args.dataset_root)
    
    if args.episode is not None:
        # Validate single episode
        ep_dir = Path(args.dataset_root) / f"episode_{args.episode:04d}"
        if not ep_dir.exists():
            print(f"Episode {args.episode:04d} not found")
            sys.exit(1)
        
        result = validator.validate_episode(ep_dir, verbose=not args.quiet)
        sys.exit(0 if result.valid else 1)
    else:
        # Validate all
        results = validator.validate_all(verbose=not args.quiet)
        
        # Exit with error if any invalid
        if any(not r.valid for r in results):
            sys.exit(1)


if __name__ == "__main__":
    main()
