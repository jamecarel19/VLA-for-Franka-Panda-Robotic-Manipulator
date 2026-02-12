# Imitation Learning Dataset Recorder

Complete production-ready dataset recording system for Franka Panda robot in Webots.

## Overview

This system records synchronized multi-modal data for vision-language-action (VLA) model training:

- **Dual camera streams** (gripper-mounted + workspace)
- **Robot states** (joints, end-effector pose, gripper)
- **Actions** (delta XYZ, delta RPY, gripper commands)
- **Language task descriptions**
- **Temporal synchronization**
- **Episode-based organization**

## Dataset Structure

```
dataset/
├── episode_0001/
│   ├── frames/
│   │   ├── 000001_grip.png
│   │   ├── 000001_work.png
│   │   ├── 000002_grip.png
│   │   ├── 000002_work.png
│   │   └── ...
│   ├── states.jsonl
│   ├── actions.jsonl
│   ├── timestamps.jsonl
│   └── meta.json
├── episode_0002/
│   └── ...
```

### File Formats

**states.jsonl** - One JSON per timestep:
```json
{
  "t": 1,
  "joints": [0.0, -0.785, 0.0, -2.356, 0.0, 1.571, 0.785],
  "ee_pose": [0.307, 0.0, 0.487, 0.0, 0.0, 0.0, 1.0],
  "gripper": 0.04,
  "object_positions": {
    "red_cube": [0.5, 0.2, 0.025],
    "box": [0.0, 0.6, 0.1]
  }
}
```

**actions.jsonl** - One JSON per timestep:
```json
{
  "t": 1,
  "delta_xyz": [0.001, 0.002, -0.001],
  "delta_rpy": [0.0, 0.0, 0.0],
  "gripper_cmd": 0
}
```

**timestamps.jsonl** - One JSON per timestep:
```json
{
  "t": 1,
  "sim_time": 0.032,
  "wall_time": 1707177648.325
}
```

**meta.json** - Episode metadata:
```json
{
  "episode_id": 1,
  "task": "pick the red cube and place in box",
  "difficulty": "easy",
  "success": true,
  "num_steps": 243,
  "operator": "jame",
  "date": "2026-02-06",
  "simulator": "Webots",
  "robot": "Franka Panda",
  "notes": ""
}
```

## Installation

```bash
# Install dependencies
pip install numpy torch torchvision opencv-python pillow
```

## Usage

### 1. Recording Episodes (Webots Controller)

```python
from recorder import DatasetRecorder, TeleopRecorderController

# In your Webots controller
controller = TeleopRecorderController()
controller.run()
```

**Keyboard Controls:**
- `R` - Start recording episode
- `S` - Stop and save (success)
- `F` - Stop and save (failure)
- `SPACE` - Return robot to home
- Arrow keys, Z/X, A/S - Robot control

### 2. Validating Dataset

```bash
# Validate all episodes
python validate_dataset.py /path/to/dataset

# Validate specific episode
python validate_dataset.py /path/to/dataset --episode 5

# Quiet mode
python validate_dataset.py /path/to/dataset --quiet
```

**Checks:**
- ✓ File completeness
- ✓ Frame synchronization
- ✓ Temporal consistency
- ✓ Data corruption
- ✓ State ranges
- ✓ Action magnitudes

### 3. Viewing Episodes

```bash
python viewer.py /path/to/dataset 1
```

**Viewer Controls:**
- `SPACE` - Play/Pause
- `LEFT/RIGHT` - Step backward/forward
- `A` - Toggle action overlay
- `S` - Toggle state overlay
- `+/-` - Adjust playback speed
- `Q/ESC` - Quit

### 4. Loading Data for Training

```python
from dataset_loader import ImitationLearningDataset, VLADataModule

# Simple dataset
dataset = ImitationLearningDataset(
    dataset_root='dataset',
    image_size=(224, 224),
    frame_stack=1,
    success_only=True
)

# With DataModule (automatic train/val split)
datamodule = VLADataModule(
    dataset_root='dataset',
    batch_size=32,
    val_split=0.1
)

train_loader = datamodule.train_dataloader()
val_loader = datamodule.val_dataloader()
```

### 5. Training a Policy

```bash
# Basic training
python train_example.py dataset --epochs 100

# With language conditioning
python train_example.py dataset --use_language --epochs 100

# Frame stacking
python train_example.py dataset --frame_stack 3 --epochs 100

# Full options
python train_example.py dataset \
    --batch_size 64 \
    --epochs 200 \
    --lr 1e-4 \
    --use_language \
    --frame_stack 3 \
    --checkpoint_dir checkpoints
```

## API Reference

### DatasetRecorder

Main recording class.

```python
recorder = DatasetRecorder(
    robot=supervisor,
    gripper_camera=gripper_cam,
    workspace_camera=workspace_cam,
    dataset_root='dataset',
    operator='jame'
)

# Track objects
recorder.add_tracked_object('red_cube', cube_node)

# Start episode
recorder.start_episode(
    task='pick the red cube',
    difficulty='easy'
)

# Log each timestep
recorder.log_step()

# End episode
recorder.end_episode(success=True, notes='Perfect!')
```

### ImitationLearningDataset

PyTorch dataset for loading episodes.

```python
dataset = ImitationLearningDataset(
    dataset_root='dataset',
    episode_ids=[1, 2, 3],  # Specific episodes or None for all
    image_size=(224, 224),
    normalize=True,
    augment=True,  # Data augmentation
    use_language=True,  # Include task embeddings
    frame_stack=3,  # Stack N frames
    success_only=True,  # Filter failed episodes
    max_episodes=100  # Limit number of episodes
)

sample = dataset[0]
# Returns:
# {
#     'gripper_image': Tensor [C, H, W],
#     'workspace_image': Tensor [C, H, W],
#     'robot_state': Tensor [15],  # joints(7) + ee_pose(7) + gripper(1)
#     'action': Tensor [7],  # delta_xyz(3) + delta_rpy(3) + gripper_cmd(1)
#     'task_embedding': Tensor [512],  # if use_language=True
#     'metadata': dict
# }
```

### VLADataModule

High-level data module with automatic train/val split.

```python
datamodule = VLADataModule(
    dataset_root='dataset',
    batch_size=32,
    num_workers=4,
    val_split=0.1,  # 10% validation
    image_size=(224, 224),
    frame_stack=1,
    success_only=False
)

train_loader = datamodule.train_dataloader()
val_loader = datamodule.val_dataloader()
```

## Advanced Features

### Frame Stacking

Stack multiple consecutive frames for temporal context:

```python
dataset = ImitationLearningDataset(
    dataset_root='dataset',
    frame_stack=3  # Stack 3 frames
)

# Returns images with shape [C*3, H, W]
```

### Custom Task Encoding

Replace simple hash-based encoding with CLIP/BERT:

```python
class ImitationLearningDataset(Dataset):
    def _encode_task(self, task_text: str) -> torch.Tensor:
        # Use CLIP text encoder
        import clip
        model, _ = clip.load("ViT-B/32")
        tokens = clip.tokenize([task_text])
        with torch.no_grad():
            embedding = model.encode_text(tokens)
        return embedding.squeeze()
```

### Data Augmentation

Built-in augmentations for training:

```python
dataset = ImitationLearningDataset(
    dataset_root='dataset',
    augment=True  # ColorJitter + RandomAffine
)
```

### Filtering Episodes

```python
# Only successful episodes
dataset = ImitationLearningDataset(
    dataset_root='dataset',
    success_only=True
)

# Specific episodes
dataset = ImitationLearningDataset(
    dataset_root='dataset',
    episode_ids=[1, 5, 10, 15]
)

# Limited number
dataset = ImitationLearningDataset(
    dataset_root='dataset',
    max_episodes=50
)
```

## File Descriptions

| File | Purpose |
|------|---------|
| `recorder.py` | Main recording system (Webots controller) |
| `validate_dataset.py` | Dataset integrity validation |
| `viewer.py` | Interactive episode visualization |
| `dataset_loader.py` | PyTorch Dataset and DataLoader |
| `train_example.py` | Example training script |

## Troubleshooting

### Camera Not Found

Ensure cameras are enabled in Webots controller:

```python
wrist_camera = robot.getDevice("wrist_camera")
wrist_camera.enable(timestep)
```

### Missing Object Tracking

Add objects to recorder:

```python
cube_node = robot.getFromDef("RED_CUBE")
recorder.add_tracked_object("red_cube", cube_node)
```

### Frame Synchronization Issues

Frames are saved every `log_step()` call. Ensure consistent calling:

```python
while robot.step(timestep) != -1:
    if recording:
        recorder.log_step()  # Call every timestep
```

### Validation Errors

Check error messages:
```bash
python validate_dataset.py dataset
```

Common issues:
- Missing frames: Check camera save paths
- Mismatched counts: Ensure all JSONL writes complete
- Corrupted images: Verify disk space

## Performance Tips

### Recording
- Use appropriate camera resolution (640x480 recommended)
- Save images as PNG (lossless, good compression)
- Log every 1-3 timesteps for efficiency

### Training
- Use `num_workers > 0` for parallel data loading
- Enable `pin_memory=True` for GPU training
- Resize images to model input size (224x224)
- Use mixed precision training for speed

### Storage
- ~1-2 MB per frame pair (PNG)
- 300 steps/episode × 2 cameras = ~600 MB/episode
- 100 episodes ≈ 60 GB

## Citation

If you use this dataset format in research:

```bibtex
@misc{panda_imitation_dataset,
  author = {Your Name},
  title = {Franka Panda Imitation Learning Dataset},
  year = {2026},
  publisher = {GitHub},
  howpublished = {\url{https://github.com/yourusername/repo}}
}
```

## License

MIT License - See LICENSE file for details.

## Contributing

Contributions welcome! Areas for improvement:
- Additional validation checks
- More sophisticated task encodings (CLIP, BERT)
- Real-time recording dashboard
- Multi-robot support
- Cloud storage integration

## Contact

For questions or issues, please open a GitHub issue or contact jame@example.com.
