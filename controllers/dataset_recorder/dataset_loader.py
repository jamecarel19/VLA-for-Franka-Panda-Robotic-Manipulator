"""
PyTorch Dataset Loader

Loads imitation learning dataset for training.
Supports multi-modal inputs and various augmentations.
"""

import json
import torch
import numpy as np
from pathlib import Path
from typing import List, Dict, Tuple, Optional, Callable
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import torchvision.transforms as T


class ImitationLearningDataset(Dataset):
    """
    PyTorch Dataset for multi-modal imitation learning.
    
    Returns:
        {
            'gripper_image': Tensor [C, H, W],
            'workspace_image': Tensor [C, H, W],
            'robot_state': Tensor [state_dim],
            'action': Tensor [action_dim],
            'task_embedding': Tensor [task_emb_dim],  # if use_language=True
            'metadata': dict
        }
    """
    
    def __init__(
        self,
        dataset_root: str,
        episode_ids: Optional[List[int]] = None,
        image_size: Tuple[int, int] = (224, 224),
        normalize: bool = True,
        augment: bool = False,
        use_language: bool = False,
        frame_stack: int = 1,
        success_only: bool = False,
        max_episodes: Optional[int] = None
    ):
        """
        Args:
            dataset_root: Path to dataset directory
            episode_ids: Specific episodes to load (None = all)
            image_size: Resize images to (H, W)
            normalize: Apply ImageNet normalization
            augment: Apply data augmentation
            use_language: Include language task embeddings
            frame_stack: Stack N consecutive frames
            success_only: Only load successful episodes
            max_episodes: Limit number of episodes
        """
        self.dataset_root = Path(dataset_root)
        self.image_size = image_size
        self.use_language = use_language
        self.frame_stack = frame_stack
        
        # Load episodes
        self.episodes = self._load_episodes(
            episode_ids, 
            success_only, 
            max_episodes
        )
        
        # Build frame index
        self.frame_index = self._build_frame_index()
        
        # Image transforms
        self.transform = self._build_transform(normalize, augment)
        
        print(f"[Dataset] Loaded {len(self.episodes)} episodes")
        print(f"[Dataset] Total frames: {len(self.frame_index)}")
        print(f"[Dataset] Image size: {image_size}")
        print(f"[Dataset] Frame stack: {frame_stack}")
    
    def _load_episodes(
        self, 
        episode_ids: Optional[List[int]],
        success_only: bool,
        max_episodes: Optional[int]
    ) -> List[Dict]:
        """Load episode metadata."""
        episodes = []
        
        # Find all episode directories
        episode_dirs = sorted([
            d for d in self.dataset_root.iterdir()
            if d.is_dir() and d.name.startswith('episode_')
        ])
        
        for ep_dir in episode_dirs:
            ep_id = int(ep_dir.name.split('_')[1])
            
            # Filter by episode_ids
            if episode_ids is not None and ep_id not in episode_ids:
                continue
            
            # Load metadata
            meta_path = ep_dir / 'meta.json'
            if not meta_path.exists():
                print(f"[Dataset] WARNING: Missing metadata for episode {ep_id}")
                continue
            
            with open(meta_path) as f:
                meta = json.load(f)
            
            # Filter by success
            if success_only and not meta.get('success', False):
                continue
            
            # Load data
            try:
                states = self._load_jsonl(ep_dir / 'states.jsonl')
                actions = self._load_jsonl(ep_dir / 'actions.jsonl')
                timestamps = self._load_jsonl(ep_dir / 'timestamps.jsonl')
            except Exception as e:
                print(f"[Dataset] WARNING: Failed to load episode {ep_id}: {e}")
                continue
            
            episodes.append({
                'id': ep_id,
                'dir': ep_dir,
                'meta': meta,
                'states': states,
                'actions': actions,
                'timestamps': timestamps,
                'num_frames': len(states)
            })
            
            # Check max_episodes
            if max_episodes and len(episodes) >= max_episodes:
                break
        
        return episodes
    
    def _load_jsonl(self, filepath: Path) -> List[Dict]:
        """Load JSONL file."""
        data = []
        with open(filepath) as f:
            for line in f:
                data.append(json.loads(line.strip()))
        return data
    
    def _build_frame_index(self) -> List[Tuple[int, int]]:
        """
        Build index of (episode_idx, frame_idx) for all valid frames.
        
        For frame stacking, exclude last (frame_stack - 1) frames from each episode.
        """
        index = []
        
        for ep_idx, episode in enumerate(self.episodes):
            num_frames = episode['num_frames']
            
            # Account for frame stacking
            valid_frames = num_frames - (self.frame_stack - 1)
            
            for frame_idx in range(valid_frames):
                index.append((ep_idx, frame_idx))
        
        return index
    
    def _build_transform(self, normalize: bool, augment: bool) -> Callable:
        """Build image transform pipeline."""
        transforms = []
        
        # Resize
        transforms.append(T.Resize(self.image_size))
        
        # Augmentation
        if augment:
            transforms.extend([
                T.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.05),
                T.RandomAffine(degrees=5, translate=(0.05, 0.05), scale=(0.95, 1.05)),
            ])
        
        # To tensor
        transforms.append(T.ToTensor())
        
        # Normalization
        if normalize:
            transforms.append(
                T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            )
        
        return T.Compose(transforms)
    
    def _load_frame_stack(self, episode: Dict, start_idx: int) -> Tuple[List[Image.Image], List[Image.Image]]:
        """Load frame stack starting from start_idx."""
        frames_dir = episode['dir'] / 'frames'
        
        grip_images = []
        work_images = []
        
        for i in range(self.frame_stack):
            frame_num = start_idx + i + 1  # Frames are 1-indexed
            
            grip_path = frames_dir / f"{frame_num:06d}_grip.png"
            work_path = frames_dir / f"{frame_num:06d}_work.png"
            
            grip_images.append(Image.open(grip_path).convert('RGB'))
            work_images.append(Image.open(work_path).convert('RGB'))
        
        return grip_images, work_images
    
    def _encode_task(self, task_text: str) -> torch.Tensor:
        """
        Encode task using sentence transformers (same as inference).
        """
        if not hasattr(self, '_text_encoder'):
            from sentence_transformers import SentenceTransformer
            self._text_encoder = SentenceTransformer('all-MiniLM-L6-v2')
        
        # Encode to 384-dim, pad to 512
        embedding = self._text_encoder.encode([task_text], convert_to_tensor=True)[0]
        if embedding.shape[0] < 512:
            padding = torch.zeros(512 - embedding.shape[0], device=embedding.device)
            embedding = torch.cat([embedding, padding])
        return embedding.cpu()  # Return on CPU for consistency with dataset loading
    
    def __len__(self) -> int:
        return len(self.frame_index)
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """Get single training sample."""
        ep_idx, frame_idx = self.frame_index[idx]
        episode = self.episodes[ep_idx]
        
        # Load images
        grip_images, work_images = self._load_frame_stack(episode, frame_idx)
        
        # Transform images
        grip_tensors = [self.transform(img) for img in grip_images]
        work_tensors = [self.transform(img) for img in work_images]
        
        # Stack if multiple frames
        if self.frame_stack > 1:
            grip_img = torch.cat(grip_tensors, dim=0)  # [C*stack, H, W]
            work_img = torch.cat(work_tensors, dim=0)
        else:
            grip_img = grip_tensors[0]
            work_img = work_tensors[0]
        
        # Get robot state (last frame in stack)
        state_data = episode['states'][frame_idx + self.frame_stack - 1]
        
        # Build state vector: [joints (7) + ee_pose (7) + gripper (1)] = 15
        robot_state = torch.tensor(
            state_data['joints'] + state_data['ee_pose'] + [state_data['gripper']],
            dtype=torch.float32
        )
        
        # Get action (last frame in stack)
        action_data = episode['actions'][frame_idx + self.frame_stack - 1]
        
        # Build action vector: [delta_xyz (3) + delta_rpy (3) + gripper_cmd (1)] = 7
        action = torch.tensor(
            action_data['delta_xyz'] + action_data['delta_rpy'] + [action_data['gripper_cmd']],
            dtype=torch.float32
        )
        
        sample = {
            'gripper_image': grip_img,
            'workspace_image': work_img,
            'robot_state': robot_state,
            'action': action,
            'metadata': {
                'episode_id': episode['id'],
                'frame_idx': frame_idx,
                'task': episode['meta']['task']
            }
        }
        
        # Add language embedding if requested
        if self.use_language:
            task_emb = self._encode_task(episode['meta']['task'])
            sample['task_embedding'] = task_emb
        
        return sample


class VLADataModule:
    """
    Data module for Vision-Language-Action training.
    
    Handles train/val splits and dataloaders.
    """
    
    def __init__(
        self,
        dataset_root: str,
        batch_size: int = 32,
        num_workers: int = 4,
        val_split: float = 0.1,
        image_size: Tuple[int, int] = (224, 224),
        frame_stack: int = 1,
        success_only: bool = False,
        **dataset_kwargs
    ):
        self.dataset_root = dataset_root
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.val_split = val_split
        self.image_size = image_size
        self.frame_stack = frame_stack
        self.success_only = success_only
        self.dataset_kwargs = dataset_kwargs
        
        # Split episodes
        self.train_episodes, self.val_episodes = self._split_episodes()
        
        print(f"[DataModule] Train episodes: {len(self.train_episodes)}")
        print(f"[DataModule] Val episodes: {len(self.val_episodes)}")
    
    def _split_episodes(self) -> Tuple[List[int], List[int]]:
        """Split episodes into train/val."""
        dataset_root = Path(self.dataset_root)
        
        # Get all episode IDs
        episode_ids = sorted([
            int(d.name.split('_')[1])
            for d in dataset_root.iterdir()
            if d.is_dir() and d.name.startswith('episode_')
        ])
        
        # Shuffle with fixed seed for reproducibility
        rng = np.random.RandomState(42)
        rng.shuffle(episode_ids)
        
        # Split
        n_val = max(1, int(len(episode_ids) * self.val_split))
        val_ids = episode_ids[:n_val]
        train_ids = episode_ids[n_val:]
        
        return train_ids, val_ids
    
    def train_dataloader(self) -> DataLoader:
        """Create training dataloader."""
        dataset = ImitationLearningDataset(
            dataset_root=self.dataset_root,
            episode_ids=self.train_episodes,
            image_size=self.image_size,
            normalize=True,
            augment=True,
            frame_stack=self.frame_stack,
            success_only=self.success_only,
            **self.dataset_kwargs
        )
        
        return DataLoader(
            dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            pin_memory=True,
            drop_last=True
        )
    
    def val_dataloader(self) -> DataLoader:
        """Create validation dataloader."""
        dataset = ImitationLearningDataset(
            dataset_root=self.dataset_root,
            episode_ids=self.val_episodes,
            image_size=self.image_size,
            normalize=True,
            augment=False,
            frame_stack=self.frame_stack,
            success_only=self.success_only,
            **self.dataset_kwargs
        )
        
        return DataLoader(
            dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=True
        )


def demo():
    """Demo usage."""
    import sys
    
    if len(sys.argv) < 2:
        print("Usage: python dataset_loader.py <dataset_root>")
        sys.exit(1)
    
    dataset_root = sys.argv[1]
    
    print("=" * 70)
    print("DATASET LOADER DEMO")
    print("=" * 70)
    
    # Create dataset
    dataset = ImitationLearningDataset(
        dataset_root=dataset_root,
        image_size=(224, 224),
        normalize=True,
        augment=False,
        frame_stack=1,
        success_only=False,
        use_language=True
    )
    
    print(f"\nDataset size: {len(dataset)}")
    
    # Load sample
    sample = dataset[0]
    
    print("\nSample keys:", sample.keys())
    print("\nSample shapes:")
    for key, val in sample.items():
        if isinstance(val, torch.Tensor):
            print(f"  {key}: {val.shape} ({val.dtype})")
        else:
            print(f"  {key}: {type(val)}")
    
    print("\nMetadata:", sample['metadata'])
    
    # Test dataloader
    print("\n" + "=" * 70)
    print("Testing DataLoader...")
    
    datamodule = VLADataModule(
        dataset_root=dataset_root,
        batch_size=8,
        num_workers=0,
        val_split=0.2,
        image_size=(224, 224),
        frame_stack=1
    )
    
    train_loader = datamodule.train_dataloader()
    
    print(f"Train batches: {len(train_loader)}")
    
    # Get batch
    batch = next(iter(train_loader))
    
    print("\nBatch shapes:")
    for key, val in batch.items():
        if isinstance(val, torch.Tensor):
            print(f"  {key}: {val.shape}")
        elif isinstance(val, dict):
            print(f"  {key}: {type(val)} (metadata)")
    
    print("\n" + "=" * 70)
    print("Demo complete!")


if __name__ == "__main__":
    demo()
