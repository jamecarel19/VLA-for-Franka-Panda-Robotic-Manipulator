"""
Model Policy Controller - Uses trained VLA model for autonomous control
"""

import torch
import torch.nn as nn
import numpy as np
from pathlib import Path
import sys

# Add parent directory to path
sys.path.append(str(Path(__file__).parent))

from train_example import SimpleVLAPolicy


class ModelPolicy:
    """Loads and runs trained VLA model for inference."""
    
    def __init__(self, checkpoint_path: str, device: str = 'cpu'):
        self.device = torch.device(device)
        
        # Load checkpoint
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        
        # Detect if model uses language (check for language_encoder in state_dict)
        use_language = any('language_encoder' in key for key in checkpoint['model_state_dict'].keys())
        
        # Initialize model (match training config)
        self.model = SimpleVLAPolicy(
            state_dim=15,
            action_dim=7,
            use_language=use_language
        )
        
        # Load weights
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.model.to(self.device)
        self.model.eval()
        
        self.use_language = use_language
        
        # Initialize text encoder if using language
        self.text_encoder = None
        if self.use_language:
            try:
                print("[ModelPolicy] Loading text encoder...")
                from sentence_transformers import SentenceTransformer
                self.text_encoder = SentenceTransformer('all-MiniLM-L6-v2')
                print(f"[ModelPolicy] Text encoder loaded: all-MiniLM-L6-v2")
            except Exception as e:
                print(f"[ModelPolicy] WARNING: Failed to load text encoder: {e}")
                print("[ModelPolicy] Falling back to zero embeddings")
        
        print(f"[ModelPolicy] Loaded model from {checkpoint_path}")
        print(f"[ModelPolicy] Device: {self.device}")
        print(f"[ModelPolicy] Language: {self.use_language}")
        if 'epoch' in checkpoint:
            print(f"[ModelPolicy] Epoch: {checkpoint['epoch']}")
        if 'val_loss' in checkpoint:
            print(f"[ModelPolicy] Val Loss: {checkpoint['val_loss']:.4f}")
    
    def predict_action(self, gripper_image: np.ndarray, workspace_image: np.ndarray, 
                      robot_state: np.ndarray, task: str = "") -> np.ndarray:
        """
        Predict action from observations.
        
        Args:
            gripper_image: [H, W, 3] RGB image (0-255)
            workspace_image: [H, W, 3] RGB image (0-255)
            robot_state: [15] robot state vector
            task: Natural language task description (if model uses language)
        
        Returns:
            action: [7] action vector (delta_xyz[3], gripper_cmd[1], padding[3])
        """
        with torch.no_grad():
            # Preprocess images
            grip_tensor = self._preprocess_image(gripper_image)
            work_tensor = self._preprocess_image(workspace_image)
            
            # Prepare robot state
            state_tensor = torch.from_numpy(robot_state).float().unsqueeze(0).to(self.device)
            
            # Create batch
            batch = {
                'gripper_image': grip_tensor,
                'workspace_image': work_tensor,
                'robot_state': state_tensor
            }
            
            # Add language embedding if needed
            if self.use_language:
                if self.text_encoder is not None:
                    # Encode task text to 384-dim embedding
                    task_embedding = self.text_encoder.encode([task], convert_to_tensor=True)
                    # Pad or project to 512 dims expected by model
                    if task_embedding.shape[1] < 512:
                        padding = torch.zeros(1, 512 - task_embedding.shape[1], device=task_embedding.device)
                        task_embedding = torch.cat([task_embedding, padding], dim=1)
                    batch['task_embedding'] = task_embedding.to(self.device)
                else:
                    # Fallback to zeros if encoder failed to load
                    batch['task_embedding'] = torch.zeros(1, 512).to(self.device)
            
            # Predict
            action = self.model(batch)
            
            # Return as numpy
            return action.cpu().numpy()[0]
    
    def _preprocess_image(self, img: np.ndarray) -> torch.Tensor:
        """
        Preprocess image for model input.
        
        Args:
            img: [H, W, 3] numpy array (0-255)
        
        Returns:
            tensor: [1, 3, 224, 224]
        """
        import cv2
        
        # Resize to 224x224
        img = cv2.resize(img, (224, 224))
        
        # Convert to float and normalize to [0, 1]
        img = img.astype(np.float32) / 255.0
        
        # ImageNet normalization
        mean = np.array([0.485, 0.456, 0.406])
        std = np.array([0.229, 0.224, 0.225])
        img = (img - mean) / std
        
        # HWC -> CHW
        img = img.transpose(2, 0, 1)
        
        # Add batch dimension
        tensor = torch.from_numpy(img).float().unsqueeze(0).to(self.device)
        
        return tensor
