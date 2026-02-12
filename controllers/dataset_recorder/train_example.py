"""
Example VLA Training Script

Demonstrates how to train a vision-language-action policy using the dataset.
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from pathlib import Path
import sys

# Add current directory to path
sys.path.append(str(Path(__file__).parent))

from dataset_loader import VLADataModule, ImitationLearningDataset


class SimpleVLAPolicy(nn.Module):
    """
    Simple baseline VLA policy.
    
    Architecture:
        - ResNet18 visual encoder (dual camera)
        - MLP for robot state
        - Optional language embedding
        - Fusion layer
        - Action prediction head
    """
    
    def __init__(
        self,
        state_dim: int = 15,
        action_dim: int = 7,
        use_language: bool = False,
        language_dim: int = 512
    ):
        super().__init__()
        
        self.use_language = use_language
        
        # Visual encoders (pretrained ResNet18)
        from torchvision.models import resnet18, ResNet18_Weights
        
        self.gripper_encoder = resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)
        self.gripper_encoder.fc = nn.Identity()  # Remove final FC
        
        self.workspace_encoder = resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)
        self.workspace_encoder.fc = nn.Identity()
        
        visual_feat_dim = 512 * 2  # Two ResNet18 outputs
        
        # Robot state encoder
        self.state_encoder = nn.Sequential(
            nn.Linear(state_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 128)
        )
        
        # Fusion
        fusion_input_dim = visual_feat_dim + 128
        
        if use_language:
            self.language_encoder = nn.Sequential(
                nn.Linear(language_dim, 256),
                nn.ReLU(),
                nn.Linear(256, 128)
            )
            fusion_input_dim += 128
        
        # Action prediction head
        self.action_head = nn.Sequential(
            nn.Linear(fusion_input_dim, 512),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, action_dim)
        )
    
    def forward(self, batch):
        """
        Forward pass.
        
        Args:
            batch: dict with keys:
                - gripper_image: [B, C, H, W]
                - workspace_image: [B, C, H, W]
                - robot_state: [B, state_dim]
                - task_embedding: [B, lang_dim] (if use_language)
        
        Returns:
            actions: [B, action_dim]
        """
        # Encode images
        grip_feat = self.gripper_encoder(batch['gripper_image'])  # [B, 512]
        work_feat = self.workspace_encoder(batch['workspace_image'])  # [B, 512]
        
        # Concatenate visual features
        visual_feat = torch.cat([grip_feat, work_feat], dim=1)  # [B, 1024]
        
        # Encode robot state
        state_feat = self.state_encoder(batch['robot_state'])  # [B, 128]
        
        # Concatenate
        features = [visual_feat, state_feat]
        
        if self.use_language:
            lang_feat = self.language_encoder(batch['task_embedding'])
            features.append(lang_feat)
        
        fused = torch.cat(features, dim=1)
        
        # Predict action
        action = self.action_head(fused)
        
        return action


def train_epoch(model, dataloader, optimizer, criterion, device):
    """Train for one epoch."""
    model.train()
    
    total_loss = 0.0
    num_batches = 0
    
    for batch in dataloader:
        # Move to device
        batch = {
            k: v.to(device) if isinstance(v, torch.Tensor) else v
            for k, v in batch.items()
        }
        
        # Forward
        pred_action = model(batch)
        loss = criterion(pred_action, batch['action'])
        
        # Backward
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
        num_batches += 1
    
    return total_loss / num_batches


def validate(model, dataloader, criterion, device):
    """Validate model."""
    model.eval()
    
    total_loss = 0.0
    num_batches = 0
    
    with torch.no_grad():
        for batch in dataloader:
            # Move to device
            batch = {
                k: v.to(device) if isinstance(v, torch.Tensor) else v
                for k, v in batch.items()
            }
            
            # Forward
            pred_action = model(batch)
            loss = criterion(pred_action, batch['action'])
            
            total_loss += loss.item()
            num_batches += 1
    
    return total_loss / num_batches


def main():
    """Training script."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Train VLA policy")
    parser.add_argument('dataset_root', type=str, help='Path to dataset')
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu')
    parser.add_argument('--use_language', action='store_true')
    parser.add_argument('--frame_stack', type=int, default=1)
    parser.add_argument('--checkpoint_dir', type=str, default='checkpoints')
    
    args = parser.parse_args()
    
    print("=" * 70)
    print("VLA POLICY TRAINING")
    print("=" * 70)
    print(f"Dataset: {args.dataset_root}")
    print(f"Device: {args.device}")
    print(f"Batch size: {args.batch_size}")
    print(f"Epochs: {args.epochs}")
    print(f"Learning rate: {args.lr}")
    print("=" * 70)
    
    # Pre-cache SentenceTransformer model if using language
    if args.use_language:
        print("\n[Setup] Pre-loading SentenceTransformer model...")
        from sentence_transformers import SentenceTransformer
        _ = SentenceTransformer('all-MiniLM-L6-v2')
        print("[Setup] Model cached successfully\n")
    
    # Create data module
    datamodule = VLADataModule(
        dataset_root=args.dataset_root,
        batch_size=args.batch_size,
        num_workers=0,  # Use 0 to avoid multi-process model loading issues
        val_split=0.1,
        image_size=(224, 224),
        frame_stack=args.frame_stack,
        success_only=False,
        use_language=args.use_language
    )
    
    train_loader = datamodule.train_dataloader()
    val_loader = datamodule.val_dataloader()
    
    # Create model
    model = SimpleVLAPolicy(
        state_dim=15,
        action_dim=7,
        use_language=args.use_language
    )
    model = model.to(args.device)
    
    # Optimizer and loss
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    criterion = nn.MSELoss()
    
    # Checkpoint directory
    checkpoint_dir = Path(args.checkpoint_dir)
    checkpoint_dir.mkdir(exist_ok=True)
    
    # Training loop
    best_val_loss = float('inf')
    
    for epoch in range(args.epochs):
        print(f"\nEpoch {epoch + 1}/{args.epochs}")
        
        # Train
        train_loss = train_epoch(model, train_loader, optimizer, criterion, args.device)
        print(f"  Train loss: {train_loss:.6f}")
        
        # Validate
        val_loss = validate(model, val_loader, criterion, args.device)
        print(f"  Val loss:   {val_loss:.6f}")
        
        # Save checkpoint
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            checkpoint_path = checkpoint_dir / 'best_model.pth'
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_loss': val_loss,
            }, checkpoint_path)
            print(f"  ✓ Saved best model (val_loss={val_loss:.6f})")
    
    print("\n" + "=" * 70)
    print("Training complete!")
    print(f"Best validation loss: {best_val_loss:.6f}")
    print("=" * 70)


if __name__ == "__main__":
    main()
