"""
DeepLabV3+ Training Script for TTPLA Cable Detection
=====================================================
Semantic segmentation with ResNet-50/101 backbone.
Uses Weighted BCE + Dice Loss for thin cable detection.

Usage:
    python src/train_deeplabv3.py --backbone resnet50  # or resnet101
"""

import os
import json
import argparse
import numpy as np
from pathlib import Path
from tqdm import tqdm
import cv2

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch.cuda.amp import autocast, GradScaler
import torchvision
from torchvision.models.segmentation import deeplabv3_resnet50, deeplabv3_resnet101
import albumentations as A
from albumentations.pytorch import ToTensorV2

# =============================================================================
# CONFIGURATION
# =============================================================================
class Config:
    # Paths
    DATA_ROOT = Path(__file__).parent.parent / "data"  # FIX: Correct path
    OUTPUT_DIR = Path(__file__).parent.parent / "models" / "deeplabv3_output"
    
    # Training
    BACKBONE = "resnet50"  # or "resnet101"
    NUM_CLASSES = 1  # Binary: cable vs background
    IMAGE_SIZE = 768  # FIX: Larger size to preserve thin cables
    BATCH_SIZE = 2    # FIX: Reduced for larger images
    NUM_WORKERS = 4
    EPOCHS = 100
    LR = 1e-4
    WEIGHT_DECAY = 1e-4
    
    # Loss weights
    BCE_WEIGHT = 1.0
    TVERSKY_WEIGHT = 2.0  # FIX: Tversky instead of Dice
    POS_WEIGHT = 10.0  # Weight for positive class (cables are rare)
    
    # Device
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    AMP_ENABLED = True
    
    # Checkpointing
    SAVE_EVERY = 10
    EVAL_EVERY = 5

# =============================================================================
# DATASET
# =============================================================================
class TTPLADataset(Dataset):
    """TTPLA Dataset for semantic segmentation."""
    
    def __init__(self, root_dir, split='train', transform=None, val_split=0.2):
        self.root_dir = Path(root_dir)
        self.split = split
        self.transform = transform
        
        # Load annotations
        if split in ['train', 'val']:
            json_path = self.root_dir / 'train' / 'train.json'
            self.img_dir = self.root_dir / 'train'
        else:
            json_path = self.root_dir / 'test' / 'test.json'
            self.img_dir = self.root_dir / 'test'
        
        with open(json_path, 'r') as f:
            coco_data = json.load(f)
        
        # Build image list
        all_images = sorted(coco_data['images'], key=lambda x: x['id'])
        
        if split in ['train', 'val']:
            num_val = int(len(all_images) * val_split)
            if split == 'val':
                self.images = all_images[:num_val]
            else:
                self.images = all_images[num_val:]
        else:
            self.images = all_images
        
        # Build annotation mapping
        self.img_to_anns = {}
        for ann in coco_data['annotations']:
            self.img_to_anns.setdefault(ann['image_id'], []).append(ann)
        
        print(f"Loaded {len(self.images)} images for {split} split")
    
    def __len__(self):
        return len(self.images)
    
    def __getitem__(self, idx):
        img_info = self.images[idx]
        
        # Load image
        img_path = self.img_dir / img_info['file_name']
        image = cv2.imread(str(img_path))
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Create mask from annotations
        h, w = img_info['height'], img_info['width']
        mask = np.zeros((h, w), dtype=np.float32)
        
        for ann in self.img_to_anns.get(img_info['id'], []):
            for poly in ann.get('segmentation', []):
                pts = np.array(poly).reshape(-1, 2).astype(np.int32)
                cv2.fillPoly(mask, [pts], 1.0)
        
        # Extract theta for potential angle supervision
        thetas = []
        for ann in self.img_to_anns.get(img_info['id'], []):
            if 'polar_coordinates' in ann and len(ann['polar_coordinates']) > 0:
                pc = ann['polar_coordinates'][0]
                if 'theta' in pc:
                    thetas.append(pc['theta'])
        
        # Apply transforms
        if self.transform:
            transformed = self.transform(image=image, mask=mask)
            image = transformed['image']
            mask = transformed['mask']
        
        return {
            'image': image,
            'mask': mask.unsqueeze(0) if isinstance(mask, torch.Tensor) else torch.tensor(mask).unsqueeze(0),
            'image_id': img_info['id'],
            'thetas': thetas
        }

# =============================================================================
# AUGMENTATIONS
# =============================================================================
def get_train_transforms(image_size):
    """FIX: Use RandomCrop instead of RandomResizedCrop to preserve thin cables."""
    return A.Compose([
        A.PadIfNeeded(min_height=image_size, min_width=image_size,
                      border_mode=cv2.BORDER_REFLECT_101),
        A.RandomCrop(height=image_size, width=image_size),
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.2),
        A.RandomRotate90(p=0.5),
        A.ShiftScaleRotate(shift_limit=0.05, scale_limit=0.1, rotate_limit=45, p=0.3),
        A.OneOf([
            A.GaussNoise(var_limit=(10, 30)),
            A.GaussianBlur(blur_limit=(3, 5)),
        ], p=0.2),
        A.OneOf([
            A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2),
            A.HueSaturationValue(hue_shift_limit=10, sat_shift_limit=20, val_shift_limit=15),
        ], p=0.3),
        A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ToTensorV2(),
    ])

def get_val_transforms(image_size):
    """Validation: Pad + CenterCrop to preserve resolution."""
    return A.Compose([
        A.PadIfNeeded(min_height=image_size, min_width=image_size,
                      border_mode=cv2.BORDER_REFLECT_101),
        A.CenterCrop(height=image_size, width=image_size),
        A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ToTensorV2(),
    ])

# =============================================================================
# MODEL
# =============================================================================
def create_deeplabv3(backbone='resnet50', num_classes=1, pretrained=True):
    """
    Create DeepLabV3+ model with ResNet backbone.
    
    Args:
        backbone: 'resnet50' or 'resnet101'
        num_classes: Number of output classes (1 for binary)
        pretrained: Use ImageNet pretrained weights
    """
    if backbone == 'resnet50':
        model = deeplabv3_resnet50(pretrained=pretrained)
    elif backbone == 'resnet101':
        model = deeplabv3_resnet101(pretrained=pretrained)
    else:
        raise ValueError(f"Unknown backbone: {backbone}")
    
    # Modify classifier for binary segmentation
    in_channels = model.classifier[4].in_channels
    model.classifier[4] = nn.Conv2d(in_channels, num_classes, kernel_size=1)
    
    # Also modify aux classifier if present
    if model.aux_classifier is not None:
        in_channels_aux = model.aux_classifier[4].in_channels
        model.aux_classifier[4] = nn.Conv2d(in_channels_aux, num_classes, kernel_size=1)
    
    return model

# =============================================================================
# LOSS FUNCTIONS
# =============================================================================
class CombinedLoss(nn.Module):
    """
    Combined BCE + Tversky Loss for thin cable segmentation.
    Tversky (alpha=0.3, beta=0.7) penalizes False Negatives more.
    """
    
    def __init__(self, bce_weight=1.0, tversky_weight=2.0, pos_weight=10.0):
        super().__init__()
        self.bce_weight = bce_weight
        self.tversky_weight = tversky_weight
        self.pos_weight = torch.tensor([pos_weight])
    
    def tversky_loss(self, pred, target, alpha=0.3, beta=0.7):
        """Tversky Loss: alpha=0.3, beta=0.7 penalizes FN more."""
        pred_sigmoid = torch.sigmoid(pred)
        pred_flat = pred_sigmoid.view(pred.size(0), -1)
        target_flat = target.view(target.size(0), -1)
        
        TP = (pred_flat * target_flat).sum(dim=1)
        FP = ((1 - target_flat) * pred_flat).sum(dim=1)
        FN = (target_flat * (1 - pred_flat)).sum(dim=1)
        
        tversky = (TP + 1e-6) / (TP + alpha * FP + beta * FN + 1e-6)
        return (1 - tversky).mean()
    
    def forward(self, pred, target):
        pos_weight = self.pos_weight.to(pred.device)
        bce_loss = F.binary_cross_entropy_with_logits(pred, target, pos_weight=pos_weight)
        tversky_loss = self.tversky_loss(pred, target)
        
        return self.bce_weight * bce_loss + self.tversky_weight * tversky_loss

# =============================================================================
# TRAINING
# =============================================================================
class Trainer:
    def __init__(self, cfg):
        self.cfg = cfg
        self.device = cfg.DEVICE
        
        # Create output directory
        cfg.OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
        
        # Model
        print(f"Creating DeepLabV3+ with {cfg.BACKBONE} backbone...")
        self.model = create_deeplabv3(
            backbone=cfg.BACKBONE,
            num_classes=cfg.NUM_CLASSES,
            pretrained=True
        ).to(self.device)
        
        # Loss
        self.criterion = CombinedLoss(
            bce_weight=cfg.BCE_WEIGHT,
            tversky_weight=cfg.TVERSKY_WEIGHT,
            pos_weight=cfg.POS_WEIGHT
        )
        
        # Optimizer
        self.optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=cfg.LR,
            weight_decay=cfg.WEIGHT_DECAY
        )
        
        # Scheduler
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer, T_max=cfg.EPOCHS
        )
        
        # AMP
        self.scaler = GradScaler() if cfg.AMP_ENABLED else None
        
        # Datasets
        self.train_loader = DataLoader(
            TTPLADataset(cfg.DATA_ROOT, 'train', get_train_transforms(cfg.IMAGE_SIZE)),
            batch_size=cfg.BATCH_SIZE,
            shuffle=True,
            num_workers=cfg.NUM_WORKERS,
            pin_memory=True
        )
        self.val_loader = DataLoader(
            TTPLADataset(cfg.DATA_ROOT, 'val', get_val_transforms(cfg.IMAGE_SIZE)),
            batch_size=cfg.BATCH_SIZE,
            shuffle=False,
            num_workers=cfg.NUM_WORKERS,
            pin_memory=True
        )
        
        # Metrics
        self.best_dice = 0.0
    
    def train_epoch(self, epoch):
        self.model.train()
        total_loss = 0
        
        pbar = tqdm(self.train_loader, desc=f"Epoch {epoch}")
        for batch in pbar:
            images = batch['image'].to(self.device)
            masks = batch['mask'].to(self.device)
            
            self.optimizer.zero_grad()
            
            if self.cfg.AMP_ENABLED:
                with autocast():
                    outputs = self.model(images)['out']
                    loss = self.criterion(outputs, masks)
                
                self.scaler.scale(loss).backward()
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                outputs = self.model(images)['out']
                loss = self.criterion(outputs, masks)
                loss.backward()
                self.optimizer.step()
            
            total_loss += loss.item()
            pbar.set_postfix({'loss': loss.item()})
        
        return total_loss / len(self.train_loader)
    
    @torch.no_grad()
    def validate(self):
        self.model.eval()
        total_loss = 0
        total_dice = 0
        
        for batch in tqdm(self.val_loader, desc="Validation"):
            images = batch['image'].to(self.device)
            masks = batch['mask'].to(self.device)
            
            with autocast(enabled=self.cfg.AMP_ENABLED):
                outputs = self.model(images)['out']
                loss = self.criterion(outputs, masks)
            
            # Compute Dice
            pred = torch.sigmoid(outputs) > 0.5
            intersection = (pred * masks).sum()
            union = pred.sum() + masks.sum()
            dice = (2 * intersection + 1e-6) / (union + 1e-6)
            
            total_loss += loss.item()
            total_dice += dice.item()
        
        return total_loss / len(self.val_loader), total_dice / len(self.val_loader)
    
    def save_checkpoint(self, epoch, dice, is_best=False):
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'dice': dice,
            'config': {
                'backbone': self.cfg.BACKBONE,
                'image_size': self.cfg.IMAGE_SIZE,
            }
        }
        
        # Save latest
        torch.save(checkpoint, self.cfg.OUTPUT_DIR / f"checkpoint_epoch_{epoch}.pth")
        
        # Save best
        if is_best:
            torch.save(checkpoint, self.cfg.OUTPUT_DIR / "best_model.pth")
            print(f"  ✅ New best model saved! Dice: {dice:.4f}")
    
    def train(self):
        print(f"\n{'='*60}")
        print(f"🚀 DeepLabV3+ Training")
        print(f"   Backbone: {self.cfg.BACKBONE}")
        print(f"   Device: {self.device}")
        print(f"   Epochs: {self.cfg.EPOCHS}")
        print(f"{'='*60}\n")
        
        for epoch in range(1, self.cfg.EPOCHS + 1):
            # Train
            train_loss = self.train_epoch(epoch)
            
            # Validate
            if epoch % self.cfg.EVAL_EVERY == 0:
                val_loss, val_dice = self.validate()
                
                print(f"\nEpoch {epoch}: Train Loss={train_loss:.4f} | Val Loss={val_loss:.4f} | Dice={val_dice:.4f}")
                
                # Save best
                if val_dice > self.best_dice:
                    self.best_dice = val_dice
                    self.save_checkpoint(epoch, val_dice, is_best=True)
            
            # Save checkpoint
            if epoch % self.cfg.SAVE_EVERY == 0:
                self.save_checkpoint(epoch, self.best_dice)
            
            # Step scheduler
            self.scheduler.step()
        
        print(f"\n✅ Training complete! Best Dice: {self.best_dice:.4f}")

# =============================================================================
# MAIN
# =============================================================================
def main():
    parser = argparse.ArgumentParser(description="Train DeepLabV3+ on TTPLA")
    parser.add_argument('--backbone', type=str, default='resnet50', 
                        choices=['resnet50', 'resnet101'])
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--batch_size', type=int, default=4)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--image_size', type=int, default=512)
    args = parser.parse_args()
    
    # Update config
    Config.BACKBONE = args.backbone
    Config.EPOCHS = args.epochs
    Config.BATCH_SIZE = args.batch_size
    Config.LR = args.lr
    Config.IMAGE_SIZE = args.image_size
    Config.OUTPUT_DIR = Path(__file__).parent.parent / "models" / f"deeplabv3_{args.backbone}"
    
    # Train
    trainer = Trainer(Config)
    trainer.train()

if __name__ == "__main__":
    main()
