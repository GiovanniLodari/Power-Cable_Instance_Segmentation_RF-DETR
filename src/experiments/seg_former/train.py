import os
import json
import argparse
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from tqdm import tqdm
import cv2
import time
from typing import Optional, Union, List

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
# from torch.cuda.amp import autocast, GradScaler # Deprecated
import albumentations as A
from albumentations.pytorch import ToTensorV2
from skimage.morphology import skeletonize

# Richiede: pip install segmentation-models-pytorch timm monai
import segmentation_models_pytorch as smp
from segmentation_models_pytorch.base import SegmentationModel, SegmentationHead, ClassificationHead
from segmentation_models_pytorch.encoders import get_encoder
from segmentation_models_pytorch.decoders.unetplusplus.decoder import UnetPlusPlusDecoder

from monai.losses import SoftclDiceLoss

# =============================================================================
# CUSTOM MODEL WRAPPER
# =============================================================================
class CustomUnetPlusPlus(SegmentationModel):
    def __init__(
        self,
        encoder_name: str = "resnet34",
        encoder_depth: int = 5,
        encoder_weights: Optional[str] = "imagenet",
        decoder_use_norm: Union[bool, str, dict] = "batchnorm",
        decoder_channels: List[int] = (256, 128, 64, 32, 16),
        decoder_attention_type: Optional[str] = None,
        decoder_interpolation: str = "nearest",
        in_channels: int = 3,
        classes: int = 1,
        activation: Optional[Union[str, callable]] = None,
        aux_params: Optional[dict] = None,
        **kwargs
    ):
        super().__init__()
        
        # 1. Inizializza l'encoder
        self.encoder = get_encoder(
            encoder_name,
            in_channels=in_channels,
            depth=encoder_depth,
            weights=encoder_weights,
            **kwargs,
        )

        # 2. FIX SICURO PER I CANALI
        # Invece di inserire '0', leggiamo esattamente cosa ci dà l'encoder
        # e configuriamo il decoder di conseguenza.
        enc_channels = list(self.encoder.out_channels)
        
        # FIX: Force correct channels for mit_b5 to avoid 0-dim channels causing crashes
        if encoder_name == "mit_b5":
             enc_channels = [3, 64, 128, 320, 512]
        
        # Se l'encoder ha meno stadi di quanto richiesto da 'encoder_depth',
        # riduciamo la profondità del decoder invece di inventare canali vuoti.
        actual_depth = len(enc_channels) - 1
        if actual_depth != encoder_depth:
            print(f"⚠️ Warning: Encoder has {actual_depth} stages, but depth={encoder_depth} was requested.")
            print(f"   Adjusting decoder to depth={actual_depth} to prevent crash.")
            encoder_depth = actual_depth
            # Tagliamo anche i canali del decoder se sono troppi
            decoder_channels = decoder_channels[:actual_depth]

        self.decoder = UnetPlusPlusDecoder(
            encoder_channels=tuple(enc_channels),
            decoder_channels=decoder_channels,
            n_blocks=encoder_depth, # Usa la profondità reale!
            use_norm=decoder_use_norm,
            center=True if encoder_name.startswith("vgg") else False,
            attention_type=decoder_attention_type,
            interpolation_mode=decoder_interpolation,
        )

        self.segmentation_head = SegmentationHead(
            in_channels=decoder_channels[-1],
            out_channels=classes,
            activation=activation,
            kernel_size=3,
            # MiT parte da stride 4. Unet++ con depth 4 finisce a stride 4.
            # Dobbiamo fare upsampling 4x alla fine per tornare a 700x700.
            upsampling=4 if "mit" in encoder_name else 1 
        )

        if aux_params is not None:
            self.classification_head = ClassificationHead(
                in_channels=self.encoder.out_channels[-1], **aux_params
            )
        else:
            self.classification_head = None

        self.name = "unetplusplus-{}".format(encoder_name)
        self.initialize()


# =============================================================================
# CONFIGURATION
# =============================================================================
class Config:
    PROJECT_ROOT = Path(__file__).resolve().parents[3]
    DATA_ROOT = PROJECT_ROOT / "data"
    OUTPUT_DIR = PROJECT_ROOT / "models" / "segformer_mit_b5_sota"
    
    # Architecture
    ENCODER = "mit_b5" 
    ENCODER_WEIGHTS = "imagenet"
    
    # Training Hyperparams
    NUM_CLASSES = 1
    INPUT_SIZE = 700 
    TRAIN_SIZE = 704
    TRAIN_CROP_SIZE = 512 # Optimization: Train on smaller crops, Val on full
    BATCH_SIZE = 2 
    NUM_WORKERS = 4 
    EPOCHS = 80
    VAL_INTERVAL = 5 # Validation frequency
    
    # Optimizer / Scheduler
    LR = 6e-5 
    WEIGHT_DECAY = 1e-2
    
    # Loss Balancing
    TVERSKY_ALPHA = 0.2 
    TVERSKY_BETA = 0.8 
    
    # Topology Preservation (Optional/Advanced)
    # WARNING: High compute cost. Can cause instability.
    # Keep 0.0 to disable. Try 0.1 if AP is low due to fragmentation.
    CLDICE_WEIGHT = 0.1 
    CLDICE_START_EPOCH = 30 # Start later to avoid instability
    
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    AMP_ENABLED = True
    
    SAVE_EVERY = 5
    
    DEBUG = False

# =============================================================================
# DATASET
# =============================================================================
class TTPLADataset(Dataset):
    def __init__(self, root_dir, split='train', transform=None):
        self.root_dir = Path(root_dir)
        self.transform = transform
        
        if split == 'train':
            json_path = self.root_dir / 'combined' / 'train_combined.json'
            self.img_dir = self.root_dir / 'all_images'
            if not json_path.exists():
                json_path = self.root_dir / 'train' / 'train.json'
                self.img_dir = self.root_dir / 'train'
        elif split == 'val':
            json_path = self.root_dir / 'combined' / 'val_combined.json'
            self.img_dir = self.root_dir / 'all_images'
            if not json_path.exists():
                json_path = self.root_dir / 'train' / 'val.json'
                if not json_path.exists():
                     json_path = self.root_dir / 'val' / 'val.json'
                     self.img_dir = self.root_dir / 'val'
        
        print(f"Loading {split} annotations from: {json_path}")
        with open(json_path, 'r') as f:
            coco = json.load(f)
            
        self.images = sorted(coco['images'], key=lambda x: x['id'])
        self.img_to_anns = {}
        for ann in coco['annotations']:
            self.img_to_anns.setdefault(ann['image_id'], []).append(ann)

    def __len__(self):
        return len(self.images)
    
    def __getitem__(self, idx):
        img_info = self.images[idx]
        img_path = self.img_dir / img_info['file_name']
        
        image = cv2.imread(str(img_path))
        if image is None:
            image = np.zeros((700, 700, 3), dtype=np.uint8)
        else:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        h, w = img_info['height'], img_info['width']
        mask = np.zeros((h, w), dtype=np.float32)
        
        for ann in self.img_to_anns.get(img_info['id'], []):
            for poly in ann.get('segmentation', []):
                pts = np.array(poly).reshape(-1, 2).astype(np.int32)
                cv2.fillPoly(mask, [pts], 1.0)
                
        if self.transform:
            transformed = self.transform(image=image, mask=mask)
            image = transformed['image']
            mask = transformed['mask']
            
            
        # Ensure mask is (1, H, W) tensor
        if isinstance(mask, torch.Tensor):
            if mask.ndim == 2:
                mask = mask.unsqueeze(0)
        else:
            mask = torch.tensor(mask).unsqueeze(0)

        # Ensure mask is float for SoftclDiceLoss/MixPrecision
        mask = mask.float()

        return {
            'image': image,
            'mask': mask
        }

# =============================================================================
# AUGMENTATIONS
# =============================================================================
def get_train_transforms():
    return A.Compose([
        A.PadIfNeeded(min_height=Config.TRAIN_CROP_SIZE, min_width=Config.TRAIN_CROP_SIZE, 
                      border_mode=cv2.BORDER_CONSTANT, value=0),
        A.RandomCrop(height=Config.TRAIN_CROP_SIZE, width=Config.TRAIN_CROP_SIZE),
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.5),
        A.RandomRotate90(p=0.5),
        A.ShiftScaleRotate(shift_limit=0.05, scale_limit=0.05, rotate_limit=15, p=0.4),
        A.OneOf([
            A.RandomBrightnessContrast(),
            A.HueSaturationValue(),
            A.GaussNoise(var_limit=(10.0, 30.0)),
        ], p=0.4),
        A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ToTensorV2(),
    ])

def get_val_transforms():
    return A.Compose([
        A.PadIfNeeded(min_height=Config.TRAIN_SIZE, min_width=Config.TRAIN_SIZE, 
                      border_mode=cv2.BORDER_CONSTANT, value=0),
        A.CenterCrop(height=Config.TRAIN_SIZE, width=Config.TRAIN_SIZE),
        A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ToTensorV2(),
    ])

# =============================================================================
# METRICS
# =============================================================================
def calculate_proxy_metrics(pred_logits, target_mask):
    pred_prob = torch.sigmoid(pred_logits)
    pred_bin = (pred_prob > 0.5).detach().cpu().numpy().astype(np.uint8)
    target_bin = target_mask.detach().cpu().numpy().astype(np.uint8)
    
    ious, aps, ars = [], [], []
    
    for i in range(pred_bin.shape[0]):
        p = pred_bin[i, 0]
        t = target_bin[i, 0]
        
        # IoU
        inter = np.logical_and(p, t).sum()
        union = np.logical_or(p, t).sum()
        ious.append((inter + 1e-6) / (union + 1e-6))
        
        # Instance Approximation
        n_labels_p, labels_p = cv2.connectedComponents(p, connectivity=8)
        n_labels_t, labels_t = cv2.connectedComponents(t, connectivity=8)
        
        # AR
        gt_objects = n_labels_t - 1
        found = 0
        if gt_objects > 0:
            for lbl in range(1, n_labels_t):
                gt_mask = (labels_t == lbl)
                overlap = np.logical_and(gt_mask, p).sum()
                if overlap / gt_mask.sum() > 0.5:
                    found += 1
            ars.append(found / gt_objects)
        else:
            ars.append(1.0 if (n_labels_p - 1) == 0 else 0.0)
            
        # AP
        pred_objects = n_labels_p - 1
        correct = 0
        if pred_objects > 0:
            for lbl in range(1, n_labels_p):
                p_mask = (labels_p == lbl)
                overlap = np.logical_and(p_mask, t).sum()
                if overlap / p_mask.sum() > 0.5:
                    correct += 1
            aps.append(correct / pred_objects)
        else:
            aps.append(1.0 if gt_objects == 0 else 0.0)
            
    return np.mean(ious), np.mean(aps), np.mean(ars)

# =============================================================================
# TRAINER
# =============================================================================
class Trainer:
    def __init__(self):
        Config.OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
        self.device = Config.DEVICE
        
        # Optimization: Enable cuDNN benchmark
        torch.backends.cudnn.benchmark = True
        
        print(f"🏗️  Model: U-Net++ | Encoder: {Config.ENCODER} | Attn: SCSE")
        
        # CustomUnetPlusPlus handles the MiT encoder compatibility
        self.model = CustomUnetPlusPlus(
            encoder_name=Config.ENCODER,
            encoder_weights=Config.ENCODER_WEIGHTS,
            in_channels=3,
            classes=1,
            # We set depth 4 because MiT has 4 blocks. 
            # If we set 5, we'll just replicate the last one or need padding.
            encoder_depth=5,
            decoder_channels=(256, 128, 64, 32, 16),
            activation=None,
            decoder_attention_type="scse" 
        ).to(self.device)
        
        # Optimization: Torch Compile (PyTorch 2.0+)
        if hasattr(torch, 'compile'):
            print("🚀 Compiling model with torch.compile()...")
            try:
                self.model = torch.compile(self.model, mode="reduce-overhead")
            except Exception as e:
                print(f"⚠️ torch.compile failed: {e}. Continuing without compilation.")
        
        self.criterion_tversky = smp.losses.TverskyLoss(
            mode='binary', alpha=Config.TVERSKY_ALPHA, beta=Config.TVERSKY_BETA, from_logits=True
        )
        self.criterion_focal = smp.losses.FocalLoss(mode='binary', gamma=2.0)
        
        if Config.CLDICE_WEIGHT > 0:
            print(f"🔗 Enabling clDice Loss (Weight: {Config.CLDICE_WEIGHT}) | Backend: MONAI")
            self.criterion_cldice = SoftclDiceLoss(iter_=3, smooth=1.0)
        else:
            self.criterion_cldice = None
        
        self.optimizer = torch.optim.AdamW(
            self.model.parameters(), lr=Config.LR, weight_decay=Config.WEIGHT_DECAY
        )
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer, T_max=Config.EPOCHS, eta_min=1e-7
        )
        # mixed precision
        self.scaler = torch.amp.GradScaler('cuda') if Config.AMP_ENABLED else None
        
        self.train_loader = DataLoader(
            TTPLADataset(Config.DATA_ROOT, 'train', get_train_transforms()),
            batch_size=Config.BATCH_SIZE, shuffle=True, 
            num_workers=Config.NUM_WORKERS, pin_memory=True, drop_last=True,
            persistent_workers=True if Config.NUM_WORKERS > 0 else False
        )
        self.val_loader = DataLoader(
            TTPLADataset(Config.DATA_ROOT, 'val', get_val_transforms()),
            batch_size=Config.BATCH_SIZE, shuffle=False, 
            num_workers=Config.NUM_WORKERS, pin_memory=True
        )
        
        self.history = {'iou': [], 'ap': [], 'ar': [], 'loss': []}
        self.best_score = 0.0

    def train_one_epoch(self, current_epoch):
        self.model.train()
        total_loss = 0
        pbar = tqdm(self.train_loader, desc="Train")
        steps = 0
        for i, batch in enumerate(pbar):
            if Config.DEBUG and i >= 10:
                print("Debug mode: breaking training loop")
                break
                
            imgs = batch['image'].to(self.device)
            masks = batch['mask'].to(self.device)
            
            self.optimizer.zero_grad()
            with torch.amp.autocast('cuda', enabled=Config.AMP_ENABLED):
                preds = self.model(imgs)
                # Output might be smaller due to depth/stride logic
                if preds.shape[-2:] != masks.shape[-2:]:
                    preds = F.interpolate(preds, size=masks.shape[-2:], mode='bilinear', align_corners=False)
                
                loss = 0.7 * self.criterion_tversky(preds, masks) + \
                       0.3 * self.criterion_focal(preds, masks)
                
                # Apply clDice only if enabled and epoch >= start_epoch
                if self.criterion_cldice is not None and current_epoch >= Config.CLDICE_START_EPOCH:
                     # MONAI SoftclDiceLoss expects probabilities in [0, 1]
                     # It calculates skeletons internally on-the-fly.
                     preds_probs = torch.sigmoid(preds)
                     # Note: Signature might vary, assuming (pred, target) or (target, pred).
                     # Standard MONAI metrics often are (y_pred, y_true).
                     # Based on docstring "SoftclDiceLoss(y_true, y_pred)" we try `criterion(masks, preds_probs)`
                     # BUT usually losses are criterion(input, target). 
                     # Let's try standard PyTorch order (input, target) first unless verified otherwise.
                     # Re-reading: inspect said (self, y_true, y_pred). So we use (masks, preds_probs).
                     # FIX: Use keyword arguments to be 100% safe and avoid positional confusion.
                     # FIX: MONAI SoftclDiceLoss ignores channel 0 ([:, 1:]).
                     # We must provide (B, 2, H, W) -> [Background, Foreground].
                     
                     # 1. Expand Preds to (B, 2, H, W)
                     # preds is logits (B, 1, H, W). 
                     # We need probs for both channels.
                     pred_prob_fg = torch.sigmoid(preds)
                     pred_prob_bg = 1.0 - pred_prob_fg
                     y_pred_2ch = torch.cat([pred_prob_bg, pred_prob_fg], dim=1)
                     
                     # 2. Expand Masks to (B, 2, H, W)
                     # masks is (B, 1, H, W)
                     mask_fg = masks
                     mask_bg = 1.0 - mask_fg
                     y_true_2ch = torch.cat([mask_bg, mask_fg], dim=1)
                     
                     loss += Config.CLDICE_WEIGHT * self.criterion_cldice(y_true=y_true_2ch, y_pred=y_pred_2ch)
            
            self.scaler.scale(loss).backward()
            self.scaler.step(self.optimizer)
            self.scaler.update()
            
            total_loss += loss.item()
            pbar.set_postfix({'loss': loss.item()})
            steps += 1
            
        return total_loss / max(steps, 1)

    @torch.no_grad()
    def validate(self, current_epoch):
        self.model.eval()
        ious, aps, ars = [], [], []
        pbar = tqdm(self.val_loader, desc="Val")
        for i, batch in enumerate(pbar):
            if Config.DEBUG and i >= 10:
                print("Debug mode: breaking validation loop")
                break
            
            imgs = batch['image'].to(self.device)
            masks = batch['mask'].to(self.device)
            # Forward pass
            with torch.amp.autocast('cuda', enabled=Config.AMP_ENABLED):
                preds = self.model(imgs)
                if preds.shape[-2:] != masks.shape[-2:]:
                    preds = F.interpolate(preds, size=masks.shape[-2:], mode='bilinear', align_corners=False)
            
            # DEBUG: Check prediction stats for first batch of valid
            if i == 0:
                prob = torch.sigmoid(preds)
                # Count pixels above thresholds
                p10 = (prob > 0.1).float().mean().item() * 100
                p30 = (prob > 0.3).float().mean().item() * 100
                p50 = (prob > 0.5).float().mean().item() * 100
                print(f" [Val Debug] Min={prob.min():.4f}, Max={prob.max():.4f}, Mean={prob.mean():.4f}")
                print(f"             Pixels >0.1: {p10:.2f}% | >0.3: {p30:.2f}% | >0.5: {p50:.2f}% | Mask Sum={masks.sum().item()}")

                # VISUAL DEBUG: Save Image + Pred + GT
                debug_img = imgs[0].detach().cpu().permute(1, 2, 0).numpy()
                # Denormalize
                mean = np.array([0.485, 0.456, 0.406])
                std = np.array([0.229, 0.224, 0.225])
                debug_img = std * debug_img + mean
                debug_img = np.clip(debug_img, 0, 1)
                
                debug_pred = prob[0, 0].detach().cpu().numpy()
                debug_gt = masks[0, 0].detach().cpu().numpy()
                
                plt.figure(figsize=(15, 5))
                plt.subplot(1, 3, 1)
                plt.imshow(debug_img)
                plt.title("Val Image")
                plt.subplot(1, 3, 2)
                plt.imshow(debug_gt, cmap='gray')
                plt.title(f"GT Mask (Sum: {debug_gt.sum():.0f})")
                plt.subplot(1, 3, 3)
                plt.imshow(debug_pred, cmap='jet', vmin=0, vmax=1)
                plt.title(f"Pred Prob (Max: {debug_pred.max():.2f})")
                plt.tight_layout()
                plt.savefig(Config.OUTPUT_DIR / f"val_debug_epoch_{current_epoch}.png")
                plt.close()
            
            biou, bap, bar = calculate_proxy_metrics(preds, masks)
            ious.append(biou)
            aps.append(bap)
            ars.append(bar)
            
        return np.mean(ious), np.mean(aps), np.mean(ars)

    def plot_metrics(self):
        plt.figure(figsize=(12, 4))
        # Ensure that history lists are not empty before plotting
        if not self.history['loss']:
            print("No loss data to plot.")
            return
        
        epochs_loss = range(1, len(self.history['loss']) + 1)
        
        plt.subplot(1, 2, 1)
        plt.plot(epochs_loss, self.history['loss'], label='Loss')
        plt.title('Training Loss')
        plt.grid(True)
        
        if self.history['iou']: # Check if validation metrics exist
            epochs_val = range(1, len(self.history['iou']) + 1)
            plt.subplot(1, 2, 2)
            plt.plot(epochs_val, self.history['iou'], label='IoU', color='green')
            plt.plot(epochs_val, self.history['ap'], label='AP@50', color='blue')
            plt.plot(epochs_val, self.history['ar'], label='AR@50', color='orange')
            plt.plot(epochs_val, [x+y for x,y in zip(self.history['ap'], self.history['ar'])], 
                     label='Sum', linestyle='--', color='red')
            plt.title('Validation Metrics')
            plt.legend()
            plt.grid(True)
        else:
            plt.subplot(1, 2, 2)
            plt.text(0.5, 0.5, "No validation data to plot.", horizontalalignment='center', verticalalignment='center', transform=plt.gca().transAxes)
            plt.title('Validation Metrics')
        
        plt.tight_layout()
        plt.savefig(Config.OUTPUT_DIR / "metrics.png")
        plt.close()

    def run(self):
        if Config.DEBUG:
            print(f"🚀 Start Training in DEBUG mode (limited batches)")
        else:
            print(f"🚀 Start Training: {Config.EPOCHS} epochs")
            
        for epoch in range(1, Config.EPOCHS + 1):
            print(f"\n--- Epoch {epoch} ---")
            loss = self.train_one_epoch(epoch)
            self.history['loss'].append(loss)
            self.scheduler.step()
            
            # Validation every VAL_INTERVAL epochs (always validate first and last)
            if epoch == 1 or epoch % Config.VAL_INTERVAL == 0 or epoch == Config.EPOCHS:
                val_metrics = self.validate(epoch)
                self.history['iou'].append(val_metrics[0])
                self.history['ap'].append(val_metrics[1])
                self.history['ar'].append(val_metrics[2])
                
                sum_score = val_metrics[1] + val_metrics[2] # AP + AR
                print(f"📊 IoU: {val_metrics[0]:.4f} | AP: {val_metrics[1]:.4f} | AR: {val_metrics[2]:.4f} | SUM: {sum_score:.4f}")
                self.plot_metrics()
                
                # Save best based on AP
                if val_metrics[1] > self.best_score:
                    self.best_score = val_metrics[1]
                    torch.save(self.model.state_dict(), Config.OUTPUT_DIR / "best_model.pth")
                    print(f"🔥 New Best AP: {self.best_score:.4f} (Saved)")
                
            # Always save checkpoint (optional: every epoch or interval)
            # torch.save(self.model.state_dict(), Config.OUTPUT_DIR / "last_model.pth")
            
            if not Config.DEBUG and epoch % Config.SAVE_EVERY == 0:
                torch.save(self.model.state_dict(), Config.OUTPUT_DIR / f"epoch_{epoch}.pth")
            
            if Config.DEBUG:
                print("Debug run completed.")
                break

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--debug', action='store_true', help='Limit training and validation to 10 batches.')
    args = parser.parse_args()
    
    Config.DEBUG = args.debug
    
    Trainer().run()
