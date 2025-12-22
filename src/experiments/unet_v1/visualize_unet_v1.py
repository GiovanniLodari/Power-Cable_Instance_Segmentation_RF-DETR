#!/usr/bin/env python3
"""
Visualize U-Net V1 Predictions (4-Panel Style)
==============================================
Restored script to generate:
1. Input Image
2. Ground Truth Mask
3. Prediction Heatmap
4. Overlay (Red=Pred, Green=GT)
"""

import os
import cv2
import json
import torch
import numpy as np
import argparse
import random
from pathlib import Path
from tqdm import tqdm
import segmentation_models_pytorch as smp
import albumentations as A
from albumentations.pytorch import ToTensorV2
from pycocotools import mask as mask_util

# CONFIG
PROJECT_ROOT = Path(__file__).resolve().parents[3]
MODEL_PATH = PROJECT_ROOT / "models" / "unetpp_resnet50" / "checkpoint_epoch_100.pth"
DATA_JSON = PROJECT_ROOT / "data" / "combined" / "val_combined.json"
OUTPUT_DIR = PROJECT_ROOT / "visualizations" / "unet_v1_restored"

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
INFERENCE_SIZE = 704

def load_model():
    print(f"Loading model from {MODEL_PATH}...")
    model = smp.UnetPlusPlus(
        encoder_name='resnet50',
        encoder_weights=None,
        in_channels=3,
        classes=1,
        activation=None
    )
    checkpoint = torch.load(MODEL_PATH, map_location=DEVICE, weights_only=False)
    state_dict = checkpoint.get('model_state_dict', checkpoint)
    model.load_state_dict(state_dict)
    model = model.to(DEVICE).eval()
    return model

def predict(model, image):
    orig_h, orig_w = image.shape[:2]
    transform = A.Compose([
        A.Resize(INFERENCE_SIZE, INFERENCE_SIZE),
        A.Normalize(),
        ToTensorV2(),
    ])
    transformed = transform(image=image)
    img_tensor = transformed['image'].unsqueeze(0).to(DEVICE)
    
    with torch.no_grad():
        logits = model(img_tensor)
        prob = torch.sigmoid(logits).squeeze().cpu().numpy()
        
    prob_orig = cv2.resize(prob, (orig_w, orig_h), interpolation=cv2.INTER_LINEAR)
    return prob_orig

def rle_to_mask(rle):
    if isinstance(rle, dict):
        return mask_util.decode(rle)
    return None

def polygons_to_mask(polygons, height, width):
    mask = np.zeros((height, width), dtype=np.uint8)
    for polygon in polygons:
        pts = np.array(polygon).reshape((-1, 2)).astype(np.int32)
        cv2.fillPoly(mask, [pts], color=1)
    return mask

def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    # Load Model
    model = load_model()
    
    # Load Data
    with open(DATA_JSON, 'r') as f:
        data = json.load(f)
        
    images = {img['id']: img for img in data['images']}
    annotations = data['annotations']
    
    # Group GT by image
    gt_by_img = {}
    for ann in annotations:
        img_id = ann['image_id']
        if img_id not in gt_by_img: gt_by_img[img_id] = []
        gt_by_img[img_id].append(ann)
        
    # Select samples (random)
    sample_ids = random.sample(list(images.keys()), 10)
    
    print(f"Visualizing {len(sample_ids)} samples...")
    
    for img_id in tqdm(sample_ids):
        img_info = images[img_id]
        fname = img_info['file_name']
        
        # Path resolution
        img_path = PROJECT_ROOT / "data" / fname
        if not img_path.exists():
             img_path = PROJECT_ROOT / "data" / "all_images" / os.path.basename(fname)
             
        if not img_path.exists():
            continue
            
        # Read Image
        img = cv2.imread(str(img_path)) # BGR
        if img is None: continue
        h, w = img.shape[:2]
        
        # 1. Ground Truth Mask
        gt_mask = np.zeros((h, w), dtype=np.uint8)
        if img_id in gt_by_img:
            for ann in gt_by_img[img_id]:
                if isinstance(ann['segmentation'], list):
                    m = polygons_to_mask(ann['segmentation'], h, w)
                else:
                    m = rle_to_mask(ann['segmentation'])
                gt_mask = np.maximum(gt_mask, m)
                
        # 2. Predict
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        prob_map = predict(model, img_rgb)
        pred_mask = (prob_map > 0.5).astype(np.uint8)
        
        # 3. Create Visualizations
        
        # Panel 1: Input
        p1 = img.copy()
        cv2.putText(p1, "Input", (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (255, 255, 255), 3)
        
        # Panel 2: GT (White on Black)
        p2 = np.zeros_like(img)
        p2[gt_mask > 0] = [255, 255, 255]
        cv2.putText(p2, "Ground Truth", (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (255, 255, 255), 3)
        
        # Panel 3: Prediction (Heatmap)
        heatmap = cv2.applyColorMap((prob_map * 255).astype(np.uint8), cv2.COLORMAP_INFERNO)
        p3 = heatmap
        cv2.putText(p3, "Prediction (Heatmap)", (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (255, 255, 255), 3)
        
        # Panel 4: Overlay (Red=Pred, Green=GT)
        # Create blank canvas
        overlay = img.copy() * 0.5 # Dim original
        
        # Paint Green GT
        overlay[gt_mask > 0] = [0, 255, 0] # BGR
        # Paint Red Pred (on top? or mix?)
        # Let's mix: Red=Pred
        overlay[pred_mask > 0] = np.where(overlay[pred_mask > 0] == [0, 255, 0], [0, 255, 255], [0, 0, 255]) # Yellow if overlap, Red if Pred only?
        # Actually standard: Green=GT, Red=Pred. Overlap=Yellow.
        
        # Re-do strictly
        p4 = img.copy()
        # Red channel for Pred
        # Green channel for GT
        # If both, Yellow.
        
        # Mask overlay logic
        colored_mask = np.zeros_like(img)
        colored_mask[gt_mask > 0, 1] = 255 # Green
        colored_mask[pred_mask > 0, 2] = 255 # Red
        
        # Add to image
        p4 = cv2.addWeighted(p4, 0.6, colored_mask, 0.4, 0)
        cv2.putText(p4, "Red=Pred, Green=GT", (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (255, 255, 255), 3)
        
        # Concatenate 2x2
        # Top: P1 P2
        # Bot: P3 P4
        top = np.hstack((p1, p2))
        bot = np.hstack((p3, p4))
        final = np.vstack((top, bot))
        
        # Save
        out_name = OUTPUT_DIR / f"viz_{os.path.basename(fname)}"
        cv2.imwrite(str(out_name), final)
        
    print(f"Done! Check {OUTPUT_DIR}")

if __name__ == "__main__":
    main()
