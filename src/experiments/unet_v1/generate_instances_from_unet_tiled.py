#!/usr/bin/env python3
"""
=========================================================
Run inference on a set of images using UNet++ V1 model.
=========================================================
"""

import os
import gc
import json
import cv2
import numpy as np
import torch
import argparse
import subprocess
import sys
import shutil
import segmentation_models_pytorch as smp
import albumentations as A
from albumentations.pytorch import ToTensorV2
from pathlib import Path
from tqdm import tqdm
from scipy import ndimage
from pycocotools import mask as mask_util

# =============================================================================
# CONFIGURATION
# =============================================================================
PROJECT_ROOT = Path(__file__).resolve().parent.parent
MODEL_PATH = PROJECT_ROOT / "models" / "backups" / "unetpp_100epochs.pth"
DATA_JSON = PROJECT_ROOT / "data" / "combined" / "val_combined.json"
OUTPUT_JSON = PROJECT_ROOT / "models" / "predictions_unet_instances.json"
TEMP_DIR = PROJECT_ROOT / "models" / "temp_preds_user"

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
INFERENCE_SIZE = 704 # Key to user's success?
THRESHOLD = 0.97

# =============================================================================
# HELPER FUNCTIONS (WORKER)
# =============================================================================
# =============================================================================
# HELPER FUNCTIONS (WORKER)
# =============================================================================
def load_model():
    print(f"   📦 (Worker) Loading model...")
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

def predict_crop(model, image_crop):
    """
    Takes an image crop, resizes to INFERENCE_SIZE, predicts, 
    and returns probability map resized back to crop size.
    """
    orig_h, orig_w = image_crop.shape[:2]
    
    transform = A.Compose([
        A.Resize(INFERENCE_SIZE, INFERENCE_SIZE),
        A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ToTensorV2(),
    ])
    
    transformed = transform(image=image_crop)
    img_tensor = transformed['image'].unsqueeze(0).to(DEVICE)
    
    with torch.no_grad():
        logits = model(img_tensor)
        prob = torch.sigmoid(logits).squeeze().cpu().numpy()
        
    # Resize back to crop original dimensions
    prob_orig = cv2.resize(prob, (orig_w, orig_h), interpolation=cv2.INTER_LINEAR)
    return prob_orig

def compute_line_params_user_formula(mask):
    """Exact copy of phase2_worker.py formula"""
    if mask.sum() < 10:
        return None
    coords = np.column_stack(np.where(mask > 0)) # (N, 2) -> (y, x)
    
    if len(coords) < 3:
        return None
    centroid = coords.mean(axis=0)
    try:
        _, _, Vt = np.linalg.svd(coords - centroid)
        theta = np.arctan2(Vt[0, 1], Vt[0, 0]) 
        
        if theta < 0:
            theta += np.pi
            
        rho = abs(centroid[0] * np.cos(theta) + centroid[1] * np.sin(theta))
        return [float(rho), float(theta)]
    except:
        return None

def extract_instances(prob_map, img_id, start_id, height, width):
    instances = []
    
    # 1. Threshold
    binary = (prob_map > THRESHOLD).astype(np.uint8)
    
    # Check minimum area
    if binary.sum() < 50:
        return instances, start_id
        
    labeled, num_features = ndimage.label(binary)
    slices = ndimage.find_objects(labeled)
    
    for i, slice_obj in enumerate(slices):
        if slice_obj is None: continue
        
        y_slice, x_slice = slice_obj
        labeled_crop = labeled[y_slice, x_slice]
        label_id = i + 1
        component_mask_crop = (labeled_crop == label_id).astype(np.uint8)
        
        area = component_mask_crop.sum()
        if area < 50: continue
        
        offset = (y_slice.start, x_slice.start)
        prob_crop = prob_map[y_slice, x_slice]
        
        # User Score: mean of prob under mask
        score = float(prob_crop[component_mask_crop > 0].mean())
        if score < 0.3: continue
        
        # User Params (Global Coords)
        y_local, x_local = np.where(component_mask_crop > 0)
        y_global = y_local + offset[0]
        x_global = x_local + offset[1]
        coords = np.column_stack((y_global, x_global))
        
        if len(coords) < 3: lines = [0.0, 0.0]
        else:
             centroid = coords.mean(axis=0)
             try:
                 _, _, Vt = np.linalg.svd(coords - centroid)
                 theta = np.arctan2(Vt[0, 1], Vt[0, 0])
                 if theta < 0: theta += np.pi
                 rho = abs(centroid[0] * np.cos(theta) + centroid[1] * np.sin(theta))
                 lines = [float(rho), float(theta)]
             except:
                 lines = [0.0, 0.0]
        
        # RLE output
        full_mask = np.zeros((height, width), dtype=np.uint8, order='F')
        full_mask[y_slice, x_slice] = component_mask_crop
        rle = mask_util.encode(full_mask)
        rle['counts'] = rle['counts'].decode('utf-8')
        del full_mask
        
        # Bbox
        rmin, rmax = y_global.min(), y_global.max()
        cmin, cmax = x_global.min(), x_global.max()
        bbox = [float(cmin), float(rmin), float(cmax - cmin + 1), float(rmax - rmin + 1)]
        
        instances.append({
            "id": start_id + i,
            "image_id": img_id,
            "category_id": 0,
            "bbox": bbox,
            "segmentation": rle,
            "area": float(area),
            "score": score,
            "lines": lines
        })
        
    return instances, start_id + len(slices)

# =============================================================================
# WORKER MODE
# =============================================================================
def run_worker(args):
    """Single image processing with 2x2 Tiling."""
    try:
        model = load_model()
        
        # Resolve image path
        img_path = Path(args.img_path)
        if not img_path.exists():
            possible_dirs = [
                PROJECT_ROOT / "data" / "val",
                PROJECT_ROOT / "data" / "train",
                PROJECT_ROOT / "data" / "all_images",
                PROJECT_ROOT / "data" / "combined" / "val"
            ]
            for d in possible_dirs:
               p = d / img_path.name
               if p.exists():
                   img_path = p
                   break
        
        image = cv2.imread(str(img_path))
        if image is None: return
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        h, w = image.shape[:2]
        
        # TILING LOGIC (2x2 Split)
        mid_h, mid_w = h // 2, w // 2
        
        # Define 4 crops: (y_start, y_end, x_start, x_end)
        crops_coords = [
            (0, mid_h, 0, mid_w),        # Top-Left
            (0, mid_h, mid_w, w),        # Top-Right
            (mid_h, h, 0, mid_w),        # Bottom-Left
            (mid_h, h, mid_w, w)         # Bottom-Right
        ]
        
        # Full Stitch Canvas
        full_prob_map = np.zeros((h, w), dtype=np.float32)
        
        for (y1, y2, x1, x2) in crops_coords:
            crop = image[y1:y2, x1:x2]
            if crop.size == 0: continue
            
            # Predict and Resize back to crop size
            prob_crop = predict_crop(model, crop)
            
            # Place in canvas
            full_prob_map[y1:y2, x1:x2] = prob_crop
            
        
        # Post-process on stitched map
        instances, _ = extract_instances(full_prob_map, args.img_id, args.start_id, h, w)
        
        # Save
        out_file = TEMP_DIR / f"{args.img_id}.json"
        with open(out_file, 'w') as f:
            json.dump(instances, f)
            
        print(f"   ✅ Saved {len(instances)} instances")
        
    except Exception as e:
        print(f"   ❌ Error in worker: {e}")
        import traceback
        traceback.print_exc()

# =============================================================================
# ORCHESTRATOR MODE
# =============================================================================
def run_orchestrator(args):
    print(f"🚀 Starting Orchestrator (User Baseline: Resize {args.size}, Threshold {args.threshold})")
    
    TEMP_DIR.mkdir(parents=True, exist_ok=True)
    
    # Resume check
    existing_ids = set()
    for f in TEMP_DIR.glob("*.json"):
        try: existing_ids.add(int(f.stem))
        except: pass
            
    if existing_ids:
        print(f"   🔄 Resuming: Found {len(existing_ids)} completed.")
    else:
        print("   🆕 Starting fresh.")
        
    with open(DATA_JSON, 'r') as f:
        coco_data = json.load(f)
    
    images = coco_data['images']
    
    pbar = tqdm(images, desc="Processing")
    for img_info in pbar:
        img_id = img_info['id']
        if img_id in existing_ids: continue
        fname = img_info['file_name']
        cmd = [
            sys.executable, __file__, 
            "--mode", "worker", 
            "--img_path", fname, 
            "--img_id", str(img_id),
            "--threshold", str(args.threshold),
            "--size", str(args.size)
        ]
        result = subprocess.run(cmd, capture_output=True, text=True)
        if result.returncode != 0:
            print(f"\n❌ Fail {fname}: {result.stderr}")
            
    # Merge
    print(f"\n🧩 Merging results...")
    all_instances = []
    final_id = 1
    for img_info in images:
        temp_file = TEMP_DIR / f"{img_info['id']}.json"
        if temp_file.exists():
            with open(temp_file, 'r') as f:
                insts = json.load(f)
            for inst in insts:
                inst['id'] = final_id
                final_id += 1
                all_instances.append(inst)
                
    print(f"💾 Saving {len(all_instances)} predictions to {OUTPUT_JSON}...")
    with open(OUTPUT_JSON, 'w') as f:
        json.dump(all_instances, f)
    # shutil.rmtree(TEMP_DIR)
    print("✅ Done.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["orchestrator", "worker"], default="orchestrator")
    parser.add_argument("--img_path", type=str)
    parser.add_argument("--img_id", type=int)
    parser.add_argument("--start_id", type=int, default=0)
    parser.add_argument("--threshold", type=float, default=0.97, help="Probability threshold")
    parser.add_argument("--size", type=int, default=704, help="Inference size")
    
    args = parser.parse_args()
    
    # Update Globals from Args
    THRESHOLD = args.threshold
    INFERENCE_SIZE = args.size
    
    # Unique Output/Temp for this config
    config_str = f"tiled_2x2_t{int(THRESHOLD*100)}_s{INFERENCE_SIZE}"
    OUTPUT_JSON = PROJECT_ROOT / "models" / f"preds_unet_{config_str}.json"
    TEMP_DIR = PROJECT_ROOT / "models" / f"temp_preds_{config_str}"

    if args.mode == "worker":
        run_worker(args)
    else:
        run_orchestrator(args)
