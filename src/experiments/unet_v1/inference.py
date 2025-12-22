#!/usr/bin/env python3
"""
Strategy D (Refactored): User Baseline Logic
============================================
Replicates the logic from phase1_inference.py + phase2_worker.py
- Resize input to 704x704
- Inference
- Resize probabilities back to original size
- Threshold 0.97
- User's SVD formula
- No explicit dilation (Resize acts as dilation)
"""

import sys
from pathlib import Path

# Add src to sys.path to allow imports if needed
PROJECT_ROOT = Path(__file__).resolve().parents[3]
sys.path.append(str(PROJECT_ROOT / "src"))

import cv2
import torch
import torch.nn.functional as F
from tqdm import tqdm
import segmentation_models_pytorch as smp
import albumentations as A
import argparse
import json
import numpy as np

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

def predict_resized(model, image, orig_h, orig_w):
    transform = A.Compose([
        A.Resize(INFERENCE_SIZE, INFERENCE_SIZE),
        A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ToTensorV2(),
    ])
    
    transformed = transform(image=image)
    img_tensor = transformed['image'].unsqueeze(0).to(DEVICE)
    
    with torch.no_grad():
        logits = model(img_tensor)
        prob = torch.sigmoid(logits).squeeze().cpu().numpy()
        
    # Resize back to original
    prob_orig = cv2.resize(prob, (orig_w, orig_h), interpolation=cv2.INTER_LINEAR)
    return prob_orig

def compute_line_params_user_formula(mask):
    """Exact copy of phase2_worker.py formula"""
    if mask.sum() < 10:
        return None
    coords = np.column_stack(np.where(mask > 0)) # (N, 2) -> (y, x) ?? 
    # phase2_worker: coords = np.column_stack(np.where(mask > 0))
    # np.where returns (row_idxs, col_idxs) -> (y, x)
    # So coords is [(y0, x0), (y1, x1), ...]
    
    if len(coords) < 3:
        return None
    centroid = coords.mean(axis=0)
    try:
        _, _, Vt = np.linalg.svd(coords - centroid)
        # Vt[0] is PC1 (direction of line)
        # Vt[0] = [v_y, v_x]
        
        theta = np.arctan2(Vt[0, 1], Vt[0, 0]) 
        # theta = atan2(v_x, v_y)
        
        if theta < 0:
            theta += np.pi
            
        # rho = abs( mean_y * cos(theta) + mean_x * sin(theta) ) ??
        # or abs( mean_y * v_y + mean_x * v_x ) ?
        # centroid[0] is y, centroid[1] is x
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
    slices = ndimage.find_objects(labeled) # Optional optimization, User looped 1..N
    
    # Replicating User Loop logic (on crops for speed/memory, but logic matches)
    for i, slice_obj in enumerate(slices):
        if slice_obj is None: continue
        
        y_slice, x_slice = slice_obj
        # Note: No padding here, User didn't pad.
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
        
        # User Params
        lines = compute_line_params_user_formula(component_mask_crop) 
        # WARNING: User passed 'component' (crop?) or full mask?
        # phase2_worker: component = (labeled == i).astype(np.uint8) -> FULL SIZE
        # I am passing CROP.
        # User formula uses coordinates. If I pass crop, coordinates are local!
        # I MUST ADJUST COORDINATES TO GLOBAL!
        
        # Correct logic:
        # Pass offsets or reconstruct indices.
        # My compute_line_params_user_formula expects mask.
        # It calculates np.where(mask > 0).
        # If I pass crop, I get local y,x.
        # I need to modify the function to accept offset, or pass Global Coords.
        
        # Let's adjust the formula inline or make a wrapper.
        # User formula: coords = np.column_stack(np.where(mask > 0))
        
        # Optimized implementation:
        y_local, x_local = np.where(component_mask_crop > 0)
        y_global = y_local + offset[0]
        x_global = x_local + offset[1]
        coords = np.column_stack((y_global, x_global)) # (y, x) per User
        
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
        
        # RLE output (needs global mask? Or crop rle + specific bbox?)
        # User used full size mask encode.
        # We can simulate full size via crop decode?
        # Safe way: create blank full array (might satisfy memory if 1 channel uint8)
        # 4k x 3k = 12MB. Safe in subprocess.
        full_mask = np.zeros((height, width), dtype=np.uint8, order='F')
        full_mask[y_slice, x_slice] = component_mask_crop
        rle = mask_util.encode(full_mask)
        rle['counts'] = rle['counts'].decode('utf-8')
        del full_mask
        
        # Bbox
        # User: mask_to_bbox(component)
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
    """Single image processing."""
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
        
        # Inference (Resize Logic)
        prob_map = predict_resized(model, image, h, w)
        
        # Post-process
        instances, _ = extract_instances(prob_map, args.img_id, args.start_id, h, w)
        
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
    config_str = f"t{int(THRESHOLD*100)}_s{INFERENCE_SIZE}"
    OUTPUT_JSON = PROJECT_ROOT / "models" / "final_predictions.json" # User selected this as best
    TEMP_DIR = PROJECT_ROOT / "models" / f"temp_preds_{config_str}"

    if args.mode == "worker":
        run_worker(args)
    else:
        run_orchestrator(args)
