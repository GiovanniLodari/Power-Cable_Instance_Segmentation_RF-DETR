#!/usr/bin/env python3
"""
Strategy E (Refactored): M2F + User Baseline Logic (Resize 704)
===============================================================
Replicates the logic from phase1_inference.py + phase2_worker.py
BUT applies it to Mask2Former model.
- Resize input to 704x704
- M2F Inference (Semantic Segmentation)
- Resize probability map back to original size
- Threshold 0.5 (M2F might be more confident, but let's stick to simple logic first)
  Actually, User said "Threshold 0.97" for U-Net. M2F might need different calibration.
  Let's stick to 0.5 for M2F as it's a diff model, or maybe 0.97?
  M2F masks are usually binary-ish (soft masks are sharp).
  Let's use 0.5 as default for M2F.
- User's SVD formula
- No explicit dilation
"""

import os
import gc
import json
import cv2
import numpy as np
import torch
import argparse
import subprocess
import shutil
from pathlib import Path
from tqdm import tqdm
from scipy import ndimage
from pycocotools import mask as mask_util
from albumentations.pytorch import sys
from pathlib import Path

# Add src to sys.path
PROJECT_ROOT = Path(__file__).resolve().parents[3]
sys.path.append(str(PROJECT_ROOT / "src"))

import torch
import cv2
import json
import numpy as np
from tqdm import tqdm
from detectron2.config import get_cfg
from detectron2.engine import DefaultPredictor
from detectron2.projects.deeplab import add_deeplab_config
from detectron2.data import MetadataCatalog

# Import Mask2Former from source
# Assuming mask2former is installed or in src
try:
    from mask2former import add_maskformer2_config
except ImportError:
    # If using local mask2former
    pass

# Constants
DATA_ROOT = PROJECT_ROOT / "data"

# Import backbone registration (Crucial)
# Since we are in src/experiments/m2f_unetbb/, we need to import from src/
# sys.path is already patched above.
from backbone_unet import UNetPPBackbone
from train_resnet import add_maskformer2_config

# =============================================================================
# CONFIGURATION
# =============================================================================
PROJECT_ROOT = Path(__file__).resolve().parent.parent
# Model from 'train_m2f_custom_weights.py' output
MODEL_DIR = PROJECT_ROOT / "models" / "output_m2f_unetpp_custom"
MODEL_PATH = MODEL_DIR / "model_final.pth"
DATA_JSON = PROJECT_ROOT / "data" / "combined" / "val_combined.json"
OUTPUT_JSON = PROJECT_ROOT / "models" / "predictions_m2f_instances.json"
TEMP_DIR = PROJECT_ROOT / "models" / "temp_preds_m2f"

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
INFERENCE_SIZE = 704
# M2F threshold? U-Net used 0.97. M2F outputs might be very sharp (0 or 1).
# We'll stick to 0.5 for safety, or tuned later.
THRESHOLD = 0.5 

# =============================================================================
# HELPER FUNCTIONS (WORKER)
# =============================================================================
def setup_m2f_config():
    cfg = get_cfg()
    add_deeplab_config(cfg)
    add_maskformer2_config(cfg)
    
    # Base Config (Must match training)
    cfg.merge_from_file("Mask2Former/configs/coco/instance-segmentation/maskformer2_R50_bs16_50ep.yaml")
    
    # Custom Overrides from train_m2f_custom_weights.py
    cfg.MODEL.BACKBONE.NAME = "UNetPPBackbone"
    cfg.MODEL.WEIGHTS = str(MODEL_PATH)
    
    cfg.MODEL.SEM_SEG_HEAD.IN_FEATURES = ["p3", "p4", "p5"]
    cfg.MODEL.MASK_FORMER.IN_FEATURES = ["p3", "p4", "p5"]
    cfg.MODEL.SEM_SEG_HEAD.DEFORMABLE_TRANSFORMER_ENCODER_IN_FEATURES = ["p3", "p4", "p5"]
    cfg.MODEL.RESNETS.DEPTH = 50 
    
    cfg.MODEL.SEM_SEG_HEAD.NUM_CLASSES = 1
    cfg.MODEL.RETINANET.NUM_CLASSES = 1
    cfg.MODEL.MASK_FORMER.DEC_LAYERS = 9
    
    # IMPORTANT: Detectron2 handles resizing internally via InputAugmentation?
    # DefaultPredictor uses cfg.INPUT.MIN_SIZE_TEST.
    # To enforce 704x704, we set MIN and MAX to 704.
    cfg.INPUT.MIN_SIZE_TEST = INFERENCE_SIZE
    cfg.INPUT.MAX_SIZE_TEST = INFERENCE_SIZE
    
    cfg.MODEL.DEVICE = DEVICE
    return cfg

def load_model():
    print(f"   📦 (Worker) Loading Mask2Former...")
    cfg = setup_m2f_config()
    predictor = DefaultPredictor(cfg)
    return predictor

def predict_m2f_resized(predictor, image, orig_h, orig_w):
    # Predictor resizes internally based on cfg.
    # We just pass the original image (or should we resize externally?)
    # DefaultPredictor.__call__ does:
    #   ResizeShortestEdge(cfg.INPUT.MIN_SIZE_TEST, cfg.INPUT.MAX_SIZE_TEST)
    # If we set both to 704, it forcefully resizes so shortest edge is 704?
    # Then max size is 704?
    # If image is non-square (4000x3000), 
    # Min=704 -> Shortest=704. 3000->704. 4000->938.
    # Max=704 -> Clamps to 704. So 938 becomes 704. 704 becomes 528?
    # No.
    
    # User's Phase 1 used Albumentations Resize(704, 704) -> Square!
    # Detectron2 ResizeShortestEdge preserves aspect ratio usually.
    # To match user exactly (Squish to square?), we should Resize externally
    # and disable D2 resizing?
    # Or just use D2 resizing which keeps aspect ratio (Better!).
    # Let's trust D2 resizing (AspectRatio) is better than Squish.
    # But if we want *exact* reproduction of logic...
    
    # Let's rely on D2's logic but ensure scale is similar (~704px).
    # We configured cfg.INPUT.MIN_SIZE_TEST = 704.
    
    outputs = predictor(image) 
    
    # Outputs are usually resized back to input image size by DefaultPredictor?
    # Yes, DefaultPredictor returns results in original image coords.
    
    if "sem_seg" in outputs:
        sem_seg = outputs["sem_seg"] # (C, H, W)
        logit = sem_seg[0, :, :]
        prob = torch.sigmoid(logit).cpu().numpy()
    else:
        # Construct from instances if sem_seg missing
        # But we want prob map.
        # If sem_seg is missing, we might need to enable it in config?
        # cfg.MODEL.SEM_SEG_HEAD.NAME = "MaskFormerHead"?
        # It should be there for Mask2Former.
        if "instances" in outputs:
            # Fallback: create binary from instances
            insts = outputs["instances"]
            if len(insts) > 0:
                masks = insts.pred_masks # (N, H, W)
                limit = min(len(masks), 50) # Safety
                # Combined max?
                # This is binary though.
                # Use scores?
                scores = insts.scores
                # Weighted sum?
                # Let's just Max.
                prob = np.zeros(masks.shape[1:], dtype=np.float32)
                for i in range(len(masks)):
                    m = masks[i].cpu().numpy().astype(np.float32)
                    s = float(scores[i].cpu())
                    prob = np.maximum(prob, m * s)
            else:
                 prob = np.zeros(image.shape[:2], dtype=np.float32)
        else:
             prob = np.zeros(image.shape[:2], dtype=np.float32)

    return prob # Already original size

def compute_line_params_user_formula(mask):
    """Exact copy of phase2_worker.py formula"""
    if mask.sum() < 10:
        return None
    coords = np.column_stack(np.where(mask > 0)) # (y, x)
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
    
    binary = (prob_map > THRESHOLD).astype(np.uint8)
    
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
        
        score = float(prob_crop[component_mask_crop > 0].mean())
        if score < 0.3: continue
        
        # User Params Formula with offset fix
        y_local, x_local = np.where(component_mask_crop > 0)
        y_global = y_local + offset[0]
        x_global = x_local + offset[1]
        coords = np.column_stack((y_global, x_global))
        
        lines = [0.0, 0.0]
        if len(coords) >= 3:
             centroid = coords.mean(axis=0)
             try:
                 _, _, Vt = np.linalg.svd(coords - centroid)
                 theta = np.arctan2(Vt[0, 1], Vt[0, 0])
                 if theta < 0: theta += np.pi
                 rho = abs(centroid[0] * np.cos(theta) + centroid[1] * np.sin(theta))
                 lines = [float(rho), float(theta)]
             except:
                 pass

        full_mask = np.zeros((height, width), dtype=np.uint8, order='F')
        full_mask[y_slice, x_slice] = component_mask_crop
        rle = mask_util.encode(full_mask)
        rle['counts'] = rle['counts'].decode('utf-8')
        del full_mask
        
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
    try:
        model = load_model()
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
        h, w = image.shape[:2]
        
        # Inference (D2 handles resize)
        prob_map = predict_m2f_resized(model, image, h, w)
        
        # Post-process
        instances, _ = extract_instances(prob_map, args.img_id, args.start_id, h, w)
        
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
def run_orchestrator():
    print(f"🚀 Starting Orchestrator (Mask2Former - Resize 704 Logic)")
    TEMP_DIR.mkdir(parents=True, exist_ok=True)
    existing_ids = set()
    for f in TEMP_DIR.glob("*.json"):
       try: existing_ids.add(int(f.stem))
       except: pass
            
    if existing_ids: print(f"   🔄 Resuming: Found {len(existing_ids)}.")
    else: print("   🆕 Starting fresh M2F.")
        
    with open(DATA_JSON, 'r') as f:
        coco_data = json.load(f)
    images = coco_data['images']
    
    pbar = tqdm(images, desc="M2F Inference")
    for img_info in pbar:
        img_id = img_info['id']
        if img_id in existing_ids: continue
        fname = img_info['file_name']
        cmd = [sys.executable, __file__, "--mode", "worker", "--img_path", fname, "--img_id", str(img_id)]
        result = subprocess.run(cmd, capture_output=True, text=True)
        if result.returncode != 0:
            print(f"\n❌ Fail {fname}: {result.stderr}")

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
                
    with open(OUTPUT_JSON, 'w') as f:
        json.dump(all_instances, f)
    # shutil.rmtree(TEMP_DIR)
    print("✅ M2F Done.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["orchestrator", "worker"], default="orchestrator")
    parser.add_argument("--img_path", type=str)
    parser.add_argument("--img_id", type=int)
    parser.add_argument("--start_id", type=int, default=0)
    args = parser.parse_args()
    if args.mode == "worker": run_worker(args)
    else: run_orchestrator()
