#!/usr/bin/env python3
"""
Strategy E (Optimized): M2F + Resize 704 + Dilation
===================================================
Attempt to fix LDS 1.98 result.
1. Dilation: M2F masks are too thin compared to U-Net's upscaled masks.
   Adding 7x7 dilation to mimic the 704->3000 upscale blur.
2. Threshold: M2F predicted 2798 lines (vs 1125 U-Net). Too much noise.
   Increasing threshold 0.5 -> 0.65.
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
from pathlib import Path
from tqdm import tqdm
from scipy import ndimage
from pycocotools import mask as mask_util
from albumentations.pytorch import ToTensorV2
import albumentations as A

# Detectron2 & Mask2Former imports
from detectron2.config import get_cfg
from detectron2.engine import DefaultPredictor
from detectron2.projects.deeplab import add_deeplab_config
from detectron2.data import MetadataCatalog
from mask2former import add_maskformer2_config

# Import backbone registration (Crucial)
sys.path.append(str(Path(__file__).resolve().parent.parent))
from backbone_unet import UNetPPBackbone
from train_resnet import add_maskformer2_config

# =============================================================================
# CONFIGURATION
# =============================================================================
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
MODEL_DIR = PROJECT_ROOT / "models" / "output_m2f_unetpp_custom"
MODEL_PATH = MODEL_DIR / "model_final.pth"
DATA_JSON = PROJECT_ROOT / "data" / "combined" / "val_combined.json"
TEMP_DIR = PROJECT_ROOT / "models" / "temp_preds_m2f_optim"

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
INFERENCE_SIZE = 704
THRESHOLDS = [0.5, 0.85, 0.95, 0.97] # Testing multiple thresholds

# =============================================================================
# HELPER FUNCTIONS
# =============================================================================
def setup_m2f_config():
    cfg = get_cfg()
    add_deeplab_config(cfg)
    add_maskformer2_config(cfg)
    cfg.merge_from_file("Mask2Former/configs/coco/instance-segmentation/maskformer2_R50_bs16_50ep.yaml")
    
    cfg.MODEL.BACKBONE.NAME = "UNetPPBackbone"
    cfg.MODEL.WEIGHTS = str(MODEL_PATH)
    
    cfg.MODEL.SEM_SEG_HEAD.IN_FEATURES = ["p3", "p4", "p5"]
    cfg.MODEL.MASK_FORMER.IN_FEATURES = ["p3", "p4", "p5"]
    cfg.MODEL.SEM_SEG_HEAD.DEFORMABLE_TRANSFORMER_ENCODER_IN_FEATURES = ["p3", "p4", "p5"]
    cfg.MODEL.RESNETS.DEPTH = 50 
    
    cfg.MODEL.SEM_SEG_HEAD.NUM_CLASSES = 1
    cfg.MODEL.RETINANET.NUM_CLASSES = 1
    cfg.MODEL.MASK_FORMER.DEC_LAYERS = 9
    
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
    outputs = predictor(image) 
    
    if "sem_seg" in outputs:
        sem_seg = outputs["sem_seg"] # (C, H, W)
        logit = sem_seg[0, :, :]
        prob = torch.sigmoid(logit).cpu().numpy()
    else:
        if "instances" in outputs:
            insts = outputs["instances"]
            if len(insts) > 0:
                masks = insts.pred_masks
                scores = insts.scores
                prob = np.zeros(masks.shape[1:], dtype=np.float32)
                for i in range(len(masks)):
                    m = masks[i].cpu().numpy().astype(np.float32)
                    s = float(scores[i].cpu())
                    prob = np.maximum(prob, m * s)
            else:
                 prob = np.zeros(image.shape[:2], dtype=np.float32)
        else:
             prob = np.zeros(image.shape[:2], dtype=np.float32)
    return prob

def compute_line_params_user_formula(mask):
    if mask.sum() < 10: return None
    coords = np.column_stack(np.where(mask > 0)) # (y, x)
    if len(coords) < 3: return None
    centroid = coords.mean(axis=0)
    try:
        _, _, Vt = np.linalg.svd(coords - centroid)
        theta = np.arctan2(Vt[0, 1], Vt[0, 0]) 
        if theta < 0: theta += np.pi
        rho = abs(centroid[0] * np.cos(theta) + centroid[1] * np.sin(theta))
        return [float(rho), float(theta)]
    except:
        return None

def extract_instances(prob_map, img_id, start_id, height, width, threshold):
    instances = []
    
    # 1. OPTIMIZATION: Dilation of probability map or binary mask
    binary = (prob_map > threshold).astype(np.uint8)
    
    if binary.sum() < 50:
        return instances, start_id
    
    # DILATION applied globally first (simulating U-Net blur)
    kernel = np.ones((7, 7), np.uint8)
    binary_dilated = cv2.dilate(binary, kernel, iterations=1)
    
    # Label ON DILATED MASK!
    labeled, num_features = ndimage.label(binary_dilated)
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
        
        # FIX: Compute score on the "Core" (original detected pixels) to avoid dilution by zeros
        core_mask = component_mask_crop & (prob_crop > threshold)
        if core_mask.sum() == 0:
            score = 0.0
        else:
            score = float(prob_crop[core_mask].mean())
            
        if score < 0.3: continue
        
        # User Params Formula 
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
        
        prob_map = predict_m2f_resized(model, image, h, w)
        
        all_res = {}
        curr_id = args.start_id
        
        for t in THRESHOLDS:
             # Need unique IDs? Worker doesn't know global order.
             # Orchestrator will fix IDs.
             insts, _ = extract_instances(prob_map, args.img_id, 0, h, w, t)
             all_res[str(t)] = insts
        
        out_file = TEMP_DIR / f"{args.img_id}.json"
        with open(out_file, 'w') as f:
            json.dump(all_res, f)
        print(f"   ✅ Saved instances for {THRESHOLDS}")
        
    except Exception as e:
        print(f"   ❌ Error in worker: {e}")
        import traceback
        traceback.print_exc()

def run_orchestrator():
    print(f"🚀 Starting Orchestrator (M2F Multi-Threshold: {THRESHOLDS})")
    TEMP_DIR.mkdir(parents=True, exist_ok=True)
    existing_ids = set()
    for f in TEMP_DIR.glob("*.json"):
       try: existing_ids.add(int(f.stem))
       except: pass
            
    if existing_ids: print(f"   🔄 Resuming: Found {len(existing_ids)}.")
    else: print("   🆕 Starting fresh.")
        
    with open(DATA_JSON, 'r') as f:
        coco_data = json.load(f)
    images = coco_data['images']
    
    pbar = tqdm(images, desc="M2F Infer")
    for img_info in pbar:
        img_id = img_info['id']
        if img_id in existing_ids: continue
        fname = img_info['file_name']
        cmd = [sys.executable, __file__, "--mode", "worker", "--img_path", fname, "--img_id", str(img_id)]
        result = subprocess.run(cmd, capture_output=True, text=True)
        if result.returncode != 0:
            print(f"\n❌ Fail {fname}: {result.stderr}")

    print("🧩 Merging results...")
    
    # Initialize containers for each threshold
    final_preds = {t: [] for t in THRESHOLDS}
    final_ids = {t: 1 for t in THRESHOLDS}
    
    for img_info in images:
        temp_file = TEMP_DIR / f"{img_info['id']}.json"
        if temp_file.exists():
            with open(temp_file, 'r') as f:
                data = json.load(f)
            
            for t_str, insts in data.items():
                t = float(t_str)
                if t not in final_preds: continue
                
                for inst in insts:
                    inst['id'] = final_ids[t]
                    final_ids[t] += 1
                    final_preds[t].append(inst)
    
    for t in THRESHOLDS:
        out_path = PROJECT_ROOT / "models" / f"predictions_m2f_optim_t{t}.json"
        print(f"💾 T={t}: Saving {len(final_preds[t])} preds to {out_path.name}")
        with open(out_path, 'w') as f:
             json.dump(final_preds[t], f)
             
    print("✅ M2F Multi-Optim Done.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["orchestrator", "worker"], default="orchestrator")
    parser.add_argument("--img_path", type=str)
    parser.add_argument("--img_id", type=int)
    parser.add_argument("--start_id", type=int, default=0)
    args = parser.parse_args()
    if args.mode == "worker": run_worker(args)
    else: run_orchestrator()
