
#!/usr/bin/env python3
"""
Mask2Former Emulator Pipeline
=============================
Emulates the user's U-Net inference loop behavior for Mask2Former:
1. Loads M2F model.
2. Runs inference on validation set (resizing output to original resolution via DefaultPredictor).
3. Applies Thresholding.
4. Calculates SVD-based Line Parameters (rho, theta).
5. Outputs COCO JSON.

Options:
--upscale: (Bool) If True, performs extra bilinear upscaling of logits (Not enabled by default).
--threshold: Score threshold (default 0.05).
"""

import os
import cv2
import json
import torch
import torch.nn.functional as F
import argparse
import numpy as np
import random
from tqdm import tqdm
from pathlib import Path

# Detectron2 imports
from detectron2.config import get_cfg
from detectron2.engine import DefaultPredictor
from detectron2.data import MetadataCatalog, DatasetCatalog
from detectron2.utils.visualizer import Visualizer
from detectron2.projects.deeplab import add_deeplab_config
from pycocotools import mask as mask_util
from detectron2.data.datasets import register_coco_instances

import sys
# Hack to import project modules from src/ if running from root
if os.getcwd() not in sys.path:
    sys.path.append(os.getcwd())
src_path = os.path.join(os.getcwd(), 'src')
if src_path not in sys.path:
    sys.path.append(src_path)

from train_m2f_robust import add_maskformer2_config
from backbone_unet import UNetPPBackbone

# =============================================================================
# OPTIMIZED SVD LOGIC (Matching User's Formula)
# =============================================================================
def compute_line_params_svd(mask):
    """
    Computes (rho, theta) using SVD on the mask pixel coordinates.
    Matches u-net logic.
    """
    if mask.sum() < 10:
        return None
    
    # Get coordinates of all foreground pixels (y, x)
    coords = np.column_stack(np.where(mask > 0))
    
    if len(coords) < 3:
        return None
        
    centroid = coords.mean(axis=0)
    try:
        # Singular Value Decomposition
        # coords - centroid centers the data
        _, S, Vt = np.linalg.svd(coords - centroid)
        
        # Linearity Metric
        linearity = S[0] / (S[1] + 1e-6)
        
        # Vt[0] corresponds to the principal axis (direction of the line)
        # Vector is (vy, vx) corresponding to (y, x) axis
        v_y, v_x = Vt[0, 0], Vt[0, 1]
        
        # Calculate theta (angle with x-axis? or y-axis?)
        # Original: theta = np.arctan2(Vt[0, 1], Vt[0, 0]) -> arctan2(vx, vy)
        theta = np.arctan2(v_x, v_y) 
        
        if theta < 0:
            theta += np.pi
            
        # Rho 
        rho = abs(centroid[0] * np.cos(theta) + centroid[1] * np.sin(theta))
        
        return [float(rho), float(theta), float(linearity)]
    except:
        return None

def polar_to_endpoints(rho, theta, width, height):
    """
    Computes endpoints (x1, y1, x2, y2) of a line definition (rho, theta)
    clipped to image dimensions (width, height).
    """
    ct = np.cos(theta)
    st = np.sin(theta)
    pts = []
    # Intersection with left (x=0)
    if abs(st) > 1e-3:
        y = rho / st
        if 0 <= y <= height: pts.append((0, y))
    # Intersection with right (x=width)
    if abs(st) > 1e-3:
        y = (rho - width * ct) / st
        if 0 <= y <= height: pts.append((width, y))
    # Intersection with top (y=0)
    if abs(ct) > 1e-3:
        x = rho / ct
        if 0 <= x <= width: pts.append((x, 0))
    # Intersection with bottom (y=height)
    if abs(ct) > 1e-3:
        x = (rho - height * st) / ct
        if 0 <= x <= width: pts.append((x, height))
            
    unique_pts = sorted(list(set(pts)))
    if len(unique_pts) >= 2:
        return unique_pts[0][0], unique_pts[0][1], unique_pts[-1][0], unique_pts[-1][1]
    return 0, 0, 0, 0

def mask_to_bbox(mask):
    rows = np.any(mask, axis=1)
    cols = np.any(mask, axis=0)
    if not rows.any():
        return [0, 0, 0, 0]
    y_min, y_max = np.where(rows)[0][[0, -1]]
    x_min, x_max = np.where(cols)[0][[0, -1]]
    return [float(x_min), float(y_min), float(x_max - x_min + 1), float(y_max - y_min + 1)]

def setup_cfg(args):
    cfg = get_cfg()
    add_deeplab_config(cfg)
    add_maskformer2_config(cfg)
    
    # Load config template (Strategy H uses this)
    cfg.merge_from_file("Mask2Former/configs/coco/instance-segmentation/maskformer2_R50_bs16_50ep.yaml")
    
    cfg.MODEL.WEIGHTS = args.weights
    cfg.MODEL.MASK_FORMER.TEST.SEMANTIC_ON = False
    cfg.MODEL.MASK_FORMER.TEST.INSTANCE_ON = True
    cfg.MODEL.MASK_FORMER.TEST.PANOPTIC_ON = False
    
    # CRITICAL FIX: Match training class count
    cfg.MODEL.SEM_SEG_HEAD.NUM_CLASSES = 1
    
    # CRITICAL FIX: Match Custom Backbone
    cfg.MODEL.BACKBONE.NAME = "UNetPPBackbone"
    cfg.MODEL.BACKBONE.FREEZE_AT = 0
    
    # Feature Keys (UNetPP specific)
    cfg.MODEL.SEM_SEG_HEAD.IN_FEATURES = ["p3", "p4", "p5"]
    cfg.MODEL.MASK_FORMER.IN_FEATURES = ["p3", "p4", "p5"]
    cfg.MODEL.MAS_FORMER_IN_FEATURES = ["p3", "p4", "p5"]
    cfg.MODEL.SEM_SEG_HEAD.DEFORMABLE_TRANSFORMER_ENCODER_IN_FEATURES = ["p3", "p4", "p5"]
    
    # CRITICAL FIX: Match Decoder Depth (9 layers)
    cfg.MODEL.MASK_FORMER.DEC_LAYERS = 9
    
    # CRITICAL FIX: Training data was BGR
    cfg.INPUT.FORMAT = "BGR"
    
    # CRITICAL FIX: Match Training Divisibility
    cfg.MODEL.MASK_FORMER.SIZE_DIVISIBILITY = 96
    
    # We set internal threshold low to get candidates, then filter manually
    cfg.MODEL.MASK_FORMER.TEST.OBJECT_MASK_THRESHOLD = 0.0 
    
    if args.upscale:
        print(" [Config] Upscaling enabled: MIN_SIZE_TEST = 1248")
        cfg.INPUT.MIN_SIZE_TEST = 1248
        cfg.INPUT.MAX_SIZE_TEST = 4000
    
    if args.cpu:
        cfg.MODEL.DEVICE = "cpu"
        
    return cfg

# =============================================================================
# HELPER: SKELETONIZATION
# =============================================================================
def skeletonize_opencv(img):
    """
    Iterative skeletonization using OpenCV.
    """
    skel = np.zeros(img.shape, np.uint8)
    element = cv2.getStructuringElement(cv2.MORPH_CROSS, (3,3))
    temp_img = img.copy()
    
    while True:
        eroded = cv2.erode(temp_img, element)
        temp = cv2.dilate(eroded, element)
        temp = cv2.subtract(temp_img, temp)
        skel = cv2.bitwise_or(skel, temp)
        temp_img = eroded.copy()
        if cv2.countNonZero(temp_img) == 0:
            break
    return skel

def cartesian_to_polar(x1, y1, x2, y2):
    # Convert line segment to rho, theta
    # Line equation: x*cos(theta) + y*sin(theta) = rho
    # Normal vector
    dx = x2 - x1
    dy = y2 - y1
    
    # Check for vertical line
    if dx == 0:
        theta = 0 # Vertical line normal is Horizontal (0 degrees)
        # Wait, theta in formulation is angle of Normal vector from X axis?
        # If line is vertical (x=const), Normal is horizontal (y=0). theta=0.
        # If line is horizontal (y=const), Normal is vertical (x=0). theta=pi/2.
        
        # User Logic: theta is angle of line direction?
        # No, compute_line_params_svd uses Vt[0] which is line direction.
        # User formula: theta = arctan2(v_x, v_y)
        # If line is Vertical (dy large, dx=0). v=(0, 1). theta = arctan2(0, 1) = 0.
        # If line is Horizontal (dx large, dy=0). v=(1, 0). theta = arctan2(1, 0) = pi/2.
        pass

    # However, let's use the explicit vector logic to MATCH SVD logic
    # Vector V = (dx, dy) normalized
    length = np.sqrt(dx*dx + dy*dy)
    if length == 0: return 0, 0
    vx = dx / length
    vy = dy / length
    
    # In SVD logic: v_y, v_x = Vt[0, 0], Vt[0, 1]. Vt[0] is (vy, vx)??
    # SVD on (y, x) coords.
    # col 0 is y, col 1 is x.
    # Vt[0] corresponds to PC1. It has 2 components: (component_y, component_x).
    # So v_y = component_y, v_x = component_x.
    
    # Here (x1, y1) are (col, row).
    # dx = x2 - x1 (change in col)
    # dy = y2 - y1 (change in row)
    
    # So v_x is proportional to dx? Yes.
    # v_y is proportional to dy? Yes.
    
    # theta = np.arctan2(v_x, v_y)
    theta = np.arctan2(vx, vy) 
    
    if theta < 0:
        theta += np.pi
        
    # Rho calculation (projection of centroid)
    cx = (x1 + x2) / 2
    cy = (y1 + y2) / 2
    
    # rho = abs(centroid[0] * np.cos(theta) + centroid[1] * np.sin(theta))
    # centroid[0] is y, centroid[1] is x. (Recall: coords = (y, x))
    rho = abs(cy * np.cos(theta) + cx * np.sin(theta))
    
    return float(rho), float(theta)


def main(args):
    dataset_name = "ttpla_combined_val_emulator"
    register_coco_instances(dataset_name, {}, "data/combined/val_combined.json", "data/all_images")
    
    cfg = setup_cfg(args)
    predictor = DefaultPredictor(cfg)
    
    # Load Dataset
    dataset_dicts = DatasetCatalog.get(dataset_name)
    
    # Output structure
    coco_predictions = []
    
    pred_global_id = 1
    
    print(f"Running M2F Emulator (Green Strategy) on {len(dataset_dicts)} images...")
    print(f"   Weights: {args.weights}")
    print(f"   Threshold: {args.threshold} (Union)")
    print(f"   Upscale: {args.upscale}")
    
    for img_info in tqdm(dataset_dicts):
        file_name = img_info["file_name"]
        image_id = img_info["image_id"]
        
        # Read Image
        img = cv2.imread(file_name)
        if img is None: continue
        
        # 1. Inference
        outputs = predictor(img)
        instances = outputs["instances"].to("cpu")
        
        # 2. Probability Map Accumulation
        scores = instances.scores.numpy()
        pred_masks = instances.pred_masks.numpy() # (N, H, W) bool
        
        prob_map = np.zeros(img.shape[:2], dtype=np.float32)
        all_idxs = np.where(scores > 0.01)[0]
        for idx in all_idxs:
            mask = pred_masks[idx].astype(np.uint8)
            score = float(scores[idx])
            prob_map += mask.astype(np.float32) * score
            
        # 3. Union Mask (Thresholding)
        union_mask = (prob_map > args.threshold).astype(np.uint8)
        
        # 4. Morphological Closing (Connect gaps)
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        union_mask = cv2.morphologyEx(union_mask, cv2.MORPH_CLOSE, kernel, iterations=1)
        
        # 5. Extract Components
        num_labels, labels = cv2.connectedComponents(union_mask)
        
        # 6. SVD per Component (Best Strategy: Full Mask for AP, SVD for Angle)
        for label_id in range(1, num_labels + 1):
            component_mask = (labels == label_id).astype(np.uint8)
            comp_area = float(component_mask.sum())
            if comp_area < 50: continue

            # SVD Extraction for Line Parameters
            lines_data = compute_line_params_svd(component_mask)
            if not lines_data: continue
            
            rho, theta, linearity = lines_data
            
            # Use Full Component Mask for Coverage (IoU)
            # This matches the true shape of the cable better than a straight line
            rle = mask_util.encode(np.asfortranarray(component_mask))
            rle['counts'] = rle['counts'].decode('utf-8')
            
            # BBox from component mask
            bbox = mask_to_bbox(component_mask)
            
            pred = {
                "id": pred_global_id,
                "image_id": image_id,
                "category_id": 0, # cable
                "bbox": bbox,
                "segmentation": rle, 
                "area": comp_area,
                "score": 0.99, # High confidence
                "lines": [float(rho), float(theta)]
            }
            
            coco_predictions.append(pred)
            pred_global_id += 1
            
    # Save
    out_dir = os.path.dirname(args.output)
    if out_dir: os.makedirs(out_dir, exist_ok=True)
    
    print(f"Saving {len(coco_predictions)} predictions to {args.output}...")
    with open(args.output, 'w') as f:
        json.dump(coco_predictions, f)
    print("Done.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--weights", default="models/output_m2f_robust_strat_h/model_final.pth")
    parser.add_argument("--output", default="models/predictions_m2f_emulator.json")
    parser.add_argument("--threshold", type=float, default=0.05)
    parser.add_argument("--min-area", type=int, default=50)
    parser.add_argument("--upscale", action="store_true", help="Enable high-res logic (TODO)")
    parser.add_argument("--cpu", action="store_true")
    # mask-thickness removed (not used for component mask)
    args = parser.parse_args()
    main(args)
