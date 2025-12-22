
#!/usr/bin/env python3
"""
Visualize Mask2Former predictions with SVD lines.
"""

import os
import cv2
import json
import torch
import random
import argparse
import numpy as np
from pathlib import Path
from tqdm import tqdm

from detectron2.config import get_cfg
from detectron2.engine import DefaultPredictor
from detectron2.data import MetadataCatalog, DatasetCatalog
from detectron2.utils.visualizer import Visualizer, ColorMode
from detectron2.projects.deeplab import add_deeplab_config
from detectron2 import model_zoo

# Import project modules
# Since we run this script from the root as 'python src/visualize_m2f_predictions.py',
# sys.path[0] will be '.../src'. So we can import directly.
import sys
if os.getcwd() not in sys.path:
    sys.path.append(os.getcwd())
# Ensure src is in path for consistency if running as module
src_path = os.path.join(os.getcwd(), 'src')
if src_path not in sys.path:
    sys.path.append(src_path)

from train_m2f_robust import add_maskformer2_config, M2FRobustTrainer
from train_resnet import load_ttpla_dataset
# =============================================================================
# OPTIMIZED SVD LOGIC (Matching User's Formula)
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

# Import custom backbone
from backbone_unet import UNetPPBackbone

def compute_line_params_svd(mask):
    """
    Computes User's (rho, theta) AND drawing params (centroid, vector).
    """
    if mask.sum() < 10:
        return None, None
    
    # Get coordinates of all foreground pixels (y, x)
    coords = np.column_stack(np.where(mask > 0))
    
    if len(coords) < 3:
        return None, None
        
    centroid = coords.mean(axis=0) # [y, x]
    try:
        # Singular Value Decomposition
        # coords - centroid centers the data
        # U, S, Vt = svd(A)
        _, S, Vt = np.linalg.svd(coords - centroid)
        
        # Linearity Metric
        # S[0] is spread along principal axis (Length-ish)
        # S[1] is spread along secondary axis (Width-ish)
        # Add small epsilon to avoid div zero
        linearity = S[0] / (S[1] + 1e-6)
        
        # Vt[0] corresponds to the principal axis (direction of the line)
        # Vector is (vy, vx) corresponding to (y, x) axis
        v_y, v_x = Vt[0, 0], Vt[0, 1]
        
        # --- USER PARAMETERS (For Output JSON) ---
        # Theta: Angle from Y-axis towards X-axis?
        theta = np.arctan2(v_x, v_y) 
        if theta < 0: theta += np.pi
            
        # Rho: Projection of centroid along the line direction?
        # (Note: This looks like a projection, potentially not standard Hough Rho)
        rho = abs(centroid[0] * np.cos(theta) + centroid[1] * np.sin(theta))
        
        user_params = [float(rho), float(theta)]
        
        # --- DRAWING PARAMETERS (For Visualization) ---
        # We return centroid (y,x) and vector (y,x)
        # We also pass linearity for debug
        draw_params = (centroid, (v_y, v_x), linearity)
        
        return user_params, draw_params
    except:
        return None, None

def draw_line_from_vector(image, centroid, vector, color=(0, 0, 255), thickness=2):
    """Draws a line passing through centroid with direction vector."""
    y0, x0 = centroid
    vy, vx = vector
    
    # Extrapolate
    scale = 2000
    x1 = int(x0 + scale * vx)
    y1 = int(y0 + scale * vy)
    x2 = int(x0 - scale * vx)
    y2 = int(y0 - scale * vy)
    
    cv2.line(image, (x1, y1), (x2, y2), color, thickness)
    return image

def setup_cfg(args):
    cfg = get_cfg()
    add_deeplab_config(cfg)
    add_maskformer2_config(cfg)
    
    # Load config from file or standard M2F config
    # We'll use the one from train_m2f_robust setup
    cfg.merge_from_file("Mask2Former/configs/coco/instance-segmentation/maskformer2_R50_bs16_50ep.yaml")
    
    # Override with our training settings
    cfg.MODEL.WEIGHTS = args.weights
    cfg.MODEL.MASK_FORMER.TEST.SEMANTIC_ON = False
    cfg.MODEL.MASK_FORMER.TEST.PANOPTIC_ON = False
    
    # CRITICAL FIX: Match training configuration
    cfg.MODEL.SEM_SEG_HEAD.NUM_CLASSES = 1
    
    # CRITICAL FIX: Match Custom Backbone
    cfg.MODEL.BACKBONE.NAME = "UNetPPBackbone"
    cfg.MODEL.BACKBONE.FREEZE_AT = 0
    
    # Feature Keys (UNetPP specific - must match training)
    cfg.MODEL.SEM_SEG_HEAD.IN_FEATURES = ["p3", "p4", "p5"]
    cfg.MODEL.MASK_FORMER.IN_FEATURES = ["p3", "p4", "p5"]
    cfg.MODEL.SEM_SEG_HEAD.DEFORMABLE_TRANSFORMER_ENCODER_IN_FEATURES = ["p3", "p4", "p5"]
    cfg.MODEL.PIXEL_DECODER_NAME = "MSDeformAttnPixelDecoder" # Ensure explicit
    
    # CRITICAL FIX: Match Decoder Depth (9 layers in training vs 10 default)
    cfg.MODEL.MASK_FORMER.DEC_LAYERS = 9
    
    # CRITICAL FIX: Training data was BGR (due to mapper flip-flop).
    # Config default is "RGB", so DefaultPredictor would flip it.
    # We must set "BGR" so DefaultPredictor keeps it BGR, matching training input.
    cfg.INPUT.FORMAT = "BGR"
    
    # CRITICAL FIX: Match Training Divisibility (96)
    cfg.MODEL.MASK_FORMER.SIZE_DIVISIBILITY = 96
    
    # Thresholds
    cfg.MODEL.MASK_FORMER.TEST.OBJECT_MASK_THRESHOLD = 0.0 
    
    if args.upscale:
        print(" [Config] Upscaling enabled: MIN_SIZE_TEST = 1248")
        cfg.INPUT.MIN_SIZE_TEST = 1248
        cfg.INPUT.MAX_SIZE_TEST = 4000
    
    cfg.DATALOADER.NUM_WORKERS = 0
    
    if args.cpu:
        cfg.MODEL.DEVICE = "cpu"
    
    return cfg

def main(args):
    # Register Dataset similar to train_m2f_robust.py
    from detectron2.data.datasets import register_coco_instances
    register_coco_instances("ttpla_combined_val_viz", {}, "data/combined/val_combined.json", "data/all_images")
    
    # Setup Config
    cfg = setup_cfg(args)
    # Ensure test dataset matches
    cfg.DATASETS.TEST = ("ttpla_combined_val_viz",)
    
    predictor = DefaultPredictor(cfg)
    
    # Output Dir
    os.makedirs(args.output, exist_ok=True)
    
    # Load Dataset
    dataset_dicts = DatasetCatalog.get("ttpla_combined_val_viz")
    
    # Pick random samples, but Include 76_4330.jpg if present
    targets = [d for d in dataset_dicts if "76_4330.jpg" in d["file_name"]]
    others = [d for d in dataset_dicts if "76_4330.jpg" not in d["file_name"]]
    
    random.seed(42)
    selection = random.sample(others, min(len(others), args.num_samples - len(targets)))
    samples = targets + selection
    
    print(f"Visualizing {len(samples)} samples to {args.output}...")
    
    for d in tqdm(samples):
        img_path = d["file_name"]
        img = cv2.imread(img_path)
        if img is None: continue
        
        # Inference
        outputs = predictor(img)
        
        base_name = os.path.basename(img_path)
        
        # Get instances
        instances = outputs["instances"].to("cpu")
        H, W = instances.image_size
        
        # DEBUG: Print score stats
        if len(instances) > 0:
            raw_scores = instances.scores.numpy()
            print(f"[{base_name}] {len(instances)} raw queries. Max Score: {raw_scores.max():.4f}, Mean: {raw_scores.mean():.4f}")
            if raw_scores.max() < args.threshold:
                print(f" -> ALL below threshold {args.threshold}")
        else:
            print(f"[{base_name}] NO instances returned by model.")
            
        # Filter by threshold
        scores = instances.scores.numpy()
        mask_inds = np.where(scores > args.threshold)[0]
        
        if len(mask_inds) == 0:
            continue
            
        # Draw masks
        # We use Detectron2 visualizer for masks/boxes
        # Create a filtered instances object
        filtered_instances = instances[mask_inds]
        v = Visualizer(img[:, :, ::-1], MetadataCatalog.get("ttpla_combined_val_viz"), scale=1.0) # RGB for Viz
        out = v.draw_instance_predictions(filtered_instances)
        # Fix: Make array contiguous for OpenCV
        viz_img = np.ascontiguousarray(out.get_image()[:, :, ::-1]) # Back to BGR
        
        # Draw SVD lines manually & Print Scores
        masks = filtered_instances.pred_masks.numpy()
        kept_scores = filtered_instances.scores.numpy()
        
        print(f"  > Processing {len(masks)} candidates (Thresh {args.threshold})")
        for i in range(len(masks)):
            mask = masks[i]
            score = kept_scores[i]
            area = mask.sum()
            
            # 1. Score Filter
            if score < 0.4: 
                continue
            
            # 2. Area Filter
            # U-Net used 50. We make it configurable.
            if area < args.min_area:
                continue

            # 3. Component Splitting (User Logic Alignment)
            # M2F might predict multiple disjoint segments as one mask.
            # We must split them to get accurate lines.
            mask_u8 = mask.astype(np.uint8)
            num_labels, labels = cv2.connectedComponents(mask_u8)
            
            # Iterate over components
            for label_id in range(1, num_labels + 1):
                component_mask = (labels == label_id).astype(np.uint8)
                comp_area = component_mask.sum()
                
                # Filter small components
                if comp_area < args.min_area: 
                    continue
                
                # Thickness Filter on Component
                contours, _ = cv2.findContours(component_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                if len(contours) == 0: continue
                perimeter = cv2.arcLength(contours[0], True)
                if perimeter == 0: continue
                
                thickness = 2 * comp_area / perimeter
                if thickness > args.max_thickness:
                    continue
                
                # Compute Line on Component
                user_params, draw_params = compute_line_params_svd(component_mask)
                
                if draw_params is not None:
                    centroid, vector, linearity = draw_params
                    
                    # Draw Red Line (Re-enabled for verification)
                    draw_line_from_vector(viz_img, centroid, vector, color=(0, 0, 255), thickness=2) 
                    
                    # Draw Centroid
                    cy, cx = int(centroid[0]), int(centroid[1])
                    cv2.circle(viz_img, (cx, cy), 3, (0, 255, 255), -1)
                    
                    # Draw Info
                    text = f"S{score:.2f} T{thickness:.1f}"
                    cv2.putText(viz_img, text, (cx + 10, cy), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
                    print(f"    - Mask {i} Comp {label_id}: Score {score:.2f} Area {comp_area} Thickness {thickness:.2f}")
        
        # --- Combined Probability Map (Mimic U-Net Output) ---
        # Accumulate ALL detections > 0.01 to show what the model "sees" fundamentally
        # regardless of our strict filtering threshold.
        scores = instances.scores.numpy()
        pred_masks = instances.pred_masks.numpy() # (N, H, W) bool
        
        prob_map = np.zeros(img.shape[:2], dtype=np.float32)
        
        # We still iterate to populate prob_map
        all_idxs = np.where(scores > 0.01)[0]
        
        for idx in all_idxs:
            mask = pred_masks[idx].astype(np.uint8)
            score = float(scores[idx])
            prob_map += mask.astype(np.float32) * score
            
        # Normalize Prob Map for display (0-255)
        # Clip at 1.0 for visualization so extremely confident overlaps don't wrap/washout? 
        # Actually standard colormap expects 0-255.
        prob_map_disp = (prob_map * 255).clip(0, 255).astype(np.uint8)
        prob_map_color = cv2.applyColorMap(prob_map_disp, cv2.COLORMAP_INFERNO)
        
        # Resize to original image size for concatenation (if upscaled)
        # image_size is (H, W) which is Output size.
        # img is Input size. 
        # DefaultPredictor handles resizing internally, but 'img' returned by cv2.imread is original.
        # 'viz_img' is from Visualizer, which might have scaled? No, Visualizer uses img shape.
        # Wait, if we use --upscale, M2F output is 1248-ish.
        # DefaultPredictor resizes OUTPUT mask back to INPUT image size.
        # So 'masks' are already at 'img' resolution. Perfect.
        
        # Combine Side-by-Side
        combined = np.hstack((img, prob_map_color, viz_img))
        
        # --- NEW: Union Strategy Visualization (Green Lines = Hough on Skeleton) ---
        # mimic what would happen if we treated M2F as a semantic segmenter
        # Apply strict threshold here
        union_mask = (prob_map > args.threshold).astype(np.uint8)
        
        # Morphological Closing
        # Reduced iterations to 1 to avoid merging close cables
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        union_mask = cv2.morphologyEx(union_mask, cv2.MORPH_CLOSE, kernel, iterations=1)
        
        # Skeletonize
        skel = skeletonize_opencv(union_mask)
        
        # HoughLinesP on Skeleton
        # MinLineLength: 100 (Reject road markings/noise)
        # MaxLineGap: 50 (Bridge larger gaps)
        lines_p = cv2.HoughLinesP(skel, 1, np.pi / 180, threshold=30, minLineLength=100, maxLineGap=50)
        
        # Draw on a copy of image
        union_viz = img.copy()
        
        count = 0
        if lines_p is not None:
            count = len(lines_p)
            for line in lines_p:
                x1, y1, x2, y2 = line[0]
                cv2.line(union_viz, (x1, y1), (x2, y2), (0, 255, 0), 2)
        
        print(f"  > Union Strategy (Hough): Found {count} segments")
        
        # Save Union Viz
        cv2.imwrite(os.path.join(args.output, f"union_{base_name}"), union_viz)
        
        # --- NEW: Ultimate Comparison 2x2 Grid ---
        # Top-Left: Original | Top-Right: Prob Map
        # Bot-Left: Red Lines | Bot-Right: Green Lines
        
        # Ensure all same size (they are)
        top_row = np.hstack((img, prob_map_color))
        bot_row = np.hstack((viz_img, union_viz))
        comparison = np.vstack((top_row, bot_row))
        
        cv2.imwrite(os.path.join(args.output, f"comparison_{base_name}"), comparison)
        
        # Save isolated prob
        cv2.imwrite(os.path.join(args.output, f"prob_{base_name}"), prob_map_color)
        
    print("Done.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--weights", type=str, default="models/output_m2f_robust_strat_h/model_final.pth", help="Path to weights")
    parser.add_argument("--output", default="models/visualizations")
    parser.add_argument("--num-samples", type=int, default=10)
    parser.add_argument("--threshold", type=float, default=0.35, help="Confidence threshold") # UPDATED to 0.35
    parser.add_argument("--min-area", type=int, default=50, help="Minimum pixel area for mask")
    parser.add_argument("--max-thickness", type=float, default=8.0, help="Maximum thickness (2*Area/Perim)")
    parser.add_argument("--upscale", action="store_true", help="Upscale input to 1248")
    parser.add_argument("--cpu", action="store_true")
    args = parser.parse_args()
    main(args)
