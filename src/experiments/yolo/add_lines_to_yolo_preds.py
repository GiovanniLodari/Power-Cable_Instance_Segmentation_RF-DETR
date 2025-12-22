
import json
import numpy as np
import cv2
import math
from pycocotools import mask as coco_mask
import argparse
import os
from tqdm import tqdm

def fit_line_to_mask(mask_rle, height, width):
    """
    Decodes RLE mask and fits a line to it.
    Returns (rho, theta).
    """
    # Decode RLE
    mask = coco_mask.decode(mask_rle)
    if mask.ndim == 3:
        mask = mask[:, :, 0]
    
    # Get points
    y_idxs, x_idxs = np.where(mask > 0)
    
    if len(x_idxs) < 2:
        return None, None
        
    pts = np.column_stack((x_idxs, y_idxs)).astype(np.float32)
    
    # Fit line: returns (vx, vy, x0, y0)
    # where (vx, vy) is a normalized vector collinear to the line
    # and (x0, y0) is a point on the line
    [vx, vy, x0, y0] = cv2.fitLine(pts, cv2.DIST_L2, 0, 0.01, 0.01)
    
    vx = float(vx)
    vy = float(vy)
    x0 = float(x0)
    y0 = float(y0)
    
    # Convert to polar coordinates (rho, theta)
    # Line equation: -vy * x + vx * y + (vy*x0 - vx*y0) = 0
    # Normal vector is (-vy, vx)
    
    # theta is the angle of the normal vector
    theta = math.atan2(vx, -vy)
    rho = x0 * math.cos(theta) + y0 * math.sin(theta)
    
    # Normalize theta to [0, pi) for consistency with common evaluations if needed
    # But usually standard polar range is sufficient.
    # We'll stick to standard math for now.
    
    return rho, theta

def process_predictions(input_path, output_path, gt_path=None):
    # Build mapping if GT is provided
    filename_to_id = {}
    if gt_path:
        print(f"Loading GT from {gt_path} to build ID mapping...")
        with open(gt_path, 'r') as f:
            gt = json.load(f)
        for img in gt['images']:
            # map "basename_no_ext" -> id
            # e.g. "41_00301" -> 0
            fname = os.path.splitext(img['file_name'])[0]
            filename_to_id[fname] = img['id']
            # Also map full name just in case
            filename_to_id[img['file_name']] = img['id']
            
    print(f"Loading predictions from {input_path}...")
    with open(input_path, 'r') as f:
        preds = json.load(f)
        
    print(f"Processing {len(preds)} predictions...")
    
    count = 0
    mapped_ids = 0
    for pred in tqdm(preds):
        # Fix image_id if mapping exists
        if gt_path:
            raw_id = str(pred['image_id'])
            # try direct lookup or os.path.splitext
            if raw_id in filename_to_id:
                pred['image_id'] = filename_to_id[raw_id]
                mapped_ids += 1
            else:
                # If prediction ID is already an int or string int, it might be fine, or might be wrong.
                # YOLO usually outputs filename stems.
                pass

        # Force category_id to 0 to match GT
        pred['category_id'] = 0

        seg = pred.get('segmentation', None)
        if not seg:
            continue
            
        # size is typically [h, w] in RLE dict from YOLO, but COCO RLE 'counts' needs to be decoded carefully
        # YOLO export uses pycocotools which usually puts 'size': [h, w]
        
        h, w = seg['size']
        rho, theta = fit_line_to_mask(seg, h, w)
        
        if rho is not None:
            pred['lines'] = [rho, theta]
            count += 1
            
    if gt_path:
        print(f"Remapped IDs for {mapped_ids} predictions.")
            
    print(f"Added lines to {count} masks.")
    
    print(f"Saving to {output_path}...")
    with open(output_path, 'w') as f:
        json.dump(preds, f)
        
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('input', help='Input predictions.json path')
    parser.add_argument('output', help='Output predictions_with_lines.json path')
    parser.add_argument('--gt', help='Optional path to GT json for ID mapping', default=None)
    args = parser.parse_args()
    
    process_predictions(args.input, args.output, args.gt)
