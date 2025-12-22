
import json
import cv2
import numpy as np
import os
import argparse
from pycocotools import mask as mask_util
from tqdm import tqdm

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--img", type=str, required=True, help="Path to image file")
    parser.add_argument("--gt", type=str, required=True, help="Path to GT JSON")
    parser.add_argument("--pred", type=str, required=True, help="Path to Predictions JSON")
    parser.add_argument("--out", type=str, required=True, help="Path to output visualization")
    args = parser.parse_args()
    
    # Load Data
    with open(args.gt, 'r') as f:
        gt_data = json.load(f)
    
    with open(args.pred, 'r') as f:
        pred_data = json.load(f)
        
    # Find Image ID
    filename = os.path.basename(args.img)
    img_info = None
    for img in gt_data['images']:
        if img['file_name'].endswith(filename):
            img_info = img
            break
            
    if not img_info:
        print(f"Image {filename} not found in GT")
        return
        
    img_id = img_info['id']
    print(f"Image ID: {img_id}")
    
    # Load Image
    image = cv2.imread(args.img)
    if image is None:
        print("Could not read image")
        return
    
    h_img, w_img = image.shape[:2]
    
    # Create Mask Overlay
    # Green = GT, Red = Pred
    
    gt_canvas = np.zeros((h_img, w_img), dtype=np.uint8)
    pred_canvas = np.zeros((h_img, w_img), dtype=np.uint8)
    
    # Get GT Annotations
    gt_anns = [a for a in gt_data['annotations'] if a['image_id'] == img_id]
    print(f"Found {len(gt_anns)} GT annotations")
    
    for ann in gt_anns:
        seg = ann['segmentation']
        if isinstance(seg, list):
            # Polygon
            for poly in seg:
                pts = np.array(poly).reshape(-1, 2).astype(np.int32)
                cv2.fillPoly(gt_canvas, [pts], 1)
        elif isinstance(seg, dict):
            # RLE
            m = mask_util.decode(seg)
            gt_canvas = np.maximum(gt_canvas, m)
            
    # Get Predictions
    preds = [p for p in pred_data if p['image_id'] == img_id]
    print(f"Found {len(preds)} Predictions")
    
    for p in preds:
        if p['score'] < 0.1: continue # Only show substantial preds
        
        seg = p['segmentation']
        if isinstance(seg, dict):
            m = mask_util.decode(seg)
            # Ensure size matches
            if m.shape[:2] != (h_img, w_img):
                 m = cv2.resize(m, (w_img, h_img), interpolation=cv2.INTER_NEAREST)
            pred_canvas = np.maximum(pred_canvas, m)
            
    # Combine
    # GT = Green, Pred = Blue
    vis = image.copy()
    
    # Green channel for GT
    vis[:, :, 1] = np.where(gt_canvas > 0, 255, vis[:, :, 1])
    
    # Red channel for Pred
    vis[:, :, 2] = np.where(pred_canvas > 0, 255, vis[:, :, 2]) # BGR -> R is index 2? No B G R. index 2 is Red.
    
    # Save
    cv2.imwrite(args.out, vis)
    print(f"Saved to {args.out}")

if __name__ == "__main__":
    main()
