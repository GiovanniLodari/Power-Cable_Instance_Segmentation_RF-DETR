
import cv2
import json
import os
import numpy as np
import pycocotools.mask as mask_util
from pathlib import Path

def viz_predictions(pred_json, gt_json, img_root, output_path, limit=5):
    with open(pred_json, 'r') as f:
        preds = json.load(f)
    
    with open(gt_json, 'r') as f:
        gt = json.load(f)
        
    img_map = {img['id']: img for img in gt['images']}
    
    # Group preds by image
    preds_by_img = {}
    for p in preds:
        preds_by_img.setdefault(p['image_id'], []).append(p)
        
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    count = 0
    for img_id, img_preds in preds_by_img.items():
        if count >= limit: break
        
        img_info = img_map[img_id]
        file_name = img_info['file_name']
        img_path = os.path.join(img_root, file_name)
        
        image = cv2.imread(img_path)
        if image is None: continue
        
        # Draw Preds in Red
        for p in img_preds:
            if p['score'] < 0.3: continue
            
            bbox = [int(x) for x in p['bbox']] # x, y, w, h
            cv2.rectangle(image, (bbox[0], bbox[1]), (bbox[0]+bbox[2], bbox[1]+bbox[3]), (0, 0, 255), 2)
            
            # Mask
            rle = p['segmentation']
            mask = mask_util.decode(rle)
            
            # Draw mask overlay
            colored_mask = np.zeros_like(image)
            colored_mask[mask == 1] = [0, 0, 255] # Red
            image = cv2.addWeighted(image, 1, colored_mask, 0.5, 0)
            
        cv2.imwrite(f"{output_path}_{count}.jpg", image)
        print(f"Saved {output_path}_{count}.jpg")
        count += 1

if __name__ == "__main__":
    viz_predictions(
        "experiments/point_rend/predictions_test_499.json",
        "data/test/test.json",
        "data/test",
        "experiments/point_rend/debug_viz/viz"
    )
