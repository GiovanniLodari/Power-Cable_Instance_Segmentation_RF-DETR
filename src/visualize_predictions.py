
import json
import cv2
import numpy as np
import os
import pycocotools.mask as mask_util
import matplotlib.pyplot as plt
import random

def main():
    PRED_JSON = "output/predictions.json"
    DATA_ROOT = "data/test"
    OUTPUT_IMG = "models/vis_prediction.png"
    
    print(f"Loading predictions from {PRED_JSON}...")
    with open(PRED_JSON, 'r') as f:
        preds = json.load(f)
        
    # Group by image_id
    preds_by_img = {}
    for p in preds:
        preds_by_img.setdefault(p['image_id'], []).append(p)
        
    # Pick an image with enough predictions (to be interesting)
    target_img_id = None
    for img_id, p_list in preds_by_img.items():
        if len(p_list) > 3: # at least 3 cables
            target_img_id = img_id
            break
            
    if target_img_id is None:
        target_img_id = list(preds_by_img.keys())[0]
        
    print(f"Visualizing Image ID: {target_img_id}")
    
    # We need the filename. We can find it by scanning the dir or if we had the GT json.
    # Since we don't want to load specific GT json again, let's just find the file corresponding to ID?
    # Wait, the predictions don't have the filename. 
    # I'll rely on loading the test.json to map ID -> Filename.
    
    TEST_JSON = "data/test/test.json"
    with open(TEST_JSON, 'r') as f:
        gt = json.load(f)
        
    img_info = next((i for i in gt['images'] if i['id'] == target_img_id), None)
    if not img_info:
        print("Could not find image info.")
        return
        
    filename = img_info['file_name']
    img_path = os.path.join(DATA_ROOT, filename)
    if not os.path.exists(img_path):
        # Fallback logic from inference script
        basename = os.path.basename(filename)
        candidate = os.path.join(DATA_ROOT, basename)
        if os.path.exists(candidate):
            img_path = candidate
            
    print(f"Reading image: {img_path}")
    img = cv2.imread(img_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    
    # Draw
    vis_img = img.copy()
    overlay = img.copy()
    
    # Colors
    colors = [
        (255, 0, 0), (0, 255, 0), (0, 0, 255),
        (255, 255, 0), (0, 255, 255), (255, 0, 255)
    ]
    
    for i, p in enumerate(preds_by_img[target_img_id]):
        color = colors[i % len(colors)]
        
        # Mask
        # RLE to Mask
        rle = p['segmentation']
        mask = mask_util.decode(rle)
        
        # Apply color to mask
        overlay[mask == 1] = color
        
        # BBox
        x, y, w, h = p['bbox']
        pt1 = (int(x), int(y))
        pt2 = (int(x+w), int(y+h))
        cv2.rectangle(vis_img, pt1, pt2, color, 2)
        
    # Blend overlay
    alpha = 0.5
    cv2.addWeighted(overlay, alpha, vis_img, 1 - alpha, 0, vis_img)
    
    # Save
    plt.figure(figsize=(10, 10))
    plt.imshow(vis_img)
    plt.axis('off')
    plt.title(f"Image {target_img_id} | {len(preds_by_img[target_img_id])} Predictions (Mask + BBox)")
    plt.savefig(OUTPUT_IMG, bbox_inches='tight')
    print(f"Saved to {OUTPUT_IMG}")

if __name__ == "__main__":
    main()
