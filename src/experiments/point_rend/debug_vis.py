
import json
import cv2
import numpy as np
import pycocotools.mask as mask_util
import os

def visualize(pred_path, img_dir, output_dir):
    with open(pred_path) as f:
        preds = json.load(f)
    
    # Pick prediction for image 68 (from previous 'head' check)
    # Image 68 is "04_3420.jpg"? Need to check test.json mapping
    # Just grab the first prediction.
    
    target_pred = preds[0]
    img_id = target_pred['image_id']
    
    # Find filename
    with open('data/test_original_size/test.json') as f:
        gt = json.load(f)
        filename = next(img for img in gt['images'] if img['id'] == img_id)['file_name']
        
    print(f"Visualizing Prediction for {filename} (ID: {img_id})")
    
    img_path = os.path.join(img_dir, filename)
    img = cv2.imread(img_path)
    
    if img is None:
        print("Image not found!")
        return

    # Decode Mask
    rle = target_pred['segmentation']
    mask = mask_util.decode(rle)
    
    # Mask is (H, W) or (W, H)? Detectron2 RLE is usually (H, W).
    print(f"Mask Shape: {mask.shape}, Image Shape: {img.shape}")
    
    # Draw Green Mask
    # Create colored mask
    colored_mask = np.zeros_like(img)
    colored_mask[:, :, 1] = 255 # Green
    
    # Overlay
    alpha = 0.5
    mask_indices = mask > 0
    img[mask_indices] = cv2.addWeighted(img[mask_indices], 1-alpha, colored_mask[mask_indices], alpha, 0)
    
    # Draw BBox
    x, y, w, h = target_pred['bbox']
    cv2.rectangle(img, (int(x), int(y)), (int(x+w), int(y+h)), (0, 0, 255), 2)
    
    os.makedirs(output_dir, exist_ok=True)
    out_path = os.path.join(output_dir, "debug_vis.jpg")
    cv2.imwrite(out_path, img)
    print(f"Saved to {out_path}")

visualize(
    'experiments/train_4K/predictions_ultra_safe.json',
    'data/test_original_size/',
    'experiments/train_4K/'
)
