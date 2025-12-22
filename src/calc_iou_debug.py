
import json
import numpy as np
import cv2
from pycocotools import mask as mask_util

def main():
    gt_path = 'data/combined/val_combined.json'
    pred_path = 'experiments/yolo/val_predictions_lines.json'
    img_id = 1023
    
    with open(gt_path, 'r') as f:
        gt_data = json.load(f)
    print("Loaded GT")
        
    with open(pred_path, 'r') as f:
        pred_data = json.load(f)
    print("Loaded Preds")
        
    # Get GT Annotation 1 (Largest)
    gt_anns = [a for a in gt_data['annotations'] if a['image_id'] == img_id]
    gt_ann = sorted(gt_anns, key=lambda x: x['bbox'][2]*x['bbox'][3], reverse=True)[0]
    print(f"GT Bbox: {gt_ann['bbox']}")
    
    # Get Pred 1 (Highest Score)
    preds = [p for p in pred_data if p['image_id'] == img_id]
    pred = sorted(preds, key=lambda x: x['score'], reverse=True)[0]
    print(f"Pred Bbox: {pred['bbox']}")
    
    # Decode Masks
    h, w = 700, 700
    
    # GT Mask
    gt_mask = np.zeros((h, w), dtype=np.uint8)
    if isinstance(gt_ann['segmentation'], list):
        for poly in gt_ann['segmentation']:
            # Poly is [x,y,x,y...]
            pts = np.array(poly).reshape(-1, 2).astype(np.int32)
            cv2.fillPoly(gt_mask, [pts], 1)
    else:
        gt_mask = mask_util.decode(gt_ann['segmentation'])
        
    # Pred Mask
    pred_mask = mask_util.decode(pred['segmentation'])
    # Pred mask might be shape (w, h)? No decode returns (h,w) usually.
    if pred_mask.shape != (h, w):
        print(f"Resizing pred mask from {pred_mask.shape} to {(h, w)}")
        pred_mask = cv2.resize(pred_mask, (w, h), interpolation=cv2.INTER_NEAREST)
        
    # Compute IoU
    intersection = np.logical_and(gt_mask, pred_mask).sum()
    union = np.logical_or(gt_mask, pred_mask).sum()
    iou = intersection / union if union > 0 else 0
    
    print(f"IoU: {iou}")
    
    # Bbox IoU
    xg1, yg1, wg, hg = gt_ann['bbox']
    xg2, yg2 = xg1 + wg, yg1 + hg
    
    xp1, yp1, wp, hp = pred['bbox']
    xp2, yp2 = xp1 + wp, yp1 + hp
    
    xi1 = max(xg1, xp1)
    yi1 = max(yg1, yp1)
    xi2 = min(xg2, xp2)
    yi2 = min(yg2, yp2)
    
    inter_area = max(0, xi2 - xi1) * max(0, yi2 - yi1)
    gt_area = wg * hg
    pred_area = wp * hp
    union_area = gt_area + pred_area - inter_area
    
    bbox_iou = inter_area / union_area if union_area > 0 else 0
    print(f"Bbox IoU: {bbox_iou}")

if __name__ == "__main__":
    main()
