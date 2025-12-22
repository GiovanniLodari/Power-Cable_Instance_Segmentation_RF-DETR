
import json
import numpy as np
import pycocotools.mask as mask_util
from shapely.geometry import box, Polygon

def verify_alignment(gt_path, pred_path, image_id_to_check=None):
    print(f"Loading GT: {gt_path}")
    with open(gt_path) as f:
        gt = json.load(f)
        
    print(f"Loading Preds: {pred_path}")
    with open(pred_path) as f:
        preds = json.load(f)
        
    # Group by image
    preds_by_img = {}
    for p in preds:
        if p['image_id'] not in preds_by_img: preds_by_img[p['image_id']] = []
        preds_by_img[p['image_id']].append(p)
        
    avg_ious = []
    
    # Check first 5 images or specific ID
    checked = 0
    for img_info in gt['images']:
        if checked >= 5: break
        
        img_id = img_info['id']
        if image_id_to_check is not None and img_id != image_id_to_check: continue
        
        print(f"\n--- Checking Image {img_id} ({img_info['file_name']}) ---")
        img_w, img_h = img_info['width'], img_info['height']
        
        gt_anns = [a for a in gt['annotations'] if a['image_id'] == img_id]
        img_preds = preds_by_img.get(img_id, [])
        
        print(f"GT Objects: {len(gt_anns)}, Predictions: {len(img_preds)}")
        
        if len(gt_anns) == 0 or len(img_preds) == 0:
            continue
            
        checked += 1
        
        # Check BBox overlaps
        overlaps = 0
        best_ious = []
        
        for ann in gt_anns:
            gx, gy, gw, gh = ann['bbox']
            gt_box = box(gx, gy, gx+gw, gy+gh)
            
            best_iou = 0
            best_match = None
            
            for p in img_preds:
                px, py, pw, ph = p['bbox']
                # Basic overlap check
                # Intersection
                ix1 = max(gx, px)
                iy1 = max(gy, py)
                ix2 = min(gx+gw, px+pw)
                iy2 = min(gy+gh, py+ph)
                
                iw = max(0, ix2 - ix1)
                ih = max(0, iy2 - iy1)
                
                inter_area = iw * ih
                union_area = (gw*gh) + (pw*ph) - inter_area
                
                iou = inter_area / union_area if union_area > 0 else 0
                
                if iou > best_iou:
                    best_iou = iou
                    best_match = p
            
            best_ious.append(best_iou)
            if best_iou > 0.01:
                overlaps += 1
                # print(f"GT Box: {ann['bbox']} matched Pred Box: {best_match['bbox']} with IoU {best_iou:.4f}")
            else:
                # print(f"GT Box: {ann['bbox']} has NO match (Max IoU {best_iou:.4f})")
                pass
                
        print(f"Matched GTs (IoU > 0.01): {overlaps}/{len(gt_anns)}")
        print(f"Avg Best IoU: {np.mean(best_ious):.4f}")
        print(f"Max Best IoU: {np.max(best_ious):.4f}")

if __name__ == "__main__":
    verify_alignment(
        "data/test_original_size/test.json",
        "experiments/train_4K/predictions_with_lines.json"
    )
