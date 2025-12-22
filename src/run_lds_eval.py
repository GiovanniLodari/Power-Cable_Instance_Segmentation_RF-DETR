
import json
import numpy as np
import cv2
from collections import defaultdict
from tqdm import tqdm
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
from pycocotools import mask as coco_mask

def evaluate_segmentation(gt_json_path, pred_json_path, check_cable_class=False):
    # Load ground truth
    coco_gt = COCO(gt_json_path)
    
    # Fix: PyCOCOTools requires 'info'
    if 'info' not in coco_gt.dataset:
        coco_gt.dataset['info'] = {}

    # Load predictions
    with open(pred_json_path, 'r') as f:
        predictions = json.load(f)

    # Load results into COCO results structure
    coco_res = coco_gt.loadRes(predictions)

    # Create COCOeval object
    coco_eval = COCOeval(coco_gt, coco_res, 'segm')
    if check_cable_class:
        coco_eval.params.catIds = [0]  # id of the cable class

    # Run evaluation
    coco_eval.evaluate()
    coco_eval.accumulate()
    coco_eval.summarize()

    avg_p50 = coco_eval.stats[1]
    avg_r50 = coco_eval.stats[7] # AR maxDets=10 -> Notebook uses index 7!
    # Double check notebook: "avg_r50 = coco_eval.stats[7]"
    # Standard COCO:
    # 6: AR @[ IoU=0.50:0.95 | area=   all | maxDets=  1 ]
    # 7: AR @[ IoU=0.50:0.95 | area=   all | maxDets= 10 ]
    # 8: AR @[ IoU=0.50:0.95 | area=   all | maxDets=100 ]
    
    return avg_p50, avg_r50

def combined_analysis(gt_annotation_file, prediction_file):
    # Load ground truth data
    with open(gt_annotation_file, 'r') as f:
        gt_data = json.load(f)
    # Load prediction data
    with open(prediction_file, 'r') as f:
        pred_data = json.load(f)
        
    # Group GT lines by image id
    gt_lines_by_image = defaultdict(list)
    for ann in gt_data['annotations']:
        image_id = ann['image_id']
        if 'polar_coordinates' in ann:
            lines = [(coord['rho'], coord['theta']) for coord in ann['polar_coordinates']]
            gt_lines_by_image[image_id].extend(lines)
        else:
            # raise RuntimeError(f'no polar coord for image id {image_id}')
            pass # Skip if GT has no polar coords (some might be empty?)

    # Group predictions by image id
    pred_by_image = defaultdict(list)
    for pred in pred_data:
        pred_by_image[pred['image_id']].append(pred)
        
    angle_diffs = []
    rho_diffs = []
    
    def theta_diff(theta_pred, theta_gt):
        t = min(abs(theta_pred - theta_gt), np.pi - abs(theta_pred - theta_gt))
        return np.exp(-.12 * t)
    
    def polygons_to_mask(polygons, shape):
        mask = np.zeros(shape, dtype=np.uint8)
        for polygon in polygons:
            pts = np.array(polygon).reshape((-1, 2)).astype(np.int32)
            cv2.fillPoly(mask, [pts], color=255)
        return mask
    
    def compute_iou(mask1, mask2):
        if mask1.shape != mask2.shape: return 0
        intersection = np.logical_and(mask1, mask2).sum()
        union = np.logical_or(mask1, mask2).sum()
        return intersection / union if union > 0 else 0
    
    total_matches = 0
    total_gt_lines = 0
    total_pred_lines = 0
    
    print("Running Line Matching Analysis...")
    for image_info in tqdm(gt_data['images']):
        image_id = image_info['id']
        height, width = image_info['height'], image_info['width']
        
        # Load predictions for this image
        pred_masks = []
        pred_lines = []
        for pred in pred_by_image.get(image_id, []):
            seg = pred['segmentation']
            if isinstance(seg, list):
                mask_poly = polygons_to_mask(seg, (height, width))
                pred_masks.append(mask_poly)
            elif isinstance(seg, dict) and 'counts' in seg:
                mask_rle = coco_mask.decode(seg)
                if mask_rle.ndim == 3:
                    mask_rle = mask_rle[:, :, 0]
                mask_rle = (mask_rle * 255).astype(np.uint8)
                
                if mask_rle.shape != (height, width):
                     mask_rle = cv2.resize(mask_rle, (width, height), interpolation=cv2.INTER_NEAREST)
                     
                pred_masks.append(mask_rle)
            else:
                continue
            
            # Extract predicted line if exists
            if 'lines' in pred and len(pred['lines']) == 2:
                rho, theta = pred['lines']
                # Normalize rho as per notebook logic
                rho_norm = np.abs(rho / np.sqrt(height**2 + width**2))
                pred_lines.append((rho_norm, theta))
            else:
                pred_lines.append(None)
        
        # Load ground truth masks for this image
        gt_masks = []
        gt_lines = []
        for ann in gt_data['annotations']:
            if ann['image_id'] == image_id:
                seg = ann['segmentation']
                if isinstance(seg, list):
                    mask_poly = polygons_to_mask(seg, (height, width))
                    gt_masks.append(mask_poly)
                elif isinstance(seg, dict) and 'counts' in seg:
                    mask_rle = coco_mask.decode(seg)
                    if mask_rle.ndim == 3:
                        mask_rle = mask_rle[:, :, 0]
                    mask_rle = (mask_rle * 255).astype(np.uint8)
                    gt_masks.append(mask_rle)
                
                # Extract GT line
                if 'polar_coordinates' in ann and len(ann['polar_coordinates']) > 0:
                    rho, theta = ann['polar_coordinates'][0]['rho'], ann['polar_coordinates'][0]['theta']
                    # Normalize GT rho too
                    rho_norm = np.abs(rho / np.sqrt(height**2 + width**2))
                    gt_lines.append((rho_norm, theta))
                else:
                    gt_lines.append(None)
        
        # Detect the matching mask by IoU
        matched_gt = set()
        for pred_idx, pred_mask in enumerate(pred_masks):
            best_iou = 0
            best_gt_idx = -1
            
            for gt_idx, gt_mask in enumerate(gt_masks):
                if gt_idx in matched_gt:
                    continue
                iou = compute_iou(pred_mask, gt_mask)
                if iou > best_iou:
                    best_iou = iou
                    best_gt_idx = gt_idx
            
            # Notebook behavior: match best (implied threshold > 0)
            if best_gt_idx >= 0 and best_iou > 0.01: # Add sanity check threshold
                matched_gt.add(best_gt_idx)
                total_matches += 1
                
                # Compute differences
                pred_line = pred_lines[pred_idx]
                gt_line = gt_lines[best_gt_idx]
                
                if pred_line is not None and gt_line is not None:
                    rho_pred, theta_pred = pred_line
                    rho_gt, theta_gt = gt_line
                    
                    rho_diffs.append(abs(rho_pred - rho_gt))
                    angle_diffs.append(theta_diff(theta_pred, theta_gt))
        
        total_gt_lines += len(gt_masks)
        total_pred_lines += len(pred_masks)
    
    print(f"Total GT lines: {total_gt_lines}")
    print(f"Total predicted lines: {total_pred_lines}")
    print(f"Total matches: {total_matches}")
    print(f"Lines with coordinate differences computed: {len(rho_diffs)}")
    
    if len(rho_diffs) == 0:
        return 0, 0
    
    return np.mean(rho_diffs), np.mean(angle_diffs)

def compute_line_detection_score(gt_json_path, pred_json_path):
    print(f"Evaluating {pred_json_path} against {gt_json_path}")
    avg_p50, avg_r50 = evaluate_segmentation(gt_json_path, pred_json_path)
    rho_diff, angle_diff = combined_analysis(gt_json_path, pred_json_path)

    print(f'\nRESULTS:')
    print(f'  AP@50: {avg_p50:.4f}')
    print(f'  AR@50: {avg_r50:.4f}')
    print(f'  Angle Score: {angle_diff:.4f}')
    print(f'  Rho Diff (Norm): {rho_diff:.4f}')

    lds = avg_p50 + avg_r50 + 2 * angle_diff
    print(f'\nLDS = {lds:.4f}')
    return lds

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('gt', nargs='?', default="data/combined/val_combined.json")
    parser.add_argument('pred', nargs='?', default="models/predictions_unet_instances.json")
    args = parser.parse_args()
    
    compute_line_detection_score(args.gt, args.pred)
