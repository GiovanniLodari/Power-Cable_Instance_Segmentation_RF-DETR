
import json
import numpy as np
import cv2
from collections import defaultdict
from tqdm import tqdm
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
import os
from pycocotools import mask as coco_mask
from skimage.morphology import skeletonize

def evaluate_segmentation(gt_json_path, pred_json_path, check_cable_class=False):
    # Load ground truth
    coco_gt = COCO(gt_json_path)
    
    # Fix: PyCOCOTools requires 'info'
    if 'info' not in coco_gt.dataset:
        coco_gt.dataset['info'] = {}

    # Load predictions
    with open(pred_json_path, 'r') as f:
        predictions = json.load(f)

    # Detect expected category ID from GT
    gt_cat_ids = coco_gt.getCatIds()
    if gt_cat_ids:
        target_cat_id = gt_cat_ids[0]
        # Align predictions to GT category if they differ
        first_pred_cat = predictions[0].get('category_id') if predictions else None
        if first_pred_cat is not None and first_pred_cat != target_cat_id:
            print(f"Warning: Aligning prediction category_id ({first_pred_cat}) to GT category_id ({target_cat_id}).")
            for p in predictions:
                p['category_id'] = target_cat_id
                
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
    
    def extract_line_skeleton_svd(mask):
        """
        Extract line parameters (rho, theta) from a binary mask using SVD on skeleton.
        """
        # Skeletonize to thin the features
        if mask.max() > 0:
            skeleton = skeletonize(mask > 0)
        else:
            return None

        # Get coordinates
        y_coords, x_coords = np.where(skeleton > 0)
        
        # Fallback to full mask if skeleton is too small (e.g. dense blob)
        if len(x_coords) < 5:
            y_coords, x_coords = np.where(mask > 0)
            
        if len(x_coords) < 5:
            return None
        
        # Center the points
        cx, cy = np.mean(x_coords), np.mean(y_coords)
        coords = np.stack([x_coords - cx, y_coords - cy], axis=1)
        
        # SVD to find principal direction
        try:
            _, _, Vt = np.linalg.svd(coords, full_matrices=False)
        except:
            return None
        
        # Principal direction (first row of Vt)
        direction = Vt[0]
        
        # Angle of the line (theta in polar coordinates)
        theta = np.arctan2(direction[1], direction[0])
        
        # Convert to line angle (perpendicular)
        theta_line = theta + np.pi / 2
        if theta_line > np.pi:
            theta_line -= np.pi
        if theta_line < 0:
            theta_line += np.pi
        
        # Rho
        normal = np.array([-direction[1], direction[0]])
        rho = cx * normal[0] + cy * normal[1]
        
        if rho < 0:
            rho = -rho
            theta_line = theta_line + np.pi if theta_line < np.pi else theta_line - np.pi
            
        return rho, theta_line

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
                    # Fallback: Compute from mask using Skeleton+SVD
                    params = extract_line_skeleton_svd(gt_masks[-1] if gt_masks else np.zeros((height, width)))
                    if params:
                        rho, theta = params
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

def filter_gt_by_predictions(gt_path, pred_path):
    with open(pred_path, 'r') as f:
        preds = json.load(f)
    
    pred_image_ids = set(p['image_id'] for p in preds)
    
    with open(gt_path, 'r') as f:
        gt = json.load(f)
        
    gt_image_ids = set(img['id'] for img in gt['images'])
    
    # If predictions are a significant subset of GT, filter GT
    if len(pred_image_ids) < len(gt_image_ids) and pred_image_ids.issubset(gt_image_ids):
        print(f"Predictions ({len(pred_image_ids)}) are a subset of GT ({len(gt_image_ids)}). Filtering GT...")
        
        new_images = [img for img in gt['images'] if img['id'] in pred_image_ids]
        new_annotations = [ann for ann in gt['annotations'] if ann['image_id'] in pred_image_ids]
        
        filtered_gt = gt.copy()
        filtered_gt['images'] = new_images
        filtered_gt['annotations'] = new_annotations
        
        temp_path = gt_path.replace('.json', '_temp_subset.json')
        with open(temp_path, 'w') as f:
            json.dump(filtered_gt, f)
            
        print(f"Created temporary filtered GT: {temp_path}")
        return temp_path
        
    return gt_path

def compute_line_detection_score(gt_json_path, pred_json_path):
    # Filter GT if needed
    actual_gt_path = filter_gt_by_predictions(gt_json_path, pred_json_path)
    
    print(f"Evaluating {pred_json_path} against {actual_gt_path}")
    avg_p50, avg_r50 = evaluate_segmentation(actual_gt_path, pred_json_path)
    rho_diff, angle_diff = combined_analysis(actual_gt_path, pred_json_path)
    
    # Clean up temp file
    if actual_gt_path != gt_json_path and '_temp_subset.json' in actual_gt_path:
        os.remove(actual_gt_path)

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
    parser.add_argument('gt', nargs='?',  default="data/test/test.json")
    parser.add_argument('pred', nargs='?', default="src/experiments/rf_detr/output/predictions_lds2.json")#"src/experiments/pointrend_r101_v7/predictions_lds_skel_tta.json")
    args = parser.parse_args()
    
    compute_line_detection_score(args.gt, args.pred)
