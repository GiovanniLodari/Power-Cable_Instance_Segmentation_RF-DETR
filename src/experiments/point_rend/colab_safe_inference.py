
import os
import cv2
import torch
import json
import numpy as np
import gc
from detectron2.config import get_cfg
from detectron2.engine import DefaultPredictor
from detectron2.projects.point_rend import add_pointrend_config
from detectron2.model_zoo import model_zoo
import pycocotools.mask as mask_util
from tqdm import tqdm

# --- MEMORY SAFE CONFIGURATION ---
def setup_cfg(weights_path):
    cfg = get_cfg()
    add_pointrend_config(cfg)
    cfg.merge_from_file(model_zoo.get_config_file("COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml"))

    # 4K Specific Logic
    cfg.MODEL.ANCHOR_GENERATOR.SIZES = [[16, 32, 64, 128, 256]]
    cfg.MODEL.ANCHOR_GENERATOR.ASPECT_RATIOS = [[0.1, 0.2, 0.5, 1.0, 2.0, 5.0, 10.0]]
    cfg.MODEL.ROI_HEADS.NAME = "PointRendROIHeads"
    cfg.MODEL.ROI_MASK_HEAD.POINT_HEAD_ON = True
    cfg.MODEL.ROI_MASK_HEAD.NAME = "CoarseMaskHead"

    # High Res PointRend Parameters
    cfg.MODEL.ROI_MASK_HEAD.POOLER_RESOLUTION = 28
    cfg.MODEL.ROI_MASK_HEAD.OUTPUT_SIDE_RESOLUTION = 56
    cfg.MODEL.POINT_HEAD.TRAIN_NUM_POINTS = 2048
    cfg.MODEL.POINT_HEAD.SUBDIVISION_NUM_POINTS = 8192

    cfg.MODEL.WEIGHTS = weights_path
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.5
    cfg.MODEL.POINT_HEAD.SUBDIVISION_STEPS = 5

    # --- MEMORY SAFETY LEVEL: ULTRA ---
    # 1. Resolution: Capped to ~2300px long edge. 
    #    (PointRend upscaling handles the rest).
    cfg.INPUT.MIN_SIZE_TEST = 1300 
    cfg.INPUT.MAX_SIZE_TEST = 2300 

    # 2. Strict Detection Limits
    #    Too many cables = OOM during post-processing
    cfg.TEST.DETECTIONS_PER_IMAGE = 100    # Cap at 100 most confident cables
    cfg.MODEL.RPN.POST_NMS_TOPK_TEST = 500 # Limit RPN candidates

    return cfg

def run_inference(weights, input_json, img_root, output_file):
    cfg = setup_cfg(weights)
    predictor = DefaultPredictor(cfg)

    with open(input_json) as f:
        data = json.load(f)

    predictions = []

    print(f"Running Memory-Safe Inference on {len(data['images'])} images...")
    
    for img_info in tqdm(data['images']):
        # --- AGGRESSIVE CLEANUP START ---
        torch.cuda.empty_cache()
        gc.collect()
        # ----------------------------
        
        file_name = img_info['file_name']
        path = os.path.join(img_root, file_name)
        img = cv2.imread(path)

        if img is None:
            print(f"Warning: Image not found at {path}. Skipping.")
            continue

        try:
            # --- FP16 INFERENCE ---
            # Halves tensor memory usage
            with torch.cuda.amp.autocast():
                outputs = predictor(img)
            
            # Move immediately to CPU and detach
            instances = outputs["instances"].to("cpu")
            del outputs
            
            # Extract RLE
            # Loop manually to keep memory usage flat
            pred_masks = instances.pred_masks.numpy()
            scores = instances.scores.numpy()
            boxes = instances.pred_boxes.tensor.numpy()
            
            for i in range(len(instances)):
                mask = pred_masks[i].astype(np.uint8)
                # Compress Mask immediately
                rle = mask_util.encode(np.asfortranarray(mask))
                rle['counts'] = rle['counts'].decode('utf-8')

                predictions.append({
                    "image_id": img_info['id'],
                    "category_id": 0, 
                    "bbox": boxes[i].tolist(),
                    "score": float(scores[i]),
                    "segmentation": rle
                })
            
            del instances
            del pred_masks, scores, boxes

        except Exception as e:
            print(f"Error processing {file_name}: {e}")
            # Try to recover memory if one image fails
            torch.cuda.empty_cache()
            continue

    with open(output_file, 'w') as f:
        json.dump(predictions, f)
    print(f"Done! Saved to {output_file}")


# Define paths for inference
# ADJUST THESE IF NEEDED
weights_path = 'model_temp_chunk.pth' # If uploaded to root
input_json_path = 'data/test_original_size/test.json'
img_root_path = 'data/test_original_size/'
output_predictions_file = 'predictions_4k_safe.json'

# Run
if __name__ == "__main__":
    run_inference(weights_path, input_json_path, img_root_path, output_predictions_file)
