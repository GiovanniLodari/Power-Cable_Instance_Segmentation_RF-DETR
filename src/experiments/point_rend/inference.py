
import os
import cv2
import torch
import numpy as np
import json
from tqdm import tqdm
from detectron2.engine import DefaultPredictor
from detectron2.data import MetadataCatalog
import pycocotools.mask as mask_util
from train import setup, register_datasets

def get_predictor(weights_path):
    class Args:
        pass
    args = Args()
    args.eval_only = True
    args.resume = False
    
    cfg = setup(args)
    # If explicit weights passed, use them. Else default to model_final (or whatever config says)
    if weights_path:
        cfg.MODEL.WEIGHTS = weights_path
    else:
        cfg.MODEL.WEIGHTS = os.path.join(cfg.OUTPUT_DIR, "model_final.pth")
    
    # Inference specific
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.5 # Default, tune if needed
    cfg.TEST.DETECTIONS_PER_IMAGE = 100
    
    # PointRend Inference Params (Turbo)
    cfg.MODEL.ROI_MASK_HEAD.POINT_HEAD_ON = True # CRITICAL: Enable Point Head!
    cfg.MODEL.POINT_HEAD.SUBDIVISION_STEPS = 5
    cfg.MODEL.POINT_HEAD.SUBDIVISION_NUM_POINTS = 8192
    
    predictor = DefaultPredictor(cfg)
    return predictor

def run_inference(weights_path, input_json, output_json, img_root):
    register_datasets() # Ensure metadata is available
    predictor = get_predictor(weights_path)
    
    with open(input_json, 'r') as f:
        coco_data = json.load(f)
        
    predictions = []
    
    print(f"Running inference on {len(coco_data['images'])} images...")
    
    for img_info in tqdm(coco_data['images']):
        file_name = img_info['file_name']
        img_path = os.path.join(img_root, file_name)
        
        image = cv2.imread(img_path)
        if image is None:
            print(f"Warning: Could not read {img_path}")
            continue
            
        outputs = predictor(image)
        instances = outputs["instances"].to("cpu")
        
        # Extract Preds
        scores = instances.scores.numpy()
        classes = instances.pred_classes.numpy()
        boxes = instances.pred_boxes.tensor.numpy() # XYXY
        
        # Bitmasks (High Res from PointRend)
        # PointRend returns 'pred_masks' as (N, H, W) bitmasks (float or bool?)
        # Detectron2 post-processing usually thresholds them to boolean.
        
        pred_masks = instances.pred_masks.numpy()
        
        for i in range(len(instances)):
            class_id = int(classes[i])
            # Force class 0 (Cable) if trained as single class
            score = float(scores[i])
            bbox = boxes[i].tolist()
            
            # Convert XYXY to XYWH
            x, y, x2, y2 = bbox
            w = x2 - x
            h = y2 - y
            bbox_xywh = [x, y, w, h]
            
            # Encode Mask
            mask = pred_masks[i].astype(np.uint8)
            # RLE encode
            rle = mask_util.encode(np.asfortranarray(mask))
            rle['counts'] = rle['counts'].decode('utf-8')
            
            predictions.append({
                "image_id": img_info['id'],
                "file_name": file_name,
                "category_id": 0, # Always cable
                "bbox": bbox_xywh,
                "score": score,
                "segmentation": rle
            })
            
    with open(output_json, 'w') as f:
        json.dump(predictions, f)
    print(f"Saved {len(predictions)} predictions to {output_json}")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--weights", type=str, required=True)
    parser.add_argument("--input", type=str, required=True)
    parser.add_argument("--output", type=str, required=True)
    parser.add_argument("--img_root", type=str, required=True)
    args = parser.parse_args()
    
    run_inference(args.weights, args.input, args.output, args.img_root)
