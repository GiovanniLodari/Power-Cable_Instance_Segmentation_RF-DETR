
import os
import cv2
import torch
import numpy as np
import json
from tqdm import tqdm
from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg
from detectron2.projects.point_rend import add_pointrend_config
import pycocotools.mask as mask_util
from detectron2.model_zoo import model_zoo

def setup(args):
    cfg = get_cfg()
    add_pointrend_config(cfg)
    cfg.merge_from_file(model_zoo.get_config_file("COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml"))
    
    # 4K ARCHITECTURE CONFIG (Must match Training)
    cfg.MODEL.ANCHOR_GENERATOR.SIZES = [[16, 32, 64, 128, 256]]
    cfg.MODEL.ANCHOR_GENERATOR.ASPECT_RATIOS = [[0.1, 0.2, 0.5, 1.0, 2.0, 5.0, 10.0]]
    
    cfg.MODEL.ROI_HEADS.NAME = "PointRendROIHeads"
    cfg.MODEL.ROI_MASK_HEAD.POINT_HEAD_ON = True
    cfg.MODEL.ROI_MASK_HEAD.NAME = "CoarseMaskHead"
    cfg.MODEL.ROI_MASK_HEAD.POOLER_RESOLUTION = 28
    cfg.MODEL.ROI_MASK_HEAD.OUTPUT_SIDE_RESOLUTION = 56
    cfg.MODEL.POINT_HEAD.TRAIN_NUM_POINTS = 2048
    cfg.MODEL.POINT_HEAD.SUBDIVISION_NUM_POINTS = 8192
    
    cfg.MODEL.RPN.IOU_THRESHOLDS = [0.3, 0.7]
    cfg.MODEL.RPN.POST_NMS_TOPK_TRAIN = 2000
    cfg.MODEL.RPN.POST_NMS_TOPK_TEST = 2000
    
    cfg.INPUT.MASK_FORMAT = "bitmask"
    
    # INFERENCE SPECIFIC
    cfg.MODEL.WEIGHTS = args.weights
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.5
    cfg.MODEL.POINT_HEAD.SUBDIVISION_STEPS = 5 
    
    # RELIABILITY STRATEGY: High Res (2000px) but fit in memory.
    # Native 4K (3840) is too big for 8GB. 
    # 2000px is 2.5x standard resolution. PointRend handles the fine detail.
    cfg.INPUT.MIN_SIZE_TEST = 2000
    cfg.INPUT.MAX_SIZE_TEST = 3840
    
    # Cap detections to ensure stability
    cfg.TEST.DETECTIONS_PER_IMAGE = 300
    
    return cfg

def run_inference(weights_path, input_json, output_json, img_root):
    class Args: pass
    args = Args()
    args.weights = weights_path
    
    cfg = setup(args)
    predictor = DefaultPredictor(cfg)
    
    with open(input_json, 'r') as f:
        coco_data = json.load(f)
        
    predictions = []
    print(f"Running 4K Inference on {len(coco_data['images'])} images...")
    
    # Filter only first 20 images for quick check? No, run all.
    # Monitor memory.
    
    for img_info in tqdm(coco_data['images']):
        file_name = img_info['file_name']
        img_path = os.path.join(img_root, file_name)
        
        image = cv2.imread(img_path)
        if image is None:
            print(f"Warning: Could not read {img_path}")
            continue
            
        # Predictor handles resizing internally based on cfg.INPUT
        outputs = predictor(image)
        instances = outputs["instances"].to("cpu")
        
        # Extract features
        scores = instances.scores.numpy()
        classes = instances.pred_classes.numpy()
        boxes = instances.pred_boxes.tensor.numpy()
        pred_masks = instances.pred_masks.numpy()
        
        for i in range(len(instances)):
            score = float(scores[i])
            bbox = boxes[i].tolist()
            x, y, x2, y2 = bbox
            w = x2 - x
            h = y2 - y
            bbox_xywh = [x, y, w, h]
            
            mask = pred_masks[i].astype(np.uint8)
            rle = mask_util.encode(np.asfortranarray(mask))
            rle['counts'] = rle['counts'].decode('utf-8')
            
            predictions.append({
                "image_id": img_info['id'],
                "file_name": file_name,
                "category_id": 0, # MATCHES TEST.JSON (Checked via head check)
                "bbox": bbox_xywh,
                "score": score,
                "segmentation": rle
            })
            
    with open(output_json, 'w') as f:
        json.dump(predictions, f)
    print(f"Saved predictions to {output_json}")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    # model_final.pth might not exist if trainer didn't finish standardly.
    # model_temp_chunk.pth contains the latest weights from the smart loop.
    parser.add_argument("--weights", type=str, default="experiments/train_4K/model_temp_chunk.pth")
    parser.add_argument("--input", type=str, default="data/test_original_size/test.json")
    parser.add_argument("--output", type=str, default="experiments/train_4K/predictions_4k.json")
    parser.add_argument("--img_root", type=str, default="data/test_original_size")
    args = parser.parse_args()
    
    run_inference(args.weights, args.input, args.output, args.img_root)
