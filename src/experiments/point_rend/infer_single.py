
import argparse
import os
import cv2
import torch
import numpy as np
import json
import pycocotools.mask as mask_util
from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg
from detectron2.projects.point_rend import add_pointrend_config
from detectron2.model_zoo import model_zoo

def setup(args):
    cfg = get_cfg()
    add_pointrend_config(cfg)
    cfg.merge_from_file(model_zoo.get_config_file("COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml"))
    
    # 4K CONFIG
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
    cfg.INPUT.MASK_FORMAT = "bitmask"
    
    cfg.MODEL.WEIGHTS = args.weights
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.05 # Low threshold to capture everything (filter later)
    cfg.MODEL.POINT_HEAD.SUBDIVISION_STEPS = 5
    
    # INPUT SIZE
    # User calls with native resolution (e.g. 2160)
    cfg.INPUT.MIN_SIZE_TEST = args.scale 
    cfg.INPUT.MAX_SIZE_TEST = 4000
    
    # MEMORY OPTIMIZATION (Strict)
    # 2000 RPN proposals * 4K mask = OOM.
    # 300 was still too much (~19GB request?? Maybe peak memory).
    # Reducing to 100 detections (Cables are sparse).
    # Reducing RPN proposals to 500.
    cfg.TEST.DETECTIONS_PER_IMAGE = 100 
    cfg.MODEL.RPN.POST_NMS_TOPK_TEST = 500
    
    return cfg

def run(args):
    cfg = setup(args)
    predictor = DefaultPredictor(cfg)
    
    image = cv2.imread(args.image_path)
    if image is None:
        return
        
    # FP16 INFERENCE (Critical for 4K on 8GB)
    with torch.cuda.amp.autocast():
        outputs = predictor(image)
        
    instances = outputs["instances"].to("cpu")
    
    # ... Rest remains same ...
    

# TILED INFERENCE IMPLEMENTATION
# Strategy:
# 1. Slide a window (1500x1500) over the 4K image.
# 2. Predict on each window.
# 3. Shift box/mask coordinates to global frame.
# 4. Merge results using NMS.



# TILED INFERENCE IMPLEMENTATION (Restored)
def get_tiles(h, w, tile_size=1024, overlap=200):
    stride = tile_size - overlap
    tiles = []
    for y in range(0, h, stride):
        for x in range(0, w, stride):
            x2 = min(x + tile_size, w)
            y2 = min(y + tile_size, h)
            w_tile = x2 - x
            h_tile = y2 - y
            tiles.append((x, y, w_tile, h_tile))
    return tiles

def run(args):
    cfg = setup(args)
    # MIN_SIZE_TEST=0 prevents resizing of tiles
    cfg.INPUT.MIN_SIZE_TEST = 0 
    cfg.INPUT.MAX_SIZE_TEST = 99999
    
    predictor = DefaultPredictor(cfg)
    
    image = cv2.imread(args.image_path)
    if image is None: return
    
    H, W = image.shape[:2]
    
    # 1. Generate Tiles
    # 1500 failed (OOM). 1024 is safe (~1MP).
    tiles = get_tiles(H, W, tile_size=1024, overlap=200)
    
    global_results = []
    
    for (tx, ty, tw, th) in tiles:
        # Crop
        crop = image[ty:ty+th, tx:tx+tw]
        
        if crop.size == 0: continue
        
        # Infer on Tile
        outputs = predictor(crop)
        instances = outputs["instances"].to("cpu")
        
        scores = instances.scores.numpy()
        boxes = instances.pred_boxes.tensor.numpy()
        pred_masks = instances.pred_masks.numpy()
        
        # Shift Coordinates to Global
        for i in range(len(instances)):
            x, y, x2, y2 = boxes[i]
            w, h = x2-x, y2-y
            
            # Global Box
            bx = x + tx
            by = y + ty
            bw = w
            bh = h
            
            # Construct Full Mask RLE
            full_mask = np.zeros((H, W), dtype=np.uint8)
            full_mask[int(ty):int(ty+th), int(tx):int(tx+tw)] = pred_masks[i].astype(np.uint8)
            
            rle = mask_util.encode(np.asfortranarray(full_mask))
            rle['counts'] = rle['counts'].decode('utf-8')
            
            global_results.append({
                "category_id": 0,
                "bbox": [bx, by, bw, bh],
                "score": float(scores[i]),
                "segmentation": rle
            })

    # NMS Merging
    if len(global_results) == 0:
        with open(args.output_json, 'w') as f: json.dump([], f)
        return

    boxes_t = torch.tensor([r["bbox"] for r in global_results])
    # Convert XYWH to XYXY for NMS
    boxes_xyxy = boxes_t.clone()
    boxes_xyxy[:, 2] += boxes_xyxy[:, 0]
    boxes_xyxy[:, 3] += boxes_xyxy[:, 1]
    
    scores_t = torch.tensor([r["score"] for r in global_results])
    
    keep_indices = torch.ops.torchvision.nms(boxes_xyxy, scores_t, 0.5)
    
    final_preds = []
    for idx in keep_indices:
        final_preds.append(global_results[idx])
        
    with open(args.output_json, 'w') as f:
        json.dump(final_preds, f)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--image_path", required=True)
    parser.add_argument("--output_json", required=True)
    parser.add_argument("--weights", required=True)
    parser.add_argument("--scale", type=int, default=0) 
    args = parser.parse_args()
    
    run(args)
