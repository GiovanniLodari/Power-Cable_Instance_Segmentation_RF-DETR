
import os
import cv2
import torch
import json
import numpy as np
import gc
import argparse
from tqdm import tqdm
from detectron2.config import get_cfg
from detectron2.engine import DefaultPredictor
from detectron2.projects.point_rend import add_pointrend_config
from detectron2.model_zoo import model_zoo
import pycocotools.mask as mask_util

# --- CONFIGURATION ---
TILE_SIZE = 800
OVERLAP = 200 
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# Enable Shared Memory Fragmentation fix
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True,max_split_size_mb:64"

def setup_cfg(weights_path):
    cfg = get_cfg()
    add_pointrend_config(cfg)
    cfg.merge_from_file(model_zoo.get_config_file("COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml"))

    # Model Config
    cfg.MODEL.ROI_HEADS.NAME = "PointRendROIHeads"
    cfg.MODEL.ROI_MASK_HEAD.POINT_HEAD_ON = True
    cfg.MODEL.ROI_MASK_HEAD.NAME = "CoarseMaskHead"
    
    # 4K / Nano Anchors (CRITICAL FOR WEIGHT LOADING)
    cfg.MODEL.ANCHOR_GENERATOR.SIZES = [[16, 32, 64, 128, 256]]
    cfg.MODEL.ANCHOR_GENERATOR.ASPECT_RATIOS = [[0.1, 0.2, 0.5, 1.0, 2.0, 5.0, 10.0]]
    
    cfg.MODEL.ROI_MASK_HEAD.POOLER_RESOLUTION = 28
    cfg.MODEL.ROI_MASK_HEAD.OUTPUT_SIDE_RESOLUTION = 56
    cfg.MODEL.POINT_HEAD.TRAIN_NUM_POINTS = 2048
    cfg.MODEL.POINT_HEAD.SUBDIVISION_NUM_POINTS = 4096 # Reduced from 8192
    
    cfg.MODEL.WEIGHTS = weights_path
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.5
    cfg.MODEL.POINT_HEAD.SUBDIVISION_STEPS = 3 # Reduced from 5
    
    # MEMORY LIMITS (To allow Spill without Crash)
    cfg.MODEL.RPN.POST_NMS_TOPK_TEST = 200  # Strict limit on proposals
    cfg.TEST.DETECTIONS_PER_IMAGE = 50      # Strict limit on final output/tile
    
    # 3. Tiling requires 0 input resizing (we verify tiles manually)
    cfg.INPUT.MIN_SIZE_TEST = 0
    cfg.INPUT.MAX_SIZE_TEST = 99999
    
    cfg.MODEL.DEVICE = DEVICE
    return cfg

def get_tiles(h, w, tile_size, overlap):
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

def run_inference(weights, input_json, img_root, output_file):
    cfg = setup_cfg(weights)
    predictor = DefaultPredictor(cfg)
    
    with open(input_json) as f:
        data = json.load(f)
        
    predictions = []
    print(f"Starting ULTRA-SAFE Inference on {DEVICE}...")
    print(f"Tile Size: {TILE_SIZE}, Overlap: {OVERLAP}")

    for img_info in tqdm(data['images']):
        file_name = img_info['file_name']
        path = os.path.join(img_root, file_name)
        full_image = cv2.imread(path)
        
        if full_image is None: continue
        
        H, W = full_image.shape[:2]
        tiles = get_tiles(H, W, TILE_SIZE, OVERLAP)
        
        # Per-Image Results Accumulator
        image_boxes = []
        image_scores = []
        image_masks = [] # We'll store compact RLEs not full masks to save RAM
        
        # --- TILE LOOP ---
        for (tx, ty, tw, th) in tiles:
            # 1. Crop Memory Safe
            crop = full_image[ty:ty+th, tx:tx+tw]
            if crop.size == 0: continue
            
            # 2. Infer on GPU (Small Chunk)
            try:
                outputs = predictor(crop)
                instances = outputs["instances"].to("cpu")
                
                # Cleanup GPU immediately
                del outputs
                torch.cuda.empty_cache()
            except Exception as e:
                print(f"Tile Error: {e}")
                torch.cuda.empty_cache()
                continue
                
            # 3. Process Results on CPU
            for i in range(len(instances)):
                score = float(instances.scores[i])
                if score < 0.5: continue
                
                # Adjust Box Coords
                box = instances.pred_boxes.tensor.numpy()[i]
                box[0] += tx
                box[1] += ty
                box[2] += tx
                box[3] += ty
                
                # Construct Mask (Local -> Global)
                # We save sparse RLE directly to save RAM (Full Boolean mask for 4K is 8MB/instance!)
                # Create small mask
                local_mask = instances.pred_masks.numpy()[i].astype(np.uint8)
                
                # Place in full frame (Sparse way? No, we need RLE)
                # Allocating full 4K frame per instance is slow but necessary for RLE
                # Optimization: create full mask ONCE per tile? No, instances differ.
                # Optimization: Use RLE limits?
                
                # Reliable way:
                full_mask = np.zeros((H, W), dtype=np.uint8, order='F') # Fortran for RLE
                full_mask[ty:ty+th, tx:tx+tw] = local_mask
                
                rle = mask_util.encode(full_mask)
                rle['counts'] = rle['counts'].decode('utf-8')
                
                image_boxes.append(box)
                image_scores.append(score)
                image_masks.append(rle)

            del instances
            # Collect Garbage manually
            
        # --- NMS MERGE (CPU) ---
        if len(image_boxes) > 0:
            # Force float32 to match (Numpy is float64 by default)
            boxes_t = torch.tensor(np.array(image_boxes), dtype=torch.float32)
            scores_t = torch.tensor(np.array(image_scores), dtype=torch.float32)
            
            # NMS on CPU
            keep = torch.ops.torchvision.nms(boxes_t, scores_t, 0.5)
            
            for k in keep:
                idx = k.item()
                # XYXY -> XYWH
                x1, y1, x2, y2 = image_boxes[idx]
                w, h = x2-x1, y2-y1
                
                predictions.append({
                    "image_id": img_info['id'],
                    "category_id": 0,
                    "bbox": [float(x) for x in [x1, y1, w, h]],
                    "score": float(image_scores[idx]),
                    "segmentation": image_masks[idx]
                })

        # Periodic Save
        if len(predictions) % 20 == 0:
            with open(output_file, 'w') as f: json.dump(predictions, f)
            gc.collect()

    with open(output_file, 'w') as f:
        json.dump(predictions, f)
    print("Done.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--weights", default="experiments/train_4K/model_temp_chunk.pth")
    parser.add_argument("--input", default="data/test_original_size/test.json")
    parser.add_argument("--img_root", default="data/test_original_size/")
    parser.add_argument("--output", default="experiments/train_4K/predictions_ultra_safe.json")
    args = parser.parse_args()
    
    run_inference(args.weights, args.input, args.img_root, args.output)
