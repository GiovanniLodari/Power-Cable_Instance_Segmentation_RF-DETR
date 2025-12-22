
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
CROP_SIZE = 600
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True,max_split_size_mb:64"

def setup_cfg(weights_path):
    cfg = get_cfg()
    add_pointrend_config(cfg)
    cfg.merge_from_file(model_zoo.get_config_file("COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml"))

    # Model Config (Same as Ultra Safe)
    cfg.MODEL.ROI_HEADS.NAME = "PointRendROIHeads"
    cfg.MODEL.ROI_MASK_HEAD.POINT_HEAD_ON = True
    cfg.MODEL.ROI_MASK_HEAD.NAME = "CoarseMaskHead"
    
    cfg.MODEL.ANCHOR_GENERATOR.SIZES = [[16, 32, 64, 128, 256]]
    cfg.MODEL.ANCHOR_GENERATOR.ASPECT_RATIOS = [[0.1, 0.2, 0.5, 1.0, 2.0, 5.0, 10.0]]
    
    cfg.MODEL.ROI_MASK_HEAD.POOLER_RESOLUTION = 28
    cfg.MODEL.ROI_MASK_HEAD.OUTPUT_SIDE_RESOLUTION = 56
    cfg.MODEL.POINT_HEAD.TRAIN_NUM_POINTS = 2048
    cfg.MODEL.POINT_HEAD.SUBDIVISION_NUM_POINTS = 4096 
    
    cfg.MODEL.WEIGHTS = weights_path
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.5
    cfg.MODEL.POINT_HEAD.SUBDIVISION_STEPS = 3
    
    cfg.MODEL.RPN.POST_NMS_TOPK_TEST = 1000 
    cfg.TEST.DETECTIONS_PER_IMAGE = 100
    
    # We verify crops manually
    cfg.INPUT.MIN_SIZE_TEST = 0
    cfg.INPUT.MAX_SIZE_TEST = 99999
    
    cfg.MODEL.DEVICE = DEVICE
    return cfg

def run_smart_crop_inference(weights, input_json, img_root, output_file):
    cfg = setup_cfg(weights)
    predictor = DefaultPredictor(cfg)
    
    with open(input_json) as f:
        data = json.load(f)
        
    predictions = []
    print(f"Starting ORACLE (Smart Crop) Inference on {DEVICE}...")
    
    # We need to group annotations by image to handle them efficiently
    imgs = {i['id']: i for i in data['images']}
    anns_by_img = {}
    for a in data['annotations']:
        if a['image_id'] not in anns_by_img: anns_by_img[a['image_id']] = []
        anns_by_img[a['image_id']].append(a)
        
    for img_id, anns in tqdm(anns_by_img.items()):
        img_info = imgs[img_id]
        file_name = img_info['file_name']
        path = os.path.join(img_root, file_name)
        full_image = cv2.imread(path)
        if full_image is None: continue
        
        H, W = full_image.shape[:2]
        
        # For each True Object, make a crop and try to detect it
        for ann in anns:
            bbox = ann['bbox']
            cx = bbox[0] + bbox[2] / 2
            cy = bbox[1] + bbox[3] / 2
            
            # Create crop centered on object
            x1 = int(max(0, min(W - CROP_SIZE, cx - CROP_SIZE // 2)))
            y1 = int(max(0, min(H - CROP_SIZE, cy - CROP_SIZE // 2)))
            x2 = x1 + CROP_SIZE
            y2 = y1 + CROP_SIZE
            
            crop = full_image[y1:y2, x1:x2]
            
            # Infer
            try:
                outputs = predictor(crop)
                instances = outputs["instances"].to("cpu")
                del outputs
                torch.cuda.empty_cache()
            except:
                continue
                
            # Process results
            for i in range(len(instances)):
                score = float(instances.scores[i])
                if score < 0.5: continue
                
                # Offset Box back to Global
                box = instances.pred_boxes.tensor.numpy()[i]
                box[0] += x1
                box[1] += y1
                box[2] += x1
                box[3] += y1
                
                # Reconstruct full size RLE (Expensive but necessary for Eval)
                local_mask = instances.pred_masks.numpy()[i].astype(np.uint8)
                full_mask = np.zeros((H, W), dtype=np.uint8, order='F')
                full_mask[y1:y2, x1:x2] = local_mask
                
                rle = mask_util.encode(full_mask)
                rle['counts'] = rle['counts'].decode('utf-8')
                
                predictions.append({
                    "image_id": img_id,
                    "category_id": 0,
                    "bbox": [float(x) for x in [box[0], box[1], box[2]-box[0], box[3]-box[1]]],
                    "score": score,
                    "segmentation": rle
                })
                
    with open(output_file, 'w') as f:
        json.dump(predictions, f)
    print("Done.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--weights", default="experiments/train_4K/model_temp_chunk.pth")
    parser.add_argument("--input", default="data/test_original_size/test.json")
    parser.add_argument("--img_root", default="data/test_original_size/")
    parser.add_argument("--output", default="experiments/train_4K/predictions_smart_crop.json")
    args = parser.parse_args()
    
    run_smart_crop_inference(args.weights, args.input, args.img_root, args.output)
