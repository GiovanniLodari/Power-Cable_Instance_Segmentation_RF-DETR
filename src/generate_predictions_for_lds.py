
import os
import cv2
import cv2
import torch
import numpy as np
import json
import pycocotools.mask as mask_util
from tqdm import tqdm
from detectron2.config import get_cfg
from detectron2.engine import DefaultPredictor
from detectron2.projects.deeplab import add_deeplab_config
from mask2former import add_maskformer2_config
from detectron2.data import DatasetCatalog, MetadataCatalog
from train_resnet import load_ttpla_dataset # Re-use dataset loading
from backbone_unet import UNetPPBackbone # Strategy B Backbone


def skeletonize(img):
    """
    Morphological Skeletonization to reduce mask to 1px centerline.
    """
    img = img.copy()
    skel = np.zeros(img.shape, np.uint8)
    element = cv2.getStructuringElement(cv2.MORPH_CROSS, (3,3))
    
    while True:
        eroded = cv2.erode(img, element)
        temp = cv2.dilate(eroded, element)
        temp = cv2.subtract(img, temp)
        skel = cv2.bitwise_or(skel, temp)
        img = eroded.copy()
        
        if cv2.countNonZero(img) == 0:
            break
            
    return skel



def setup_config(weights_path):
    cfg = get_cfg()
    add_deeplab_config(cfg)
    add_maskformer2_config(cfg)
    
    # Load the EXACT config used for training
    config_file = "models/output_m2f_unetpp_custom/config.yaml"
    print(f"Loading config from {config_file}...")
    cfg.set_new_allowed(True)
    cfg.merge_from_file(config_file)
    cfg.set_new_allowed(False)
    
    # Overwrite weights path
    cfg.MODEL.WEIGHTS = weights_path
    cfg.MODEL.DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    
    # Resolution settings (align with training or use higher for cleaner masks)
    cfg.INPUT.MIN_SIZE_TEST = 1400
    cfg.INPUT.MAX_SIZE_TEST = 1866
    
    # CRITICAL: Lower thresholds for thin objects
    cfg.MODEL.MASK_FORMER.TEST.OBJECT_MASK_THRESHOLD = 0.2
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.01
    cfg.MODEL.RETINANET.SCORE_THRESH_TEST = 0.01
    
    return cfg

def mask_to_line(binary_mask):
    """
    Fits a line to a binary mask using cv2.fitLine.
    Returns (rho, theta) in pixels/radians.
    """
    # Find non-zero points
    y_idxs, x_idxs = np.nonzero(binary_mask)
    
    if len(x_idxs) < 10: # Too small to fit a line
        return None, None
        
    pts = np.column_stack((x_idxs, y_idxs)).astype(np.float32)
    
    # Fit line: returns normalized vector (vx, vy) and point on line (x0, y0)
    [vx, vy, x0, y0] = cv2.fitLine(pts, cv2.DIST_L2, 0, 0.01, 0.01)
    
    # Convert to rho, theta standard form: x*cos(theta) + y*sin(theta) = rho
    # Normal vector (nx, ny) is orthogonal to (vx, vy)
    # nx = -vy, ny = vx
    nx, ny = -vy, vx
    
    # theta is angle of normal vector
    theta = np.arctan2(ny, nx)
    
    # Ensure theta in [0, pi] for consistency with standard Hough/Polar
    if theta < 0:
        theta += np.pi
        nx = -nx
        ny = -ny
        
    # rho = x0*nx + y0*ny
    rho = x0 * nx + y0 * ny
    
    return float(rho), float(theta) # Return natives

def main():
    # Setup Paths
    weights_path = "models/output_m2f_unetpp_custom/model_final.pth"
    output_pred = "models/predictions_val_custom.json"
    output_gt = "models/val_gt_custom.json"
    val_json_path = "data/combined/val_combined.json"
    
    print(f"Loading model from {weights_path}...")
    cfg = setup_config(weights_path)
    predictor = DefaultPredictor(cfg)
    
    # Load Validation Set directly from combined JSON
    print(f"Loading Validation Set from {val_json_path}...")
    with open(val_json_path, 'r') as f:
        val_data = json.load(f)
        
    # Load original train.json just to get 'info' and 'licenses' metadata
    print("Loading metadata from data/train/train.json...")
    with open("data/train/train.json", 'r') as f:
        meta_source = json.load(f)
        
    dataset_dicts = []
    # Convert COCO standard format to Detectron2 format roughly for inference loop
    images = {img['id']: img for img in val_data['images']}
    
    # Create dataset_dicts list for iteration
    for img_info in val_data['images']:
        dataset_dicts.append(img_info)

    # ---------------------------------------------------------
    # Generate val_gt.json (Copy combined val + inject metadata)
    # ---------------------------------------------------------
    print(f"Saving Ground Truth to {output_gt}...")
    
    # Inject metadata if missing
    if "info" not in val_data:
        val_data["info"] = meta_source.get("info", {})
    if "licenses" not in val_data:
        val_data["licenses"] = meta_source.get("licenses", [])
        
    with open(output_gt, "w") as f:
        json.dump(val_data, f)
        
    # ---------------------------------------------------------
    # Inference
    # ---------------------------------------------------------
    coco_results = []
    print(f"Running inference on {len(dataset_dicts)} validation images...")
    
    for d in tqdm(dataset_dicts):
        img_path = d["file_name"]
        # Robust path resolution
        if not os.path.exists(img_path):
             # Check data/all_images (Flattened structure)
             basename = os.path.basename(img_path)
             flattened_path = os.path.join("data/all_images", basename)
             
             if os.path.exists(flattened_path):
                 img_path = flattened_path
             elif os.path.exists("data/" + img_path):
                 img_path = "data/" + img_path
             elif os.path.exists(basename):
                 img_path = basename
        
        image_id = d["id"]
        img = cv2.imread(img_path)
        if img is None:
            print(f"Warning: Could not read {img_path}")
            continue
            
        outputs = predictor(img)
        instances = outputs["instances"].to("cpu")
        
        num_instances = len(instances)
        scores = instances.scores.numpy()
        pred_masks = instances.pred_masks.numpy()
        
        for i in range(num_instances):
            score = float(scores[i])
            if score < 0.01: continue 
            
            mask = pred_masks[i].astype(np.uint8)
            
            # CRITICAL FIX: Skeletonization + Thin Dilation (Cross Kernel)
            # Cross kernel adds pixels only in 4-neighborhood, making the line ~2px effective width
            # (Center + 1 neighbor), which is ideal for matching 1.5px GT.
            # mask = skeletonize(mask)
            # kernel = cv2.getStructuringElement(cv2.MORPH_CROSS, (3,3))
            # mask = cv2.dilate(mask, kernel, iterations=1)
            
            rle = mask_util.encode(np.asfortranarray(mask))
            rle["counts"] = rle["counts"].decode("utf-8")
            
            bbox = instances.pred_boxes.tensor.numpy()[i]
            x, y, x2, y2 = bbox
            w, h = x2-x, y2-y
            
            area = float(mask_util.area(rle))
            rho, theta = mask_to_line(mask)
            
            res = {
                "image_id": image_id,
                "category_id": 0,
                "bbox": [float(x), float(y), float(w), float(h)],
                "score": score,
                "segmentation": rle,
                "area": area,
                "id": i
            }
            if rho is not None:
                res["lines"] = [rho, theta]
                
            coco_results.append(res)
            
    print(f"Saving {len(coco_results)} predictions to {output_pred}...")
    with open(output_pred, "w") as f:
        json.dump(coco_results, f)
        
    print("Done!")

if __name__ == "__main__":
    main()
