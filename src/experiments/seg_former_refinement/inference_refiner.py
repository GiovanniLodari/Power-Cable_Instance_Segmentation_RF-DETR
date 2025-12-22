
import os
import sys
import json
import cv2
import torch
import torch.nn as nn
import numpy as np
import argparse
from tqdm import tqdm
from pycocotools import mask as mask_utils
import albumentations as A
from albumentations.pytorch import ToTensorV2

# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../")))
from src.experiments.seg_former_refinement.train_refiner import CascadedCableModel
from src.experiments.seg_former.train import Config

def mask_to_line(mask):
    """Fits a line to a binary mask and returns (rho, theta)."""
    if mask.sum() < 10: # Minimum points
        return None, None
    points = cv2.findNonZero(mask)
    if points is None:
        return None, None
    vx, vy, x0, y0 = cv2.fitLine(points, cv2.DIST_L2, 0, 0.01, 0.01)
    
    # Convert to rho, theta
    # Normal vector is (-vy, vx)
    nx, ny = -vy, vx
    theta = np.arctan2(ny, nx)
    rho = x0 * nx + y0 * ny
    
    # Normalize
    if rho < 0:
        rho = -rho
        theta += np.pi
    theta = theta % (2 * np.pi)
    
    return float(rho), float(theta) # Return natives

def main(refiner_path, output_json):
    print(f"Loading Refiner from {refiner_path}...")
    
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    BASE_MODEL = "models/segformer_mit_b5_sota/best_model.pth"
    
    # Load Model
    model = CascadedCableModel(BASE_MODEL, DEVICE).to(DEVICE)
    model.refiner.load_state_dict(torch.load(refiner_path, map_location=DEVICE))
    model.eval()
    
    # Data Setup
    DATA_ROOT = "data/test"
    TEST_JSON = "data/test/test.json"
    CONF_THRESHOLD = 0.4 # Energy threshold (valleys are < 0.4)
    MIN_AREA = 30
    
    with open(TEST_JSON, 'r') as f:
        coco_gt = json.load(f)
        
    predictions = []
    
    # Transform
    transform = A.Compose([
        A.Resize(height=704, width=704),
        A.Normalize(),
        ToTensorV2(),
    ])
    
    print("Running Inference...")
    for img_info in tqdm(coco_gt['images']):
        image_id = img_info['id']
        file_name = img_info['file_name']
        img_path = os.path.join(DATA_ROOT, file_name)
        
        # Read
        image = cv2.imread(img_path)
        if image is None:
            # Try basename
            img_path = os.path.join(DATA_ROOT, os.path.basename(file_name))
            image = cv2.imread(img_path)
            
        original_h, original_w = image.shape[:2]
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Preprocess
        augmented = transform(image=image_rgb)["image"]
        input_tensor = augmented.unsqueeze(0).to(DEVICE)
        
        # Inference
        with torch.no_grad():
            energy_map, _ = model(input_tensor)
            energy_map = energy_map.squeeze().cpu().numpy() # (H, W)
            
        # Resize to Original Size
        energy_map = cv2.resize(energy_map, (original_w, original_h))
        
        # Post-Processing: Energy Thresholding
        # The energy map peaks at cable centers. Valleys are low.
        # We threshold to keep only high energy areas (centers).
        binary_mask = (energy_map > CONF_THRESHOLD).astype(np.uint8)
        
        # Connected Components
        num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(binary_mask, connectivity=8)
        
        for i in range(1, num_labels):
            area = stats[i, cv2.CC_STAT_AREA]
            if area < MIN_AREA:
                continue
                
            instance_mask = (labels == i).astype(np.uint8)
            
            # Dilate to restore thickness (Spine -> Cable)
            # Energy map threshold gives thin skeletons. We need thick masks for IoU.
            # Kernel 15x15 seems appropriate for typical cable width.
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15))
            instance_mask = cv2.dilate(instance_mask, kernel, iterations=1)
            
            # Creating RLE
            rle = mask_utils.encode(np.asfortranarray(instance_mask))
            rle['counts'] = rle['counts'].decode('utf-8')
            
            # Bbox
            x, y, w, h = stats[i, cv2.CC_STAT_LEFT], stats[i, cv2.CC_STAT_TOP], stats[i, cv2.CC_STAT_WIDTH], stats[i, cv2.CC_STAT_HEIGHT]
            bbox = [float(x), float(y), float(w), float(h)]
            
            # Score
            # Use mean energy of the instance as score
            score = float(np.mean(energy_map[instance_mask > 0]))
            
            # Lines
            rho, theta = mask_to_line(instance_mask)
            lines = [rho, theta] if rho is not None else []
            
            pred = {
                "image_id": image_id,
                "category_id": 1, 
                "bbox": bbox,
                "segmentation": rle,
                "score": score,
                "area": float(area),
                "lines": lines
            }
            predictions.append(pred)
            
        # Debug: Save first image viz
        if len(predictions) > 0 and not os.path.exists("debug_inference_refiner.png"):
             debug_mask = binary_mask * 255
             cv2.imwrite("debug_inference_refiner.png", debug_mask)
             cv2.imwrite("debug_inference_energy.png", (energy_map * 255).astype(np.uint8))
            
    print(f"Saving {len(predictions)} predictions to {output_json}...")
    with open(output_json, 'w') as f:
        json.dump(predictions, f)
    print("Done.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--refiner', type=str, default="src/experiments/seg_former_refinement/output/refiner_ep15.pth")
    parser.add_argument('--output', type=str, default="output/predictions_refiner.json")
    args = parser.parse_args()
    
    main(args.refiner, args.output)
