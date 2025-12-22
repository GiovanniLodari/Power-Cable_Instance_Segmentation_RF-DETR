
import os
import cv2
import torch
import json
import numpy as np
import pycocotools.mask as mask_util
from tqdm import tqdm
import albumentations as A
from albumentations.pytorch import ToTensorV2

# Import model architecture from train.py
# (Assuming running from project root)
import sys
sys.path.append(os.getcwd())
from src.experiments.seg_former.train import CustomUnetPlusPlus, Config

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

def get_test_transform():
    return A.Compose([
        A.Resize(height=704, width=704),
        A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ToTensorV2(),
    ])

def main(model_path, output_path):
    # Setup Paths
    MODEL_PATH = model_path
    TEST_JSON = "data/test/test.json" 
    OUTPUT_JSON = output_path
    DATA_ROOT = "data/test" # Images are here
    
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Running on {DEVICE}")
    
    # 1. Load Model
    print(f"Loading model from {MODEL_PATH}...")
    
    # Replicate instantiation from Trainer.__init__
    model = CustomUnetPlusPlus(
        encoder_name=Config.ENCODER,
        encoder_weights=Config.ENCODER_WEIGHTS,
        in_channels=3,
        classes=1,
        encoder_depth=5,
        decoder_channels=(256, 128, 64, 32, 16),
        activation=None,
        decoder_attention_type="scse"
    )
    
    # Load Weights
    state_dict = torch.load(MODEL_PATH, map_location=DEVICE)
    
    # Handle possible 'model._orig_mod.' prefix if compiled
    new_state_dict = {}
    for k, v in state_dict.items():
        if k.startswith("_orig_mod."):
             new_state_dict[k[10:]] = v
        else:
             new_state_dict[k] = v
             
    model.load_state_dict(new_state_dict)
    model.to(DEVICE)
    model.eval()
    
    # 2. Load Test Metadata
    print(f"Loading test data from {TEST_JSON}...")
    with open(TEST_JSON, 'r') as f:
        test_data = json.load(f)
        
    images_info = test_data['images']
    
    # 3. Inference Loop
    results = []
    transforms = get_test_transform()
    
    print(f"Starting inference on {len(images_info)} images...")
    
    # Kernel unused (reverted)
    
    with torch.no_grad():
        for img_info in tqdm(images_info):
            image_id = img_info['id']
            file_name = img_info['file_name']
            orig_h = img_info['height']
            orig_w = img_info['width']
            
            # Robust path finding
            img_path = os.path.join(DATA_ROOT, file_name)
            if not os.path.exists(img_path):
                 basename = os.path.basename(file_name)
                 candidate = os.path.join(DATA_ROOT, basename)
                 if os.path.exists(candidate):
                     img_path = candidate
            
            # Read Image
            image = cv2.imread(img_path)
            if image is None:
                continue
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            
            # Transform
            augmented = transforms(image=image)
            input_tensor = augmented['image'].unsqueeze(0).to(DEVICE)
            
            # Predict
            with torch.cuda.amp.autocast(enabled=True):
                output = model(input_tensor) # Logits (1, 1, 704, 704)
                prob_map = torch.sigmoid(output).squeeze().cpu().numpy().astype(np.float32) # (704, 704)
            
            # Resize back to Original Size
            prob_map_orig = cv2.resize(prob_map, (orig_w, orig_h), interpolation=cv2.INTER_LINEAR)
            
            # Threshold: 0.55 (Higher than 0.5 to separate, Lower than 0.6 to keep connected)
            binary_mask = (prob_map_orig > 0.55).astype(np.uint8)
            
            # Connected Components (Standard)
            # Connectivity 4 (Strict) -> Helps separation
            num_labels, labels = cv2.connectedComponents(binary_mask, connectivity=4)
            
            # Iterate components (skip background 0)
            for label in range(1, num_labels):
                component_mask = (labels == label).astype(np.uint8)
                
                # Filter tiny components
                if component_mask.sum() < 30: # Lowered to 30 to catch small segments
                    continue
                
                # Encoding
                rle = mask_util.encode(np.asfortranarray(component_mask))
                rle['counts'] = rle['counts'].decode('utf-8')
                area = float(mask_util.area(rle))
                
                # Score 
                score = float(prob_map_orig[component_mask == 1].mean())
                
                # Line Parameters
                # Fit to component
                rho, theta = mask_to_line(component_mask)
                
                # Create Result Entry
                bbox = mask_util.toBbox(rle).tolist() # [x, y, w, h]
                
                res = {
                    "image_id": image_id,
                    "category_id": 0, # Cable
                    "bbox": bbox,
                    "score": score,
                    "segmentation": rle,
                    "area": area,
                    "id": len(results) + 1 # Unique ID
                }
                
                if rho is not None:
                    res["lines"] = [rho, theta]
                    
                results.append(res)
                
    # 4. Save
    print(f"Saving {len(results)} predictions to {OUTPUT_JSON}...")
    with open(OUTPUT_JSON, 'w') as f:
        json.dump(results, f)
        
    print("Done.")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, default="models/segformer_mit_b5_sota/best_model.pth", help="Path to model checkpoint")
    parser.add_argument('--output', type=str, default="output/predictions.json", help="Path to output JSON")
    args = parser.parse_args()
    
    main(args.model, args.output)
