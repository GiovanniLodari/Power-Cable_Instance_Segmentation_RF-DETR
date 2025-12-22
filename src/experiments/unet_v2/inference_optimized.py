
import os
import cv2
import torch
import json
import argparse
import numpy as np
from pathlib import Path
from tqdm import tqdm
import segmentation_models_pytorch as smp
import albumentations as A
from albumentations.pytorch import ToTensorV2
from pycocotools import mask as mask_util
from scipy import ndimage

# --- CONFIG ---
# Script in src/experiments/unet_v2/
PROJECT_ROOT = Path(__file__).resolve().parents[3] 
MODEL_PATH = PROJECT_ROOT / "models" / "experiments" / "unet_v2_output" / "best_model.pth"
DATA_JSON = PROJECT_ROOT / "data" / "combined" / "val_combined.json"
OUTPUT_JSON = PROJECT_ROOT / "models" / "predictions_unet_v2_optimized.json"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# --- USER FORMULA ---
def compute_line_params(mask):
    if mask.sum() < 10: return None
    coords = np.column_stack(np.where(mask > 0))
    if len(coords) < 3: return None
    centroid = coords.mean(axis=0)
    try:
        _, _, Vt = np.linalg.svd(coords - centroid)
        theta = np.arctan2(Vt[0, 1], Vt[0, 0])
        if theta < 0: theta += np.pi
        rho = abs(centroid[0] * np.cos(theta) + centroid[1] * np.sin(theta))
        return [float(rho), float(theta)]
    except:
        return None

def load_model():
    print(f"Loading {MODEL_PATH}...")
    model = smp.UnetPlusPlus(
        encoder_name="resnet101",
        encoder_weights=None,
        in_channels=3,
        classes=1,
        activation=None
    )
    # Load weights
    checkpoint = torch.load(MODEL_PATH, map_location=DEVICE)
    if 'model_state_dict' in checkpoint:
        state_dict = checkpoint['model_state_dict']
    else:
        state_dict = checkpoint
        
    model.load_state_dict(state_dict)
    model = model.to(DEVICE).eval()
    return model

def predict_resized(model, image, target_size=704):
    h, w = image.shape[:2]
    transform = A.Compose([
        A.Resize(target_size, target_size),
        A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ToTensorV2()
    ])
    input_tensor = transform(image=image)['image'].unsqueeze(0).to(DEVICE)
    
    with torch.no_grad():
        logits = model(input_tensor)
        # Handle deep supervision list output
        if isinstance(logits, list):
            logits = logits[0] # Take the final output
            
        prob = torch.sigmoid(logits).squeeze().cpu().numpy()
        
    # Resize back to original
    prob_full = cv2.resize(prob, (w, h), interpolation=cv2.INTER_LINEAR)
    return prob_full

def main():
    model = load_model()
    
    if not DATA_JSON.exists():
        print(f"Error: Data file {DATA_JSON} not found")
        return

    with open(DATA_JSON, 'r') as f:
        coco = json.load(f)
        
    # PARAMETER TUNING
    THRESHOLDS = [0.91, 0.92, 0.94, 0.96]
    MIN_AREA = 50
    
    # Dictionary to hold predictions for each threshold
    all_preds = {t: [] for t in THRESHOLDS}
    
    print(f"Running SWEEP {THRESHOLDS} on {len(coco['images'])} images...")
    
    for img_info in tqdm(coco['images']):
        img_id = img_info['id']
        fname = img_info['file_name']
        
        # Path resolution
        p = PROJECT_ROOT / "data" / "combined" / fname 
        if not p.exists():
             p = PROJECT_ROOT / "data" / fname
        if not p.exists():
             p = PROJECT_ROOT / "data" / "all_images" / os.path.basename(fname)
            
        if not p.exists():
            continue
            
        img = cv2.imread(str(p))
        if img is None: continue
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        # Inference ONCE
        prob_map = predict_resized(model, img, target_size=704)
        
        # Threshold Loop
        for thresh in THRESHOLDS:
            # Instance Extraction
            binary_mask = (prob_map > thresh).astype(np.uint8)
            labeled, num_features = ndimage.label(binary_mask)
            
            for i in range(1, num_features + 1):
                component = (labeled == i).astype(np.uint8)
                area = component.sum()
                if area < MIN_AREA: continue
                
                # Score
                score = float(prob_map[component > 0].mean())
                
                # Lines
                lines = compute_line_params(component)
                if not lines: lines = [0.0, 0.0]
                
                # RLE
                rle = mask_util.encode(np.asfortranarray(component))
                rle['counts'] = rle['counts'].decode('utf-8')
                
                # Bbox
                y, x = np.where(component > 0)
                y0, y1, x0, x1 = y.min(), y.max(), x.min(), x.max()
                bbox = [float(x0), float(y0), float(x1-x0+1), float(y1-y0+1)]
                
                all_preds[thresh].append({
                    "image_id": img_id,
                    "category_id": 0,
                    "bbox": bbox,
                    "segmentation": rle,
                    "score": score,
                    "lines": lines,
                    "area": float(area)
                })
            
    # Save all
    for thresh, preds in all_preds.items():
        fname = PROJECT_ROOT / "models" / f"predictions_v2_{thresh}.json"
        print(f"Saving {len(preds)} preds to {fname.name}...")
        with open(fname, 'w') as f:
            json.dump(preds, f)
    
if __name__ == "__main__":
    main()
