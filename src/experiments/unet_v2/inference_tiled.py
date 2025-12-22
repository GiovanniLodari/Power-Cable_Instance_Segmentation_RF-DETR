
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
PROJECT_ROOT = Path(__file__).resolve().parents[3]
MODEL_PATH = PROJECT_ROOT / "models" / "experiments" / "unet_v2_output" / "best_model.pth"
DATA_JSON = PROJECT_ROOT / "data" / "combined" / "val_combined.json"
OUTPUT_JSON = PROJECT_ROOT / "models" / "predictions_unet_v2_tiled.json"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

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
    checkpoint = torch.load(MODEL_PATH, map_location=DEVICE)
    if 'model_state_dict' in checkpoint:
        state = checkpoint['model_state_dict']
    else:
        state = checkpoint
    model.load_state_dict(state)
    model = model.to(DEVICE).eval()
    return model

def predict_patch(model, patch, target_size=704):
    # Resize patch to Model Input Size (704)
    # This creates the "Zoom" effect if patch < 704
    # Or "Downscale" if patch > 704
    h, w = patch.shape[:2]
    transform = A.Compose([
        A.Resize(target_size, target_size),
        A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ToTensorV2()
    ])
    input_tensor = transform(image=patch)['image'].unsqueeze(0).to(DEVICE)
    
    with torch.no_grad():
        logits = model(input_tensor)
        if isinstance(logits, list): logits = logits[0]
        prob = torch.sigmoid(logits).squeeze().cpu().numpy()
        
    # Resize BACK to patch original size
    prob_original = cv2.resize(prob, (w, h), interpolation=cv2.INTER_LINEAR)
    return prob_original

def predict_2x2_tiled(model, image, pad=0):
    H, W = image.shape[:2]
    
    # Define 4 overlapping tiles (approx 60% of width/height each)
    # This ensures center is covered twice
    h_split = int(H * 0.6)
    w_split = int(W * 0.6)
    
    # Coordinates: y1, y2, x1, x2
    tiles = [
        (0, h_split, 0, w_split),             # Top-Left
        (0, h_split, W - w_split, W),         # Top-Right
        (H - h_split, H, 0, w_split),         # Bottom-Left
        (H - h_split, H, W - w_split, W)      # Bottom-Right
    ]
    
    full_prob = np.zeros((H, W), dtype=np.float32)
    count_map = np.zeros((H, W), dtype=np.float32)
    
    for (y1, y2, x1, x2) in tiles:
        patch = image[y1:y2, x1:x2]
        prob_patch = predict_patch(model, patch, target_size=704)
        
        full_prob[y1:y2, x1:x2] += prob_patch
        count_map[y1:y2, x1:x2] += 1.0
        
    # Average overlapping areas
    full_prob /= (count_map + 1e-6)
    return full_prob

def main():
    model = load_model()
    
    with open(DATA_JSON, 'r') as f:
        coco = json.load(f)
        
    predictions = []
    
    # High Threshold because Tiling increases confidence and sharpness
    THRESHOLD = 0.97
    MIN_AREA = 50
    
    print(f"Running 2x2 TILED inference (Thresh={THRESHOLD}) on {len(coco['images'])} images...")
    
    for img_info in tqdm(coco['images']):
        img_id = img_info['id']
        fname = img_info['file_name']
        
        p = PROJECT_ROOT / "data" / "combined" / fname
        if not p.exists(): p = PROJECT_ROOT / "data" / fname
        if not p.exists(): p = PROJECT_ROOT / "data" / "all_images" / os.path.basename(fname)
            
        if not p.exists(): continue
            
        img = cv2.imread(str(p))
        if img is None: continue
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        # TILED INFERENCE
        prob_map = predict_2x2_tiled(model, img)
        
        # Instance Logic (Same as before)
        binary_mask = (prob_map > THRESHOLD).astype(np.uint8)
        labeled, num_features = ndimage.label(binary_mask)
        
        for i in range(1, num_features + 1):
            component = (labeled == i).astype(np.uint8)
            area = component.sum()
            if area < MIN_AREA: continue
            score = float(prob_map[component > 0].mean())
            lines = compute_line_params(component)
            if not lines: lines = [0.0, 0.0]
            
            rle = mask_util.encode(np.asfortranarray(component))
            rle['counts'] = rle['counts'].decode('utf-8')
            y, x = np.where(component > 0)
            bbox = [float(x.min()), float(y.min()), float(x.max()-x.min()+1), float(y.max()-y.min()+1)]
            
            predictions.append({
                "image_id": img_id,
                "category_id": 0,
                "bbox": bbox,
                "segmentation": rle,
                "score": score,
                "lines": lines,
                "area": float(area)
            })
            
    print(f"Saving {len(predictions)} tiled predictions to {OUTPUT_JSON}...")
    with open(OUTPUT_JSON, 'w') as f:
        json.dump(predictions, f)
    
if __name__ == "__main__":
    main()
