
import json
import numpy as np
import cv2
import pycocotools.mask as mask_util
from tqdm import tqdm
from skimage.morphology import skeletonize

def get_line_params(binary_mask):
    # Skeletonize to get thin line
    skeleton = skeletonize(binary_mask > 0)
    points = np.column_stack(np.where(skeleton))
    
    if len(points) < 2:
        return None
    
    # Fit line using PCA (More robust for simple curves/lines than Hough sometimes)
    # y, x coordinates from np.where
    y = points[:, 0]
    x = points[:, 1]
    
    # Simple linear regression or FitLine
    # cv2.fitLine requires (N, 1, 2)
    pts_cv = points[:, [1, 0]].reshape(-1, 1, 2).astype(np.float32) # XY format
    vx, vy, x0, y0 = cv2.fitLine(pts_cv, cv2.DIST_L2, 0, 0.01, 0.01)
    
    vx, vy, x0, y0 = vx[0], vy[0], x0[0], y0[0]
    
    # Convert to Rho, Theta
    # Ax + By + C = 0
    # Slope m = vy/vx
    # Theta = atan2(vy, vx) + pi/2?
    # Standard Hough form: x cos theta + y sin theta = rho
    
    theta = np.arctan2(vy, vx) + np.pi/2
    rho = x0 * np.cos(theta) + y0 * np.sin(theta)
    
    # Normalize theta to [0, pi]
    if theta < 0:
        theta += np.pi
        rho = -rho
        
    return float(rho), float(theta)

def process_file(input_path, output_path):
    print(f"Loading {input_path}...")
    with open(input_path) as f:
        preds = json.load(f)
        
    print("Extracting lines from masks...")
    for p in tqdm(preds):
        rle = p['segmentation']
        mask = mask_util.decode(rle)
        
        line_params = get_line_params(mask)
        
        if line_params:
            p['lines'] = list(line_params)
        else:
            # Fallback: Use BBox center? Or skip?
            # If no skeleton, maybe mask is empty or blob.
            # Skip lines field.
            pass
            
    print(f"Saving to {output_path}...")
    with open(output_path, 'w') as f:
        json.dump(preds, f)

if __name__ == "__main__":
    process_file(
        'experiments/train_4K/predictions_ultra_safe.json',
        'experiments/train_4K/predictions_with_lines.json'
    )
