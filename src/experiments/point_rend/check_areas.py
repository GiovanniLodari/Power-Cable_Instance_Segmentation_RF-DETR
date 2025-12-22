
import json
import numpy as np
import pycocotools.mask as mask_util

def check_areas(json_file):
    with open(json_file, 'r') as f:
        preds = json.load(f)
    
    print(f"Total predictions: {len(preds)}")
    
    for p in preds[:5]:
        rle = p['segmentation']
        mask = mask_util.decode(rle)
        print(f"Score: {p['score']:.4f}, Mask Shape: {mask.shape}, Mask Area: {np.sum(mask)}")
        
    high_score_zero_area = 0
    for p in preds:
        if p['score'] > 0.5:
            rle = p['segmentation']
            mask = mask_util.decode(rle)
            if np.sum(mask) == 0:
                high_score_zero_area += 1
                
    print(f"High Score (>0.5) Zero Area Masks: {high_score_zero_area}")

if __name__ == "__main__":
    check_areas("experiments/point_rend/predictions_test_499.json")
