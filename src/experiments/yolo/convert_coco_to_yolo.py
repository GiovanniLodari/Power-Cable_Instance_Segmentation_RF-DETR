
import json
import os
import shutil
import numpy as np
from tqdm import tqdm
from pathlib import Path

def convert_coco_to_yolo_seg(json_path, image_dir, output_dir, split_name):
    # Create directories
    labels_dir = os.path.join(output_dir, 'labels', split_name)
    images_dir = os.path.join(output_dir, 'images', split_name)
    os.makedirs(labels_dir, exist_ok=True)
    os.makedirs(images_dir, exist_ok=True)
    
    with open(json_path, 'r') as f:
        data = json.load(f)
        
    # Map image_id to file_name
    images = {img['id']: img for img in data['images']}
    
    # Process annotations
    print(f"Converting {split_name}...")
    for img_id, img_info in tqdm(images.items()):
        file_name = img_info['file_name']
        img_w = img_info['width']
        img_h = img_info['height']
        
        # Source image path (Handle potential subdirectory structure)
        # Search in multiple potential directories
        potential_dirs = [
            image_dir, 
            "data/all_images",
            "data/train",
            "data/val",
            "data/test"
        ]
        
        src_path = None
        for p_dir in potential_dirs:
            candidate = os.path.join(p_dir, file_name)
            if os.path.exists(candidate):
                src_path = candidate
                break
            # Try basename
            candidate_base = os.path.join(p_dir, os.path.basename(file_name))
            if os.path.exists(candidate_base):
                src_path = candidate_base
                break
        
        if not src_path:
            # print(f"Warning: Image {file_name} not found. Skipping.")
            continue
            
        # Destination image path
        dst_path = os.path.join(images_dir, os.path.basename(file_name))
        if not os.path.exists(dst_path):
            shutil.copy(src_path, dst_path)
            
        # Get annotations for this image
        anns = [ann for ann in data['annotations'] if ann['image_id'] == img_id]
        
        label_content = []
        for ann in anns:
            cat_id = ann['category_id'] # Should be 0 for cable
            # Remap category_id if needed (COCO usually 1-based, YOLO 0-based)
            # Check user data. Previously checked: category_id: 0. So keep 0.
            
            # Segmentation
            if 'segmentation' in ann and ann['segmentation']:
                seg = ann['segmentation'][0] # List of Polygons
                # Polygon format: [x1, y1, x2, y2, ...]
                # Normalize to 0-1
                poly = np.array(seg).reshape(-1, 2)
                poly[:, 0] /= img_w
                poly[:, 1] /= img_h
                
                # Check bounds
                poly = np.clip(poly, 0, 1)
                
                # Format: class x1 y1 x2 y2 ...
                # poly is (N, 2). Iterate over rows.
                line = [str(cat_id)] + [f"{row[0]:.6f} {row[1]:.6f}" for row in poly]
                label_content.append(" ".join(line))
        
        # Write label file
        if label_content:
            label_file = os.path.splitext(os.path.basename(file_name))[0] + ".txt"
            with open(os.path.join(labels_dir, label_file), 'w') as f:
                f.write("\n".join(label_content))

def create_yaml(output_dir):
    yaml_content = f"""
path: {os.path.abspath(output_dir)}
train: images/train
val: images/val
test: images/test

names:
  0: cable
"""
    with open(os.path.join(output_dir, 'data.yaml'), 'w') as f:
        f.write(yaml_content)

if __name__ == "__main__":
    # Config
    # JSON Paths from previous tasks
    TRAIN_JSON = "data/combined/train_combined.json"
    VAL_JSON = "data/combined/val_combined.json"
    # TEST_JSON? Usually we use test set for final eval.
    TEST_JSON = "data/test/test.json" 
    
    # Image Dirs
    # data/train/ contains training images
    # data/val/ contains val images
    # data/test/ contains test images
    
    OUTPUT_DIR = "data/yolo_dataset"
    
    # Convert Train
    # Note: data/combined/train_combined.json images might be in data/train
    convert_coco_to_yolo_seg(TRAIN_JSON, "data/train", OUTPUT_DIR, "train")
    
    # Convert Val
    convert_coco_to_yolo_seg(VAL_JSON, "data/val", OUTPUT_DIR, "val")
    
    # Convert Test
    # Check if test.json exists
    if os.path.exists(TEST_JSON):
        convert_coco_to_yolo_seg(TEST_JSON, "data/test", OUTPUT_DIR, "test")
        
    # Create YAML
    create_yaml(OUTPUT_DIR)
    print("Conversion Complete.")
