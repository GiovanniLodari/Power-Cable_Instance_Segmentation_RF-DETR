
import os
import json
import numpy as np
import glob
from pycocotools import mask as mask_utils

def convert_labelme_to_coco(source_dir, output_file):
    print(f"Converting {source_dir} to {output_file}...")
    
    files = sorted(glob.glob(os.path.join(source_dir, "*.json")))
    files = [f for f in files if not f.endswith("train.json") and not f.endswith("test.json")] # Avoid self-referencing if re-running
    
    images = []
    annotations = []
    categories = [{"id": 0, "name": "cable"}]
    category_map = {"cable": 0}
    
    ann_id_count = 1
    
    for i, json_file in enumerate(files):
        with open(json_file, 'r') as f:
            data = json.load(f)
            
        filename = os.path.basename(json_file).replace(".json", ".jpg") # Assuming jpg matches
        
        # Check if jpg exists?
        img_path = os.path.join(source_dir, filename)
        if not os.path.exists(img_path):
            # Try png?
            filename = filename.replace(".jpg", ".png")
            img_path = os.path.join(source_dir, filename)
            if not os.path.exists(img_path):
                 print(f"Warning: Image for {json_file} not found. Skipping.")
                 continue

        height = data.get("imageHeight")
        width = data.get("imageWidth")
        
        image_info = {
            "id": i + 1, # 1-based Image ID
            "file_name": filename,
            "width": width,
            "height": height
        }
        images.append(image_info)
        
        for shape in data.get("shapes", []):
            label = shape.get("label")
            if label not in category_map:
                continue # Skip non-cable
            
            points = shape.get("points")
            if not points or len(points) < 3:
                continue # Skip bad polygons
                
            # Flatten points [x1, y1, x2, y2...]
            poly_flat = [p for point in points for p in point]
            
            # Helper for Area/BBox using pycocotools
            # pycocotools expects list of lists for polygon [[x,y...], [x,y...]] (for holes, but here 1 shape)
            rles = mask_utils.frPyObjects([poly_flat], height, width)
            area = mask_utils.area(rles)[0]
            bbox = mask_utils.toBbox(rles)[0].tolist() # [x, y, w, h]
            
            ann = {
                "id": ann_id_count,
                "image_id": image_info["id"],
                "category_id": category_map[label],
                "segmentation": [poly_flat],
                "area": float(area),
                "bbox": bbox,
                "iscrowd": 0
            }
            annotations.append(ann)
            ann_id_count += 1
            
    coco_output = {
        "images": images,
        "annotations": annotations,
        "categories": categories
    }
    
    with open(output_file, 'w') as f:
        json.dump(coco_output, f)
        
    print(f"Done. Images: {len(images)}, Annotations: {len(annotations)}")

if __name__ == "__main__":
    convert_labelme_to_coco("data/train_original_size", "data/train_original_size/train.json")
    convert_labelme_to_coco("data/test_original_size", "data/test_original_size/test.json")
