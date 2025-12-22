
import argparse
import json
import os
import cv2
import torch
import numpy as np
from ultralytics import YOLO
from pycocotools import mask as mask_util
from tqdm import tqdm

def main():
    parser = argparse.ArgumentParser(description="Run YOLOv8 inference and generate COCO format predictions.")
    parser.add_argument("--weights", type=str, required=True, help="Path to model weights (.pt)")
    parser.add_argument("--input_json", type=str, required=True, help="Path to input COCO JSON (Validation or Test)")
    parser.add_argument("--output_json", type=str, required=True, help="Path to output predictions JSON")
    parser.add_argument("--conf", type=float, default=0.001, help="Confidence threshold")
    parser.add_argument("--iou", type=float, default=0.6, help="NMS IoU threshold")
    parser.add_argument("--imgsz", type=int, default=640, help="Inference image size")
    parser.add_argument("--max_det", type=int, default=300, help="Maximum detections per image")
    
    args = parser.parse_args()

    # Load Model
    print(f"Loading model from {args.weights}...")
    model = YOLO(args.weights)

    # Load Input JSON
    print(f"Loading input data from {args.input_json}...")
    with open(args.input_json, 'r') as f:
        data = json.load(f)

    # Build Map: Image ID -> File Name
    # And handle paths logic
    images_info = data['images']
    
    predictions = []
    
    print(f"Running inference on {len(images_info)} images with conf={args.conf}, iou={args.iou}...")

    # We need to resolve image paths. Assuming they are relative to the root or exist in data/
    # Heuristic: try path as is, then data/path, then data/all_images/basename
    
    for img_info in tqdm(images_info):
        img_id = img_info['id']
        file_name = img_info['file_name']
        
        # Path Resolution
        if os.path.exists(file_name):
            img_path = file_name
        elif os.path.exists(os.path.join("data", file_name)):
            img_path = os.path.join("data", file_name)
        else:
            # Try flattened
            basename = os.path.basename(file_name)
            flattened = os.path.join("data/all_images", basename)
            if os.path.exists(flattened):
                img_path = flattened
            else:
                # Last resort: try images/val or images/test
                # This depends on dataset structure, but let's try strict first
                # If failed, skip or specific logic
                # For this repo: data/images/val or data/images/test might differ from json 'file_name'
                # Let's try to find it recursively if really needed, but usually 'data/' prefix works
                 pass
                 # If we can't find it, we might error or skip. 
                 # Let's print warning and skip to avoid crash
        
        if not os.path.exists(img_path):
             # Try standard YOLO path reconstruction if json paths are just filenames
             # data/yolo_dataset/images/val/filename ?
             # But let's check one more common location
             if "val" in args.input_json:
                 candidate = os.path.join("data/yolo_dataset/images/val", os.path.basename(file_name))
             elif "test" in args.input_json:
                 candidate = os.path.join("data/yolo_dataset/images/test", os.path.basename(file_name))
             else:
                 candidate = None
             
             if candidate and os.path.exists(candidate):
                 img_path = candidate
             else:
                 print(f"Warning: Image {file_name} not found. Skipping.")
                 continue

        # Run Inference
        results = model.predict(
            source=img_path,
            conf=args.conf,
            iou=args.iou,
            imgsz=args.imgsz,
            max_det=args.max_det,
            retina_masks=True, # High resolution masks
            verbose=False,
            device=0 if torch.cuda.is_available() else 'cpu'
        )
        
        result = results[0]
        
        if result.masks is None:
            continue
            
        # Process Results
        masks = result.masks.data.cpu().numpy() # (N, H, W) - resized to imgsz? 
        # Wait, YOLOv8 masks output might be smaller size. 
        # model.predict returns masks in original image size if retina_masks=True? 
        # Or we need to scale them. 
        # Ultralytics results.masks.data is usually the raw masks.
        # results[0].masks.xy is the segments (polygons).
        # results[0].masks.data is the bitmasks which might be lower res.
        # We want RLE. Ideally robust RLE.
        
        # Better approach: Use result.masks.xy (polygons) and convert to RLE using raw dimensions?
        # Or resize the bitmasks to original shape.
        
        # result.orig_shape is (H, W)
        h_orig, w_orig = result.orig_shape
        
        # masks data is usually smaller. process_mask_native used internally.
        # Let's use the 'xyn' or 'xy' segments to be safe, or just resize the boolean masks.
        # Resizing masks to original size is safer for high quality RLE.
        
        # Actually ultralytics result object has 'masks' which has methods/properties.
        # if retina_masks=True in predict, we get nicer masks?
        # Let's check mask size.
        
        # Iterating detections
        boxes = result.boxes.data.cpu().numpy() # x1, y1, x2, y2, conf, cls
        
        # We need to make sure we parse masks correctly.
        # The safest way is to use the segments if available to create high res masks, 
        # OR just cv2.resize the mask from .data to .orig_shape.
        
        raw_masks = result.masks.data.cpu().numpy() # (N, h_mask, w_mask)
        
        for i, box in enumerate(boxes):
            x1, y1, x2, y2, score, cls = box
            
            # Filter class if needed (we assume only cable=0)
            if int(cls) != 0:
                continue
            
            # Resize mask to original size (if needed, but retina_masks=True should match)
            raw_mask = raw_masks[i]
            if raw_mask.shape != (h_orig, w_orig):
                full_mask = cv2.resize(raw_mask, (w_orig, h_orig), interpolation=cv2.INTER_NEAREST)
            else:
                full_mask = raw_mask
            
            # Ensure strictly binary
            full_mask = (full_mask > 0.5).astype(np.uint8)
            
            # Encode to RLE
            rle = mask_util.encode(np.asfortranarray(full_mask))
            rle['counts'] = rle['counts'].decode('utf-8')

            
            # Bbox
            w_box = x2 - x1
            h_box = y2 - y1
            
            pred = {
                "image_id": img_id,
                "category_id": 0, # Force 0 for 'cable'
                "bbox": [float(x1), float(y1), float(w_box), float(h_box)],
                "score": float(score),
                "segmentation": rle, 
                # Optional: Add line fitting here?
                # For now let's stick to standard detection. 
                # We can run add_lines script after.
            }
            predictions.append(pred)

    print(f"Saving {len(predictions)} predictions to {args.output_json}...")
    with open(args.output_json, 'w') as f:
        json.dump(predictions, f)
    print("Done.")

if __name__ == "__main__":
    main()
