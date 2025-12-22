
import os
import shutil
from pathlib import Path

def setup_original_split(split, source_dir, ref_dir, target_dir):
    print(f"Processing split: {split}")
    os.makedirs(target_dir, exist_ok=True)
    
    # Get reference filenames (the 700x700 images)
    ref_files = [f for f in os.listdir(ref_dir) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
    print(f"  Found {len(ref_files)} reference images in {ref_dir}")
    
    match_count = 0
    missing_count = 0
    
    for filename in ref_files:
        basename = os.path.splitext(filename)[0]
        
        # Source paths
        src_img = os.path.join(source_dir, filename)
        src_json = os.path.join(source_dir, f"{basename}.json")
        
        # Target paths
        dst_img = os.path.join(target_dir, filename)
        dst_json = os.path.join(target_dir, f"{basename}.json")
        
        if os.path.exists(src_img):
            shutil.copy2(src_img, dst_img)
            match_count += 1
            
            # Copy JSON if exists (User said "image has associated json")
            if os.path.exists(src_json):
                shutil.copy2(src_json, dst_json)
        else:
            print(f"  ⚠️ Missing original for: {filename}")
            missing_count += 1
            
    print(f"  Matched & Copied: {match_count}")
    print(f"  Missing: {missing_count}")

if __name__ == "__main__":
    source_root = "data/data_original_size"
    
    # Train
    setup_original_split(
        "train",
        source_root,
        "data/train",
        "data/train_original_size"
    )
    
    # Test
    setup_original_split(
        "test", 
        source_root, 
        "data/test", 
        "data/test_original_size"
    )
