
import json
import os
import subprocess
from tqdm import tqdm

WEIGHTS = "experiments/train_4K/model_temp_chunk.pth"
TEST_JSON = "data/test_original_size/test.json"
IMG_ROOT = "data/test_original_size"
FINAL_OUTPUT = "experiments/train_4K/predictions_4k_robust.json"
TEMP_DIR = "experiments/train_4K/temp_preds"

os.makedirs(TEMP_DIR, exist_ok=True)

with open(TEST_JSON) as f:
    data = json.load(f)

all_preds = []

print(f"Robust Inference on {len(data['images'])} images...")

for img in tqdm(data['images']):
    img_path = os.path.join(IMG_ROOT, img['file_name'])
    temp_json = os.path.join(TEMP_DIR, f"{img['id']}.json")
    
    # Check if already done (Resume capability)
    if os.path.exists(temp_json) and os.path.getsize(temp_json) > 0:
        with open(temp_json) as f:
            p = json.load(f)
            # Add image_id and filename
            for item in p:
                item["image_id"] = img['id']
                item["file_name"] = img['file_name']
                all_preds.append(item)
        continue

    # Tiled Inference (Handles 4K internally via tiling)
    success = False
    
    # Enable Expandable Segments
    env = os.environ.copy()
    env["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True,max_split_size_mb:128"
    
    try:
        subprocess.run(
            [
                "wsl_venv/bin/python", 
                "src/experiments/point_rend/infer_single.py",
                "--image_path", img_path,
                "--output_json", temp_json,
                "--weights", WEIGHTS
                # Scale arg ignored by tiling
            ],
            check=True,
            capture_output=True,
            env=env
        )
        success = True
    except subprocess.CalledProcessError as e:
        print(f"❌ Failed Tiling for {img['file_name']}")
        print(e.stderr.decode())

    if success:
        with open(temp_json) as f:
            p = json.load(f)
            for item in p:
                item["image_id"] = img['id']
                item["file_name"] = img['file_name']
                all_preds.append(item)
    else:
        print(f"❌ Failed all methods for {img['file_name']}")

with open(FINAL_OUTPUT, 'w') as f:
    json.dump(all_preds, f)
print(f"Saved {len(all_preds)} predictions to {FINAL_OUTPUT}")
