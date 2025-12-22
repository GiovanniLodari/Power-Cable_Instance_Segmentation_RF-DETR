
import subprocess
import re
from pathlib import Path

THRESHOLDS = [0.97]
GT_PATH = "data/combined/val_combined.json"

print(f"{'Threshold':<10} | {'LDS':<10} | {'AP@50':<10} | {'AR@50':<10}")
print("-" * 50)

for t in THRESHOLDS:
    pred_path = f"models/predictions_v2_{t}.json"
    
    cmd = ["wsl_venv/bin/python", "src/run_lds_eval.py", GT_PATH, pred_path]
    try:
        # Run eval
        result = subprocess.run(cmd, capture_output=True, text=True)
        output = result.stdout
        
        # Parse LDS
        lds_match = re.search(r"LDS = (\d+\.\d+)", output)
        if lds_match:
            lds = float(lds_match.group(1))
        else:
            lds = 0.0
            
        # Parse AP
        ap_match = re.search(r"AP@50: (\d+\.\d+)", output)
        ap = float(ap_match.group(1)) if ap_match else 0.0
        
        # Parse AR
        ar_match = re.search(r"AR@50: (\d+\.\d+)", output)
        ar = float(ar_match.group(1)) if ar_match else 0.0

        print(f"{t:<10} | {lds:.4f}     | {ap:.4f}     | {ar:.4f}")
        
    except Exception as e:
        print(f"{t:<10} | ERROR: {e}")
