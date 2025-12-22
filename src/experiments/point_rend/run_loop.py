
import os
import subprocess
import time

MAX_ITER = 60000
CHUNK_SIZE = 500
CMD = ["wsl_venv/bin/python", "src/experiments/point_rend/train_4k.py"]

def get_last_iter():
    # Read metrics.json to find last iter
    try:
        json_path = "experiments/train_4K/round2/metrics.json"
        if not os.path.exists(json_path):
            return 0
        with open(json_path, 'r') as f:
            lines = f.readlines()
            if not lines: return 0
            # Last line
            import json
            last = json.loads(lines[-1])
            return last.get("iteration", 0)
    except:
        return 0

for _ in range(30): # Safety limit
    print("🚀 Starting Chunk...")
    subprocess.run(CMD)
    
    last_iter = get_last_iter()
    print(f"🛑 Chunk Finished. Last Iter: {last_iter}")
    
    if last_iter >= MAX_ITER - 1:
        print("✅ Training Complete.")
        break
    
    time.sleep(2) # Cooldown
