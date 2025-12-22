
import json
import sys

def filter_preds(input_file, output_file, threshold):
    print(f"Loading {input_file}...")
    with open(input_file) as f:
        preds = json.load(f)
    
    initial_count = len(preds)
    filtered = [p for p in preds if p['score'] > threshold]
    final_count = len(filtered)
    
    print(f"Filtered {input_file}: {initial_count} -> {final_count} (Threshold: {threshold})")
    
    with open(output_file, 'w') as f:
        json.dump(filtered, f)

if __name__ == "__main__":
    filter_preds(
        "experiments/train_4K/predictions_with_lines.json",
        "experiments/train_4K/predictions_filtered.json",
        0.75
    )
