
import torch
import argparse
import os

def convert_weights(input_path, output_path):
    print(f"Loading weights from {input_path}...")
    checkpoint = torch.load(input_path, map_location="cpu")
    
    # Detect nested state dict
    if "model_state_dict" in checkpoint:
        print("Found 'model_state_dict' key.")
        state_dict = checkpoint["model_state_dict"]
    elif "state_dict" in checkpoint:
        print("Found 'state_dict' key.")
        state_dict = checkpoint["state_dict"]
    elif "model" in checkpoint:
        print("Found 'model' key.")
        state_dict = checkpoint["model"]
    else:
        print("Assuming flat state dict.")
        state_dict = checkpoint
        
    print(f"Original keys example: {list(state_dict.keys())[:3]}")
    
    new_state_dict = {}
    renamed_count = 0
    skipped_count = 0
    
    for k, v in state_dict.items():
        # Clean up potential DataParallel prefix
        if k.startswith("module."):
            k = k[7:]
            
        # Filter out segmentation head (we use Mask2Former head now)
        if "segmentation_head" in k or "head" in k or "classification_head" in k:
            skipped_count += 1
            continue
            
        # Map Encoder/Decoder to backbone.unet structure
        # The class UNetPPBackbone has self.unet = smp.UnetPlusPlus(...)
        # And it sits inside model.backbone
        # So effective path is: backbone.unet.[encoder|decoder]...
        
        if k.startswith("encoder") or k.startswith("decoder"):
            new_key = f"backbone.unet.{k}"
            new_state_dict[new_key] = v
            renamed_count += 1
        else:
            # Other keys (maybe unknown), skip or keep? 
            # Safest is to skip unless sure
            skipped_count += 1
            
    print(f"Conversion Complete: {renamed_count} keys renamed, {skipped_count} keys skipped (heads/other).")
    print(f"Converted keys example: {list(new_state_dict.keys())[:3]}")
    
    # Save simply as the state dict
    torch.save(new_state_dict, output_path)
    print(f"Saved converted weights to {output_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Convert standard SMP U-Net++ weights to Detectron2 UNetPPBackbone format.")
    parser.add_argument("--input", required=True, help="Path to original .pth checkpoint")
    parser.add_argument("--output", required=True, help="Path to output .pth file")
    
    args = parser.parse_args()
    convert_weights(args.input, args.output)
