
import torch
import argparse
import pickle
from pathlib import Path

def convert_unet_to_m2f_encoder(unet_path, output_path):
    """
    Extracts the encoder weights from a U-Net++ checkpoint and renames them
    to match the Detectron2 ResNet backbone format used by Mask2Former.
    """
    print(f"Loading U-Net++ checkpoint from: {unet_path}")
    checkpoint = torch.load(unet_path, map_location="cpu")
    
    # Handle different checkpoint structures (some wrap in 'model_state_dict')
    if "model_state_dict" in checkpoint:
        state_dict = checkpoint["model_state_dict"]
    else:
        state_dict = checkpoint
        
    new_state_dict = {}
    converted_count = 0
    ignored_count = 0
    
    # Mapping U-Net++ (SMP ResNet) -> Detectron2 ResNet Backbone
    # SMP: encoder.layer1.0.conv1.weight
    # D2:  res2.0.conv1.weight
    
    # SMP structure for ResNet:
    # encoder.conv1 -> stem.conv1
    # encoder.bn1   -> stem.bn1
    # encoder.layer1 -> res2
    # encoder.layer2 -> res3
    # encoder.layer3 -> res4
    # encoder.layer4 -> res5
    
    for key, value in state_dict.items():
        if not key.startswith("encoder."):
            ignored_count += 1
            continue
            
        # Strip 'encoder.' prefix
        suffix = key[len("encoder."):]
        
        new_key = None
        
        # Stem layers
        if suffix.startswith("conv1"):
            new_key = suffix.replace("conv1", "stem.conv1")
        elif suffix.startswith("bn1"):
            new_key = suffix.replace("bn1", "stem.bn1")
        
        # ResNet stages
        elif suffix.startswith("layer1"):
            new_key = suffix.replace("layer1", "res2")
        elif suffix.startswith("layer2"):
            new_key = suffix.replace("layer2", "res3")
        elif suffix.startswith("layer3"):
            new_key = suffix.replace("layer3", "res4")
        elif suffix.startswith("layer4"):
            new_key = suffix.replace("layer4", "res5")
            
        if new_key:
            new_state_dict[new_key] = value.numpy() # D2 expects numpy arrays for .pkl loading usually, checking...
            # Actually D2 Checkpointer can load .pth or .pkl. Standard D2 models are .pkl with {"model": dict}.
            # But let's keep it simple: D2 matches keys.
            # Mask2Former loads backbone via `backbone.bottom_up` prefix usually?
            # Let's check how we load it. If we load as "pretrained weights", D2 expects "stem.conv1.weight" etc.
            # However, if we utilize `convert-torchvision-to-d2` logic, we might need adjustments.
            # BUT: SMP ResNet is remarkably standard.
            
            # Re-convert to torch tensor? D2 `DetectionCheckpointer` handles both.
            # Let's keep as tensor for .pth, or numpy for .pkl. 
            # D2 standard .pkl format: {"model": {key: numpy_array}, "__author__": ...}
            new_state_dict[new_key] = value
            converted_count += 1
        else:
            print(f"Skipping unknown encoder key: {key}")

    print(f"\nConversion Summary:")
    print(f"  - Converted {converted_count} keys (Encoder)")
    print(f"  - Ignored {ignored_count} keys (Decoder/Head)")
    
    # Prepare D2-style checkpoint dictionary
    d2_checkpoint = {"model": new_state_dict}
    
    print(f"Saving converted weights to: {output_path}")
    with open(output_path, "wb") as f:
        torch.save(d2_checkpoint, f)
        
    print("Done!")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Convert U-Net++ Encoder to Mask2Former Backbone")
    parser.add_argument("--unet", type=str, required=True, help="Path to U-Net++ .pth checkpoint")
    parser.add_argument("--output", type=str, required=True, help="Path for output .pth file")
    
    args = parser.parse_args()
    convert_unet_to_m2f_encoder(args.unet, args.output)
