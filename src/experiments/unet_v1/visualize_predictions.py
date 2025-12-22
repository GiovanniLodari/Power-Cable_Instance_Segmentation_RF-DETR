
import os
import cv2
import torch
import numpy as np
from tqdm import tqdm

from detectron2.config import get_cfg
from detectron2.engine import DefaultPredictor
from detectron2.utils.visualizer import Visualizer, ColorMode
from detectron2.data import DatasetCatalog, MetadataCatalog
from detectron2.projects.deeplab import add_deeplab_config
from mask2former import add_maskformer2_config

# Import specific project modules
from train_resnet import load_ttpla_dataset
# IMPORTANT: Import the backbone so it is registered!
from backbone_unet import UNetPPBackbone 

def setup_config(weights_path):
    cfg = get_cfg()
    add_deeplab_config(cfg)
    add_maskformer2_config(cfg)
    
    # Load the EXACT config used for training
    config_file = "models/output_m2f_unetpp/config.yaml"
    print(f"Loading config from {config_file}...")
    cfg.set_new_allowed(True)
    cfg.merge_from_file(config_file)
    cfg.set_new_allowed(False)
    
    # Overwrite weights path to be sure
    cfg.MODEL.WEIGHTS = weights_path
    cfg.MODEL.DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    
    # Ensure Test Resolution matches
    cfg.INPUT.MIN_SIZE_TEST = 672
    cfg.INPUT.MAX_SIZE_TEST = 1333
    
    return cfg

def main():
    # =============================================================================
    # CONFIGURATION
    # =============================================================================
    # Script in src/experiments/unet_v1/
    PROJECT_ROOT = Path(__file__).resolve().parents[3]
    MODEL_PATH = PROJECT_ROOT / "models" / "unetpp_resnet50" / "checkpoint_epoch_100.pth"
    TEST_JSON = PROJECT_ROOT / "data" / "test" / "test.json"
    output_dir = PROJECT_ROOT / "visualizations" / "unet_v1"
    os.makedirs(output_dir, exist_ok=True)
    
    weights_path = "models/output_m2f_unetpp/model_final.pth" # This line was originally here, keeping it for now as it's used below
    print(f"Loading model from {weights_path}...")
    cfg = setup_config(weights_path)
    predictor = DefaultPredictor(cfg)
    
    # Register Dataset
    dataset_name = "ttpla_val_viz"
    if dataset_name not in DatasetCatalog.list():
        DatasetCatalog.register(dataset_name, lambda: load_ttpla_dataset("data", "val")) # Use 'val' split
        MetadataCatalog.get(dataset_name).set(thing_classes=["cable"])
    
    dataset_dicts = DatasetCatalog.get(dataset_name)
    metadata = MetadataCatalog.get(dataset_name)
    
    print(f"Visualizing {len(dataset_dicts)} images...")
    
    for i, d in enumerate(tqdm(dataset_dicts)):
        img = cv2.imread(d["file_name"])
        visualizer_gt = Visualizer(img[:, :, ::-1], metadata=metadata, scale=1.0)
        visualizer_pred = Visualizer(img[:, :, ::-1], metadata=metadata, scale=1.0)
        
        # 1. Draw Ground Truth
        out_gt = visualizer_gt.draw_dataset_dict(d)
        gt_img = np.ascontiguousarray(out_gt.get_image()[:, :, ::-1])
        
        # 2. Run Inference
        outputs = predictor(img)
        
        # 3. Draw Predictions
        # Filter low confidence - DEBUG: Lowering to 0.05 and printing
        instances = outputs["instances"].to("cpu")
        print(f"Image {i}: Found {len(instances)} raw instances. Scores: {instances.scores}")
        
        # Keep only scores > 0.05 to see everything
        instances = instances[instances.scores > 0.05]
        print(f"Image {i}: Showing {len(instances)} instances > 0.05")
        
        out_pred = visualizer_pred.draw_instance_predictions(instances)
        pred_img = np.ascontiguousarray(out_pred.get_image()[:, :, ::-1])
        
        # 4. Concatenate Side-by-Side
        # Add labels text
        gt_img = cv2.putText(gt_img, "Ground Truth", (30, 60), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 255, 0), 3)
        pred_img = cv2.putText(pred_img, "Prediction (U-Net++)", (30, 60), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 255), 3)
        
        comparison = np.concatenate((gt_img, pred_img), axis=1)
        
        # Save
        save_path = os.path.join(output_dir, f"comparison_nopad_{i}.png")
        cv2.imwrite(save_path, comparison)
        
        if i >= 19: # Limit to first 20 for speed/storage
            break
            
    print(f"Saved visualizations to {output_dir}")

if __name__ == "__main__":
    main()
