
import os
import sys
import json
import random
import cv2
import numpy as np
import torch
from pathlib import Path
from detectron2.config import get_cfg
from detectron2.engine import DefaultTrainer, default_argument_parser, default_setup, launch
from detectron2.data import DatasetCatalog, MetadataCatalog
from detectron2.projects.point_rend import add_pointrend_config
from detectron2 import model_zoo
from detectron2.checkpoint import DetectionCheckpointer

# --- DATASET LOADING (Adaped from train_resnet.py) ---
def load_ttpla_dataset(root_dir: str, split: str = 'train', val_split: float = 0.2):
    root_path = Path(root_dir)
    if split in ['train', 'val']:
        json_file = root_path / 'train' / 'train.json'
        img_root = root_path / 'train'
    else: 
        json_file = root_path / 'test' / 'test.json'
        img_root = root_path / 'test'

    print(f"Loading dataset from {json_file}...")
    with open(json_file, 'r') as f:
        coco_data = json.load(f)
    
    # Sort for deterministic split
    all_images = sorted(coco_data['images'], key=lambda x: x['id'])
    
    if split in ['train', 'val']:
        num_val = int(len(all_images) * val_split)
        images = all_images[:num_val] if split == 'val' else all_images[num_val:]
    else:
        images = all_images
    
    # Fast lookup
    valid_img_ids = {img['id'] for img in images}
    img_to_anns = {}
    for ann in coco_data['annotations']:
        if ann['image_id'] in valid_img_ids:
            img_to_anns.setdefault(ann['image_id'], []).append(ann)
    
    dataset_dicts = []
    for img_info in images:
        record = {
            "file_name": str(img_root / img_info['file_name']),
            "image_id": img_info['id'],
            "height": img_info['height'],
            "width": img_info['width']
        }
        
        objs = []
        for ann in img_to_anns.get(img_info['id'], []):
            x, y, w, h = ann['bbox']
            # Basic cleaning
            if w <= 1 or h <= 1:
                continue
                
            obj = {
                "bbox": [x, y, w, h],
                "bbox_mode": 1, # BoxMode.XYWH_ABS
                "segmentation": ann['segmentation'],
                "category_id": 0, # Force class 0 (Cable)
            }
            objs.append(obj)
            
        if len(objs) > 0:
            record["annotations"] = objs
            dataset_dicts.append(record)
            
    print(f"Loaded {len(dataset_dicts)} images for split '{split}'")
    return dataset_dicts

# --- REGISTRATION ---
def register_datasets(root_dir="data"):
    for d in ["train", "val", "test"]:
        DatasetCatalog.register(f"ttpla_{d}", lambda d=d: load_ttpla_dataset(root_dir, split=d))
        MetadataCatalog.get(f"ttpla_{d}").set(thing_classes=["cable"])

# --- CONFIG ---
def setup(args):
    cfg = get_cfg()
    add_pointrend_config(cfg)
    
    # Load Base Config from Model Zoo (Standard Mask R-CNN)
    cfg.merge_from_file(model_zoo.get_config_file("COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml"))
    
    # PointRend Architecture Injection
    cfg.MODEL.ROI_HEADS.NAME = "PointRendROIHeads"
    cfg.MODEL.ROI_MASK_HEAD.NAME = "CoarseMaskHead" # Essential for PointRend
    cfg.MODEL.ROI_MASK_HEAD.POINT_HEAD_ON = True # CRITICAL: Enable Point Head!
    cfg.MODEL.ROI_MASK_HEAD.POOLER_RESOLUTION = 28 # Increased from 14 to capture thin cables
    cfg.MODEL.ROI_MASK_HEAD.OUTPUT_SIDE_RESOLUTION = 56 # Increased from 7 to 56 to avoid aliasing
    
    cfg.MODEL.POINT_HEAD.NAME = "StandardPointHead"
    cfg.MODEL.POINT_HEAD.FC_DIM = 256
    cfg.MODEL.POINT_HEAD.NUM_CLASSES = 1
    cfg.MODEL.POINT_HEAD.IN_FEATURES = ["p2", "p3", "p4", "p5"]
    
    # Weights
    cfg.MODEL.WEIGHTS = os.path.join(os.getcwd(), "experiments/point_rend/weights/model_final_edd263.pkl")
    
    # Datasets
    cfg.DATASETS.TRAIN = ("ttpla_train",)
    cfg.DATASETS.TEST = ("ttpla_val",)
    
    # Workers
    cfg.DATALOADER.NUM_WORKERS = 4
    
    # Input Resolution (User Request: 700x700 Native)
    cfg.INPUT.MASK_FORMAT = "bitmask" # Required for PointRend
    cfg.INPUT.MIN_SIZE_TRAIN = (700,)
    cfg.INPUT.MAX_SIZE_TRAIN = 700
    cfg.INPUT.MIN_SIZE_TEST = 700
    cfg.INPUT.MAX_SIZE_TEST = 700
    cfg.INPUT.CROP.ENABLED = False # No cropping, keep context
    
    # Solver (Optimized for Stability)
    cfg.SOLVER.IMS_PER_BATCH = 2 
    cfg.SOLVER.BASE_LR = 0.0025
    cfg.SOLVER.MAX_ITER = 6000
    cfg.SOLVER.STEPS = (4000, 5000)
    cfg.SOLVER.CHECKPOINT_PERIOD = 500
    
    # Output
    cfg.OUTPUT_DIR = "experiments/point_rend/output"
    os.makedirs(cfg.OUTPUT_DIR, exist_ok=True)
    
    # --- TURBO OPTIMIZATIONS (BALANCED) ---
    
    # 1. Extreme Anchors
    cfg.MODEL.ANCHOR_GENERATOR.ASPECT_RATIOS = [[0.1, 0.2, 0.5, 1.0, 2.0, 5.0, 10.0]]
    cfg.MODEL.ANCHOR_GENERATOR.SIZES = [[16, 32, 64, 128, 256]]
    
    # 2. Point Head (Lighter Training, Heavy Inference)
    cfg.MODEL.POINT_HEAD.TRAIN_NUM_POINTS = 2048       # Reduced from 4096 to save GPU
    cfg.MODEL.POINT_HEAD.SUBDIVISION_NUM_POINTS = 8192 # Keep high for inference
    cfg.MODEL.POINT_HEAD.SUBDIVISION_STEPS = 5
    
    # 3. ROI Heads
    cfg.MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE = 256 # Reduced from 512 for memory
    cfg.MODEL.ROI_HEADS.NUM_CLASSES = 1
    
    return cfg

class Trainer(DefaultTrainer):
    # Minimal overrides if needed
    pass

def main(args):
    register_datasets()
    cfg = setup(args)
    
    if args.eval_only:
        model = Trainer.build_model(cfg)
        checkpointer = DetectionCheckpointer(model, save_dir=cfg.OUTPUT_DIR)
        checkpointer.resume_or_load(cfg.MODEL.WEIGHTS, resume=args.resume)
        res = Trainer.test(cfg, model)
        return res

    trainer = Trainer(cfg) 
    # Manual Weight Loading to handle Shape Mismatches (due to Resolution change)
    # trainer.resume_or_load(resume=False) <- Replaced by below logic
    
    checkpointer = DetectionCheckpointer(trainer.model, save_dir=cfg.OUTPUT_DIR)
    
    if args.resume:
        # If resuming, let it crash if mismatch (shouldn't happen on same run)
        trainer.resume_or_load(resume=True)
    else:
        # If starting new, load weights but filter mismatches
        weight_path = cfg.MODEL.WEIGHTS
        if weight_path:
            # Download if URL
            from detectron2.utils.file_io import PathManager
            weight_path = PathManager.get_local_path(weight_path)
            
            print(f"Loading weights from {weight_path} with loose matching...")
            import pickle
            with open(weight_path, "rb") as f:
                checkpoint = pickle.load(f, encoding='latin1') # 'latin1' for legacy python2 compatibility if needed, safer
            # checkpoint = torch.load(weight_path, map_location="cpu")
            state_dict = checkpoint.get("model", checkpoint)
            
            model_state = trainer.model.state_dict()
            new_state_dict = {}
            import numpy as np
            for k, v in state_dict.items():
                if isinstance(v, np.ndarray):
                    v = torch.from_numpy(v)
                
                if "coarse_head" in k:
                    print(f"⚠️ Explicitly Skipping {k} (Resetting Coarse Head)")
                    continue
                
                if k in model_state:
                    if v.shape != model_state[k].shape:
                        print(f"⚠️ Skipping {k}: Shape Mismatch {v.shape} vs {model_state[k].shape} (Expected due to config change)")
                        continue
                new_state_dict[k] = v
            
            # Load filtered state dict
            trainer.model.load_state_dict(new_state_dict, strict=False)
            print("Weights loaded successfully (with skipped layers).")
    return trainer.train()

if __name__ == "__main__":
    args = default_argument_parser().parse_args()
    print("Command Line Args:", args)
    launch(
        main,
        num_gpus_per_machine=1,
        num_machines=1,
        machine_rank=0,
        dist_url="auto",
        args=(args,),
    )
