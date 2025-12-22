# train_m2f_robust.py
# Strategy H: Robust Retraining with Safe Jittering & Hyper-Sampling
# Goal: Fix thin cable recall and scale invariance to beat LDS 2.9

import os
import torch
import detectron2.utils.comm as comm
from detectron2.checkpoint import DetectionCheckpointer
from detectron2.config import get_cfg
from detectron2.engine import DefaultTrainer, default_argument_parser, default_setup, launch
from detectron2.projects.deeplab import add_deeplab_config
from train_resnet import (
    load_ttpla_dataset,
    BackgroundReplacementMapper, 
    SegTrainer, 
    setup_seg_config, 
    add_maskformer2_config
)
from backbone_unet import UNetPPBackbone
from detectron2.data import DatasetCatalog, MetadataCatalog

def setup(args):
    cfg = get_cfg()
    add_deeplab_config(cfg)
    add_maskformer2_config(cfg)
    
    # 1. Base Config
    cfg.merge_from_file("Mask2Former/configs/coco/instance-segmentation/maskformer2_R50_bs16_50ep.yaml")
    
    # 2. Dataset Registration
    from detectron2.data.datasets import register_coco_instances
    # Use existing combined strategy
    register_coco_instances("ttpla_combined_train", {}, "data/combined/train_combined.json", "data/all_images")
    register_coco_instances("ttpla_combined_val", {}, "data/combined/val_combined.json", "data/all_images")

# Config
    # Script in src/experiments/m2f_unetbb/
    # Root is ../../../
    PROJECT_ROOT = Path(__file__).resolve().parents[3]
    DATA_ROOT = PROJECT_ROOT / "data"
    # Strategy H was Output Dir
    OUTPUT_DIR = PROJECT_ROOT / "models" / "output_m2f_robust_strat_h"

    # Add directory to sys.path to allow local imports if needed
    sys.path.append(str(Path(__file__).parent))
    
    from backbone_unet import UNetPPBackbone # Import from local file
    cfg = setup_seg_config(str(OUTPUT_DIR)) # Uses custom segmentation setup
    cfg.OUTPUT_DIR = str(OUTPUT_DIR)

    # 4. Dataset Assignment
    cfg.DATASETS.TRAIN = ("ttpla_combined_train",)
    cfg.DATASETS.TEST = ("ttpla_combined_val",)

    # 5. Backbone: UNet++ (ResNet50)
    # Check for weights
    custom_weights_path = "models/unetpp_converted.pth"
    if os.path.exists(custom_weights_path):
        print(f"LOADING CUSTOM BACKBONE WEIGHTS: {custom_weights_path}")
        cfg.MODEL.WEIGHTS = custom_weights_path
    else:
        # Fallback to standard ResNet if custom backbone weights missing (should be there)
        print("WARNING: Custom backbone weights not found. Using ImageNet initialization.")
        
    cfg.MODEL.BACKBONE.NAME = "UNetPPBackbone"
    cfg.MODEL.BACKBONE.FREEZE_AT = 0 
    
    # =================================================================
    # STRATEGY H CONFIGURATION (CORE CHANGES)
    # =================================================================
    
    # A. Safe Jittering + Fixed Crop (Resolution Stability)
    # Range [768, 896] -> Crop 768
    # Logic: Model sees slightly different scales (zoom in/out) but ALWAYS receives 768x768 input.
    # This prevents the "Divisible by 32" crashes and OOM spikes.
    cfg.INPUT.MIN_SIZE_TRAIN = (768, 800, 832, 864, 896)
    cfg.INPUT.MAX_SIZE_TRAIN = 2048
    cfg.INPUT.CROP.ENABLED = True
    cfg.INPUT.CROP.TYPE = "absolute"
    cfg.INPUT.CROP.SIZE = (768, 768)
    
    # B. Hyper-Sampling (Focus on Thin Lines)
    # Standard is 12544 points. We double it to ~25k to ensure thin cables are sampled.
    cfg.MODEL.MASK_FORMER.POINT_TRAIN_SAMPLED_POINTS = 24576
    
    # Importance Sample Ratio: 0.9 (90% Hard/Boundary pixels, 10% Uniform)
    # Default is 0.75. Increasing this forces the model to fix errors on cable boundaries.
    cfg.MODEL.MASK_FORMER.IMPORTANCE_SAMPLE_RATIO = 0.90
    
    # C. Loss Balancing & Hyper-Parameters
    # Boosting Dice Loss (5.0 -> 6.0) - A moderate boost for structure.
    # We rely more on Hyper-Sampling (24k points) to find the cables.
    cfg.MODEL.MASK_FORMER.DICE_WEIGHT = 6.0
    cfg.MODEL.MASK_FORMER.MASK_WEIGHT = 6.0
    
    # D. Solver / Batching
    cfg.SOLVER.IMS_PER_BATCH = 1 # Physical batch size per GPU
    cfg.SOLVER.WARMUP_ITERS = 2000
    cfg.SOLVER.MAX_ITER = 60000
    cfg.SOLVER.CHECKPOINT_PERIOD = 2000
    cfg.SOLVER.STEPS = (45000, 55000)
    
    # Feature Keys (UNetPP specific)
    cfg.MODEL.SEM_SEG_HEAD.IN_FEATURES = ["p3", "p4", "p5"]
    cfg.MODEL.MASK_FORMER.IN_FEATURES = ["p3", "p4", "p5"]
    cfg.MODEL.SEM_SEG_HEAD.DEFORMABLE_TRANSFORMER_ENCODER_IN_FEATURES = ["p3", "p4", "p5"]
    cfg.MODEL.RESNETS.DEPTH = 50 
    
    cfg.INPUT.RANDOM_FLIP = "horizontal"
    cfg.INPUT.SIZE_DIVISIBILITY = 32 # Avoid size mismatch errors  # E. Rotation Augmentation (Crucial for cables)
    # Detectron2's "Augmentation" list is usually handled in the Mapper. 
    # But for Mask2Former, we need to check if it supports config-based rotation.
    # Standard M2F config doesn't have INPUT.ROTATION_RANGE.
    # We must inject it into the Mapper if we want it. 
    # However, 'train_resnet.py' uses a custom mapper 'BackgroundReplacementMapper' or standard DatasetMapper?
    # It uses 'BackgroundReplacementMapper' if background replacement is on, else...
    # Let's stick to safe config first. If M2F doesn't support it OOB, we skip to avoid crashing custom mappers.
    # Actually, let's verify if we can easily add it. 
    # Given the complexity of custom mappers, sticking to Scale+Crop+Flip is safer for "Robust" start.
    # User asked "Can we improve?". I will add it if I'm sure it won't break the mapper.
    # The 'train_resnet.py' doesn't seem to expose Rotation easily.
    # I will SKIP Rotation to ensure stability as requested ("Safe").
    cfg.INPUT.SIZE_DIVISIBILITY = 32
    
    cfg.freeze()
    default_setup(cfg, args)
    return cfg

def main(args):
    cfg = setup(args)
class M2FRobustTrainer(SegTrainer):
    @classmethod
    def build_evaluator(cls, cfg, dataset_name, output_folder=None):
        if output_folder is None:
            output_folder = os.path.join(cfg.OUTPUT_DIR, "inference")
        from detectron2.evaluation import COCOEvaluator
        return COCOEvaluator(dataset_name, output_dir=output_folder)

def main(args):
    cfg = setup(args)
    trainer = M2FRobustTrainer(cfg)
    trainer.resume_or_load(resume=args.resume)
    
    if args.eval_only:
        model = trainer.build_model(cfg)
        DetectionCheckpointer(model, save_dir=cfg.OUTPUT_DIR).resume_or_load(
            cfg.MODEL.WEIGHTS, resume=args.resume
        )
        res = trainer.test(cfg, model)
        return res
        
    return trainer.train()

if __name__ == "__main__":
    parser = default_argument_parser()
    args = parser.parse_args()
    launch(
        main,
        args.num_gpus,
        num_machines=args.num_machines,
        machine_rank=args.machine_rank,
        dist_url=args.dist_url,
        args=(args,),
    )
