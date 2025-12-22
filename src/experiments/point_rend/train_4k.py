
import os
import copy
import torch
import logging
import numpy as np
import detectron2.utils.comm as comm
from detectron2.checkpoint import DetectionCheckpointer
from detectron2.config import get_cfg
from detectron2.engine import DefaultTrainer, default_argument_parser, default_setup, launch
from detectron2.data import build_detection_train_loader, DatasetMapper, detection_utils as utils
from detectron2.data import transforms as T
from detectron2.projects.point_rend import add_pointrend_config
import detectron2.projects.point_rend # Register PointRend components
from detectron2.data.datasets import register_coco_instances

# Register 4K Datasets
register_coco_instances("ttpla_train_4k", {}, "data/train_original_size/train.json", "data/train_original_size")
register_coco_instances("ttpla_test_4k", {}, "data/test_original_size/test.json", "data/test_original_size")

class TrainMapper4K(DatasetMapper):
    def __init__(self, cfg, is_train=True):
        self.is_train = is_train
        self.image_format = cfg.INPUT.FORMAT
        self.crop_size = (600, 600) 
        
        # We define AugmentationList dynamically in __call__ based on GT
        self.aug = [] # Unused placeholder

    def __call__(self, dataset_dict):
        dataset_dict = copy.deepcopy(dataset_dict)
        image = utils.read_image(dataset_dict["file_name"], format=self.image_format)
        utils.check_image_size(dataset_dict, image)

        # SMART CROP STRATEGY: Center crop around a random annotation
        aug_input = T.AugInput(image)
        crop_tfm = None
        
        # If we have annotations, pick one to focus on
        if "annotations" in dataset_dict and len(dataset_dict["annotations"]) > 0:
            import random
            ann = random.choice(dataset_dict["annotations"])
            bbox = ann["bbox"] # XYWH
            cx = bbox[0] + bbox[2] / 2
            cy = bbox[1] + bbox[3] / 2
            
            cw, ch = self.crop_size
            
            # Jitter
            cx += random.randint(-100, 100)
            cy += random.randint(-100, 100)
            
            # Clamp
            h, w = image.shape[:2]
            x1 = int(max(0, min(w - cw, cx - cw // 2)))
            y1 = int(max(0, min(h - ch, cy - ch // 2)))
            
            crop_tfm = T.CropTransform(x1, y1, cw, ch)
        else:
            # Fallback
            crop_tfm = T.RandomCrop("absolute", self.crop_size).get_transform(aug_input)

        transforms = T.AugmentationList([
             crop_tfm,
             T.RandomFlip()
        ])(aug_input)
        
        image_cropped = aug_input.image
        
        annos = [
            utils.transform_instance_annotations(
                obj, transforms, image_cropped.shape[:2]
            )
            for obj in dataset_dict.pop("annotations")
            if obj.get("iscrowd", 0) == 0
        ]
        
        instances = utils.annotations_to_instances(
            annos, image_cropped.shape[:2], mask_format="bitmask"
        )
        
        # DEBUG VIZ
        if not os.path.exists("debug_4k_crop.jpg") and len(instances) > 0:
             try:
                 import cv2
                 vis_img = image_cropped.copy()
                 for i in range(len(instances)):
                     box = instances[i].gt_boxes.tensor[0].numpy()
                     x1, y1, x2, y2 = map(int, box)
                     cv2.rectangle(vis_img, (x1, y1), (x2, y2), (0, 255, 0), 2)
                 cv2.imwrite("debug_4k_crop.jpg", vis_img[:, :, ::-1])
                 print("Saved debug_4k_crop.jpg")
             except Exception as e:
                 print(f"Viz failed: {e}")

        instances = utils.filter_empty_instances(instances)
        
        # STRICT CHECK
        if len(instances) > 0:
            non_empty_mask_indices = []
            for i in range(len(instances)):
                mask_sum = instances[i].gt_masks.tensor.sum()
                if mask_sum > 0:
                     non_empty_mask_indices.append(i)
            
            if len(non_empty_mask_indices) != len(instances):
                instances = instances[non_empty_mask_indices]
        
        dataset_dict["image"] = torch.as_tensor(np.ascontiguousarray(image_cropped.transpose(2, 0, 1)))
        dataset_dict["instances"] = instances
        return dataset_dict

def setup(args):
    cfg = get_cfg()
    add_pointrend_config(cfg)
    
    from detectron2.model_zoo import model_zoo
    cfg.merge_from_file(model_zoo.get_config_file("COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml"))
    
    # 4K OPTIMIZATIONS
    cfg.MODEL.ANCHOR_GENERATOR.SIZES = [[16, 32, 64, 128, 256]]
    cfg.MODEL.ANCHOR_GENERATOR.ASPECT_RATIOS = [[0.1, 0.2, 0.5, 1.0, 2.0, 5.0, 10.0]]
    
    cfg.MODEL.ROI_HEADS.NAME = "PointRendROIHeads"
    cfg.MODEL.ROI_MASK_HEAD.POINT_HEAD_ON = True
    cfg.MODEL.ROI_MASK_HEAD.NAME = "CoarseMaskHead"
    cfg.MODEL.ROI_MASK_HEAD.POOLER_RESOLUTION = 28
    cfg.MODEL.ROI_MASK_HEAD.OUTPUT_SIDE_RESOLUTION = 56
    cfg.MODEL.POINT_HEAD.TRAIN_NUM_POINTS = 2048
    cfg.MODEL.POINT_HEAD.SUBDIVISION_NUM_POINTS = 8192
    
    cfg.MODEL.RPN.IOU_THRESHOLDS = [0.3, 0.7]
    cfg.MODEL.RPN.POST_NMS_TOPK_TRAIN = 2000
    cfg.MODEL.RPN.POST_NMS_TOPK_TEST = 2000
    
    cfg.INPUT.MASK_FORMAT = "bitmask"
    cfg.DATASETS.TRAIN = ("ttpla_train_4k",)
    cfg.DATASETS.TEST = ("ttpla_test_4k",)
    cfg.DATALOADER.NUM_WORKERS = 1 # Keep 1 for safety
    
    cfg.SOLVER.IMS_PER_BATCH = 1
    cfg.SOLVER.BASE_LR = 0.0005 # Reduce for fine-tuning
    cfg.SOLVER.MAX_ITER = 60000
    cfg.SOLVER.STEPS = (30000, 40000)
    cfg.SOLVER.CHECKPOINT_PERIOD = 1000
    
    cfg.OUTPUT_DIR = "experiments/train_4K/round2"
    # Load previous best weights
    cfg.MODEL.WEIGHTS = os.path.join(os.getcwd(), "experiments/train_4K/model_temp_chunk.pth")

    return cfg

class Trainer(DefaultTrainer):
    @classmethod
    def build_train_loader(cls, cfg):
        try:
            return build_detection_train_loader(cfg, mapper=TrainMapper4K(cfg, is_train=True))
        except Exception as e:
            print(f"Error building loader: {e}")
            raise e
            
    def run_step(self):
        super().run_step()
        # Segmented Training Strategy:
        # Exit every 500 steps to clear memory leaks/fragmentation
        if self.iter > 0 and (self.iter + 1) % 500 == 0:
            
            # PERIODIC CHECKPOINT (Every 2500)
            if (self.iter + 1) % 2500 == 0:
                chk_name = f"model_{self.iter+1:07d}" # detectron adds .pth automatically? No it takes name.
                # checkpointer.save takes 'name' (without extension usually, but saves as name.pth)
                self.checkpointer.save(chk_name)
                print(f"💾 Saved permanent checkpoint: {chk_name}.pth")
                
                # SINCRO: Copy to shared folder
                try:
                    import shutil
                    # D2 saves to output_dir/name.pth
                    src = os.path.join(self.cfg.OUTPUT_DIR, chk_name + ".pth")
                    dst = os.path.join("shared_checkpoints", chk_name + ".pth")
                    shutil.copy(src, dst)
                    print(f"📡 Synced to shared folder: {dst}")
                except Exception as e:
                    print(f"Sync Warning: {e}")
                
            # Save chunk state before exit
            self.checkpointer.save("model_temp_chunk")
            print(f"Reached chunk limit {self.iter}. Exiting for Restart...")
            # We raise an exception to exit the python process uncleanly? 
            # Or just return? DefaultTrainer eats returns?
            # Raising StopIteration stops training cleanly.
            raise StopIteration

def main(args):
    cfg = setup(args)
    os.makedirs(cfg.OUTPUT_DIR, exist_ok=True)
    
    trainer = Trainer(cfg)
    
    # SMART RESUME LOGIC
    # If checkpoint exists in output dir, resume=True (continue training)
    # If not, resume=False (load weights, start fresh iter 0)
    has_checkpoint = os.path.exists(os.path.join(cfg.OUTPUT_DIR, "last_checkpoint"))
    
    if has_checkpoint:
        print("🔄 Found existing checkpoint in round2. Resuming...")
        trainer.resume_or_load(resume=True)
    else:
        print("🆕 No checkpoint found. Starting fresh (Iter 0) with loaded weights.")
        trainer.resume_or_load(resume=False)
    
    # CUSTOM WEIGHT LOADING (Only for fresh start)
    if trainer.iter == 0:
        weight_path = cfg.MODEL.WEIGHTS
        print(f"Loading CUSTOM weights from {weight_path}...")
        
        if weight_path.endswith(".pkl"):
            import pickle
            with open(weight_path, "rb") as f:
                checkpoint = pickle.load(f, encoding='latin1')
        else:
            # Assume .pth
            checkpoint = torch.load(weight_path, map_location="cpu")
            
        state_dict = checkpoint.get("model", checkpoint)
        model_state = trainer.model.state_dict()
        new_state_dict = {}
        
        for k, v in state_dict.items():
            if isinstance(v, np.ndarray):
                v = torch.from_numpy(v)
                
            if "coarse_head" in k:
                 continue
                 
            if k in model_state:
                if v.shape != model_state[k].shape:
                    print(f"Skipping {k}: Shape Mismatch")
                    continue
            new_state_dict[k] = v
        
        trainer.model.load_state_dict(new_state_dict, strict=False)
        print("✅ Custom weights loaded successfully.")
    else:
        print(f"🔄 Resumed from iteration {trainer.iter}.")
    
    try:
        trainer.train()
    except StopIteration:
        print("Chunk Finished.")
        return

if __name__ == "__main__":
    args = default_argument_parser().parse_args()
    launch(
        main,
        args.num_gpus,
        num_machines=args.num_machines,
        machine_rank=args.machine_rank,
        dist_url=args.dist_url,
        args=(args,),
    )
