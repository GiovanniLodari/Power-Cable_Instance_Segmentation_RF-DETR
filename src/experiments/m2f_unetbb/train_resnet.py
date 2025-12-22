import os
import json
import re
import numpy as np
import sys
import copy
import logging
import warnings
import cv2
import datetime
import asciichartpy
import matplotlib.pyplot as plt
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import random

# Rich imports
from rich.console import Console
from rich.table import Table

# --- 1. SETUP PATHS ---
project_root = Path(__file__).resolve().parent.parent
if str(project_root / "Mask2Former") not in sys.path:
    sys.path.append(str(project_root / "Mask2Former"))

import torch
from torch import nn
from torch.nn import functional as F

# Detectron2
from detectron2.config import get_cfg, configurable
from detectron2.data import DatasetCatalog, MetadataCatalog, build_detection_train_loader
from detectron2.data import transforms as T
from detectron2.data import detection_utils as utils
from detectron2.engine import DefaultTrainer, HookBase
from detectron2.structures import BitMasks, PolygonMasks, Instances
from detectron2.modeling import META_ARCH_REGISTRY, SEM_SEG_HEADS_REGISTRY
from detectron2.utils.events import EventStorage, get_event_storage, CommonMetricPrinter, JSONWriter

# Mask2Former imports
from mask2former.config import add_maskformer2_config
from mask2former.modeling.criterion import SetCriterion
from mask2former.modeling.matcher import HungarianMatcher
from mask2former.modeling.meta_arch.mask_former_head import MaskFormerHead
from mask2former.modeling.transformer_decoder.mask2former_transformer_decoder import MultiScaleMaskedTransformerDecoder
from mask2former.maskformer_model import MaskFormer
from mask2former.maskformer_model import MaskFormer
from mask2former.utils.misc import nested_tensor_from_tensor_list

# --- MONKEY PATCH: Disable JIT for Matcher (Fix Global Alloc Error) ---
from mask2former.modeling.matcher import batch_dice_loss, batch_sigmoid_ce_loss
import mask2former.modeling.matcher as matcher_module

matcher_module.batch_dice_loss_jit = batch_dice_loss
matcher_module.batch_sigmoid_ce_loss_jit = batch_sigmoid_ce_loss
print("WARNING: Monkey-patched Mask2Former matcher to disable JIT (Fixing Global Alloc Error)")
# ----------------------------------------------------------------------

from detectron2.projects.point_rend.point_features import (
    get_uncertain_point_coords_with_randomness,
    point_sample,
)

def calculate_uncertainty(logits):
    """
    We estimate uncerainty as L1 distance between 0.0 and the logit prediction in 'logits' for the
        foreground class in `classes`.
    """
    assert logits.shape[1] == 1
    gt_class_logits = logits.clone()
    return -(torch.abs(gt_class_logits))


warnings.filterwarnings("ignore")
os.environ["PYTHONWARNINGS"] = "ignore"
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True" # Fix: Help fragmentation

logging.getLogger("detectron2").setLevel(logging.WARNING)
logging.getLogger("fvcore").setLevel(logging.WARNING)
logging.getLogger("mask2former").setLevel(logging.WARNING)
logging.getLogger("timm").setLevel(logging.WARNING)

from contextlib import contextmanager

@contextmanager
def suppress_output():
    """
    Un context manager che ridirige stdout e stderr a /dev/null.
    Zittisce completamente qualsiasi libreria (Detectron2, PyTorch, ecc.)
    """
    with open(os.devnull, "w") as devnull:
        old_stdout = sys.stdout
        old_stderr = sys.stderr
        sys.stdout = devnull
        sys.stderr = devnull
        try:
            yield
        finally:
            sys.stdout = old_stdout
            sys.stderr = old_stderr

# =============================================================================
# 2. DATA LOADING & UTILS
# =============================================================================

# --- NEW: Adaptive Scheduler Wrapper ---
class AdaptiveScheduler:
    """
    FIX: Wrapper funzionante per ReduceLROnPlateau
    """
    def __init__(self, optimizer, mode='min', factor=0.5, patience=5, verbose=True, min_lr=1e-7):
        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode=mode, factor=factor, patience=patience, 
            verbose=verbose, min_lr=min_lr, threshold=1e-4
        )
        self.current_lr = optimizer.param_groups[0]['lr']
    
    def step(self):
        # Detectron2 calls this every iteration. We do nothing here.
        pass
    
    def step_metric(self, metric):
        # We call this manually in ValidationLossHook
        old_lr = self.current_lr
        self.scheduler.step(metric)
        self.current_lr = self.scheduler.optimizer.param_groups[0]['lr']
        if old_lr != self.current_lr:
            print(f"\n📉 LR DECAY: {old_lr:.2e} → {self.current_lr:.2e}")
    
    def state_dict(self):
        return self.scheduler.state_dict()
    
    def load_state_dict(self, state_dict):
        self.scheduler.load_state_dict(state_dict)

# --- NEW: Early Stopping Hook ---
class EarlyStoppingHook(HookBase):
    def __init__(self, eval_period, patience=5, min_delta=0.001, verbose=True):
        self.eval_period = eval_period
        self.patience = patience
        self.min_delta = min_delta
        self.verbose = verbose
        self.best_loss = float('inf')
        self.epochs_no_improve = 0

    def after_step(self):
        if (self.trainer.iter + 1) % self.eval_period != 0:
            return

        storage = get_event_storage()
        if "val_total_loss" not in storage.histories():
            return
            
        current_loss = storage.histories()["val_total_loss"].latest()

        if current_loss < self.best_loss - self.min_delta:
            self.best_loss = current_loss
            self.epochs_no_improve = 0
            if self.verbose:
                print(f"\n✅ Validation loss improved to {current_loss:.4f}. Patience reset.")
        else:
            self.epochs_no_improve += 1
            if self.verbose:
                print(f"\n⚠️  Validation loss plateau. Patience: {self.epochs_no_improve}/{self.patience}")
            if self.epochs_no_improve >= self.patience:
                if self.verbose:
                    print("\n🛑 EARLY STOPPING TRIGGERED!")
                # FIX: Raise StopIteration invece di modificare trainer.max_iter
                raise StopIteration

def load_ttpla_dataset(root_dir: str, split: str = 'train', val_split: float = 0.2):
    if split in ['train', 'val']:
        json_file = Path(root_dir) / 'train' / 'train.json'
        img_root = Path(root_dir) / 'train'
    else: 
        json_file = Path(root_dir) / 'test' / 'test.json'
        img_root = Path(root_dir) / 'test'

    print(f"Loading dataset from {json_file}...")
    with open(json_file, 'r') as f:
        coco_data = json.load(f)
    if 'info' not in coco_data:
        coco_data['info'] = {"description": "TTPLA Dataset", "version": "1.0"}
    
    all_images = sorted(coco_data['images'], key=lambda x: x['id'])
    
    if split in ['train', 'val']:
        num_val = int(len(all_images) * val_split)
        images = all_images[:num_val] if split == 'val' else all_images[num_val:]
    else:
        images = all_images
    
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
            x1 = np.clip(x, 0, img_info['width'])
            y1 = np.clip(y, 0, img_info['height'])
            x2 = np.clip(x + w, 0, img_info['width'])
            y2 = np.clip(y + h, 0, img_info['height'])
            
            w_new = x2 - x1
            h_new = y2 - y1
            
            if w_new <= 1 or h_new <= 1:
                continue
            
            polar_points = [0.0, 0.0, 0.0, 0.0]
            has_polar = False
            
            if 'polar_coordinates' in ann and len(ann['polar_coordinates']) > 0:
                pc = ann['polar_coordinates'][0]
                if 'start_point' in pc and 'end_point' in pc:
                    s = pc['start_point']
                    e = pc['end_point']
                    sx = np.clip(s[0], 0, img_info['width'])
                    sy = np.clip(s[1], 0, img_info['height'])
                    ex = np.clip(e[0], 0, img_info['width'])
                    ey = np.clip(e[1], 0, img_info['height'])
                    polar_points = [sx, sy, ex, ey]
                    has_polar = True

            objs.append({
                "bbox": [x1, y1, w_new, h_new],
                "bbox_mode": 1, 
                "segmentation": ann['segmentation'],
                "category_id": 0,
                "polar_points_raw": polar_points,
                "has_polar_valid": has_polar
            })
            
        if len(objs) > 0:
            record["annotations"] = objs
            record["annotations"] = objs
            dataset_dicts.append(record)

    # FIX: PyCOCOTools crashes if 'info' is missing in the dataset registry meta? 
    # Actually detectron2 registers it as list of dicts. 
    # The error comes from COCOEvaluator trying to load ground truth.
    # We need to rely on the JSON file passing 'info'.
    # If the JSON file itself does not have 'info', we must inject it when registering?
    # No, 'register_coco_instances' loads the JSON directly.
    # So we need to patch the JSON file? 
    # OR we patch COCOEvaluator via M2FRobustTrainer. 
    # But patching the JSON is safer/cleaner.
    
    print(f"Loaded {len(dataset_dicts)} images for split '{split}'")
    return dataset_dicts

def get_polar_from_points(x1, y1, x2, y2):
    if abs(x1 - x2) < 1e-4 and abs(y1 - y2) < 1e-4:
        return 0.0, 0.0
    a = y1 - y2
    b = x2 - x1
    c = -a * x1 - b * y1
    theta = np.arctan2(b, a)
    if theta < 0: theta += np.pi
    rho = np.abs(c) / (np.sqrt(a**2 + b**2) + 1e-6)
    return float(rho), float(theta)

# =============================================================================
# 3. MAPPER (Synchronized Filtering)
# =============================================================================

class BackgroundReplacementMapper:
    def __init__(self, cfg, is_train=True):
        self.is_train = is_train
        self.background_files = sorted(list((project_root / "copy-paste").glob("*.jpg")))
        
        if len(self.background_files) == 0:
            print("WARNING: No background images found in 'copy-paste'. Background replacement disabled.")
        else:
            print(f"Loaded {len(self.background_files)} background images for injection.")
        
        self.debug_count = 0
        self.debug_limit = 50
        self.debug_dir = project_root / "models" / "debug_augmentations"
        if self.is_train:
            os.makedirs(self.debug_dir, exist_ok=True)
        
        # FIX: Safer augmentations - NO CROP (cables get cut!)
        # FIX: Dynamic Augmentation building from Config to support Zoom (LSJ)
        # We respect cfg.INPUT.MIN_SIZE_TRAIN, MAX_SIZE_TRAIN, and CROP
        self.augs_list = utils.build_augmentation(cfg, is_train=True)
        
        # Add minimal color jitter if not present in build_augmentation default
        # (Though usually included if configured, here we manually add it safely)
        # Check if RandomBrightness/Contrast are already there? Simpler to just prepend specific ones.
        self.augs_list.insert(0, T.RandomBrightness(0.7, 1.3))
        self.augs_list.insert(1, T.RandomContrast(0.7, 1.3))
        self.augs_list.insert(2, T.RandomSaturation(0.8, 1.2))

    def _replace_background(self, original_img, annotations):
        bg_path = random.choice(self.background_files)
        bg_img = utils.read_image(str(bg_path), format="BGR")
        
        h, w = original_img.shape[:2]
        bg_img = cv2.resize(bg_img, (w, h))
        
        cable_mask = np.zeros((h, w), dtype=np.uint8)
        for ann in annotations:
            segm = ann.get("segmentation", None)
            if segm:
                for poly in segm:
                    poly_np = np.array(poly).reshape((-1, 2)).astype(np.int32)
                    cv2.fillPoly(cable_mask, [poly_np], 255)
        
        # FIX: Dilatazione più aggressiva per preservare bordi
        kernel = np.ones((5, 5), np.uint8)
        cable_mask = cv2.dilate(cable_mask, kernel, iterations=2)
        
        mask_bool = cable_mask > 0
        
        final_img = bg_img.copy()
        final_img[mask_bool] = original_img[mask_bool]
        
        return final_img

    def __call__(self, dataset_dict):
        dataset_dict = copy.deepcopy(dataset_dict)
        image = utils.read_image(dataset_dict["file_name"], format="BGR")
        
        # Background replacement probability (dynamic based on iteration)
        bg_prob = 0.3  # Default warmup
        try:
            current_iter = get_event_storage().iter
            if current_iter >= 5000:
                bg_prob = 0.7  # Higher after warmup
        except:
            pass
            
        # Background replacement
        if self.is_train and len(self.background_files) > 0 and random.random() < bg_prob:
            try:
                image = self._replace_background(image, dataset_dict["annotations"])
                
                if self.debug_count < self.debug_limit:
                    debug_path = self.debug_dir / f"aug_{self.debug_count}.jpg"
                    cv2.imwrite(str(debug_path), image)
                    self.debug_count += 1
                    
            except Exception as e:
                print(f"Background replacement failed: {e}")
        
        # Apply augmentations (NO dynamic crop logic - crop removed entirely)
        aug_input = T.AugInput(image)
        transforms = T.AugmentationList(self.augs_list)(aug_input)
        image = aug_input.image
        h, w = image.shape[:2]
        
        image = image[:, :, ::-1] # BGR -> RGB
        
        image = image[:, :, ::-1]
        image_tensor = torch.as_tensor(image.transpose(2, 0, 1).astype("float32"))
        dataset_dict["image"] = image_tensor

        annos = []
        for obj in dataset_dict.pop("annotations"):
            try:
                new_obj = utils.transform_instance_annotations(obj, transforms, image.shape[:2])
                # FIX: Removed polar coordinates logic as requested (was incomplete/broken)

                if new_obj.get("segmentation"):
                    annos.append(new_obj)
            except Exception as e:
                continue

        instances = utils.annotations_to_instances(annos, image.shape[:2])
        
        if len(instances) > 0 and hasattr(instances, 'gt_masks'):
            if isinstance(instances.gt_masks, PolygonMasks):
                instances.gt_masks = BitMasks.from_polygon_masks(instances.gt_masks, h, w)
            
            # Removed polar coordinates extraction
            
            # VERIFY INPUTS (User Request)
            assert not torch.isnan(image_tensor).any(), "Image contains NaNs!"
            assert not torch.isinf(image_tensor).any(), "Image contains Infs!"
            assert not torch.isnan(instances.gt_masks.tensor).any(), "GT Masks contain NaNs!"
            
            mask_areas = instances.gt_masks.tensor.sum(dim=(1, 2))
            keep_indices = mask_areas > 0
            instances = instances[keep_indices]
        else:
            instances = Instances((h, w))
            instances.gt_classes = torch.tensor([], dtype=torch.int64)
            instances.gt_masks = torch.zeros((0, h, w), dtype=torch.float32)
            instances.gt_polars = torch.zeros((0, 2), dtype=torch.float32)
            instances.gt_polars_valid = torch.zeros((0), dtype=torch.bool)

        dataset_dict["instances"] = instances
        return dataset_dict

# =============================================================================
# 4. ARCHITECTURE & MONKEY PATCHING (FIXED)
# =============================================================================

def forward_prediction_heads_with_embed(self, output, mask_features, attn_mask_target_size):
    decoder_output = self.decoder_norm(output)
    decoder_output = decoder_output.transpose(0, 1)
    
    outputs_class = self.class_embed(decoder_output)
    mask_embed = self.mask_embed(decoder_output)
    outputs_mask = torch.einsum("bqc,bchw->bqhw", mask_embed, mask_features)
    
    attn_mask = F.interpolate(
        outputs_mask, 
        size=attn_mask_target_size, 
        mode="bilinear", 
        align_corners=False
    )
    attn_mask = (
        attn_mask.sigmoid()
        .flatten(2)
        .unsqueeze(1)
        .repeat(1, self.num_heads, 1, 1)
        .flatten(0, 1) < 0.5
    ).bool()
    attn_mask = attn_mask.detach()
    
    return outputs_class, outputs_mask, attn_mask, decoder_output

def patched_set_aux_loss(self, outputs_class, outputs_seg_masks, outputs_embed):
    return [
        {"pred_logits": a, "pred_masks": b, "pred_embeddings": c}
        for a, b, c in zip(outputs_class[:-1], outputs_seg_masks[:-1], outputs_embed[:-1])
    ]

def patched_forward(self, x, mask_features, mask=None):
    assert len(x) == self.num_feature_levels
    src = []
    pos = []
    size_list = []

    for i in range(self.num_feature_levels):
        size_list.append(x[i].shape[-2:])
        pos.append(self.pe_layer(x[i], None).flatten(2))
        src.append(self.input_proj[i](x[i]).flatten(2) + self.level_embed.weight[i][None, :, None])
        pos[-1] = pos[-1].permute(2, 0, 1)
        src[-1] = src[-1].permute(2, 0, 1)

    _, bs, _ = src[0].shape
    query_embed = self.query_embed.weight.unsqueeze(1).repeat(1, bs, 1)
    output = self.query_feat.weight.unsqueeze(1).repeat(1, bs, 1)

    predictions_class = []
    predictions_mask = []
    predictions_embed = []

    outputs_class, outputs_mask, attn_mask, decoder_output = self.forward_prediction_heads_with_embed(
        output, mask_features, size_list[0]
    )
    
    predictions_class.append(outputs_class)
    predictions_mask.append(outputs_mask)
    predictions_embed.append(decoder_output)

    for i in range(self.num_layers):
        level_index = i % self.num_feature_levels
        attn_mask[torch.where(attn_mask.sum(-1) == attn_mask.shape[-1])] = False
        
        output = self.transformer_cross_attention_layers[i](
            output, src[level_index],
            memory_mask=attn_mask,
            memory_key_padding_mask=None,
            pos=pos[level_index], query_pos=query_embed
        )
        output = self.transformer_self_attention_layers[i](
            output, tgt_mask=None,
            tgt_key_padding_mask=None,
            query_pos=query_embed
        )
        output = self.transformer_ffn_layers[i](output)

        outputs_class, outputs_mask, attn_mask, decoder_output = self.forward_prediction_heads_with_embed(
            output, mask_features, size_list[(i + 1) % self.num_feature_levels]
        )
            
        predictions_class.append(outputs_class)
        predictions_mask.append(outputs_mask)
        predictions_embed.append(decoder_output)

    out = {
        'pred_logits': predictions_class[-1],
        'pred_masks': predictions_mask[-1],
        'pred_embeddings': predictions_embed[-1],
        'aux_outputs': self._set_aux_loss(predictions_class, predictions_mask, predictions_embed)
    }
    return out

MultiScaleMaskedTransformerDecoder.forward_prediction_heads_with_embed = forward_prediction_heads_with_embed
MultiScaleMaskedTransformerDecoder.forward = patched_forward
MultiScaleMaskedTransformerDecoder._set_aux_loss = patched_set_aux_loss

# =============================================================================
# 5. LOSS FUNCTIONS (FIXED & BALANCED)
# =============================================================================

def compute_skeleton_recall_loss(pred_logits, gt_mask):
    """
    FIX: Skeleton Recall più stabile con smoothing e NaN safeguards
    """
    # NaN Check on inputs
    if torch.isnan(pred_logits).any() or torch.isinf(pred_logits).any():
        return torch.tensor(0.0, device=pred_logits.device)
    if torch.isnan(gt_mask).any() or torch.isinf(gt_mask).any():
        return torch.tensor(0.0, device=pred_logits.device)
    
    pred_probs = torch.sigmoid(pred_logits)
    target_region = gt_mask.float()
    
    # FIX: Smooth per evitare oscillazioni
    smooth = 1.0
    intersection = (pred_probs * target_region).sum(dim=(2, 3))
    target_area = target_region.sum(dim=(2, 3)) + smooth
    
    recall = intersection / target_area
    
    result = (1.0 - recall).mean()
    
    # NaN Check on output
    if torch.isnan(result) or torch.isinf(result):
        return torch.tensor(0.0, device=pred_logits.device)
    
    return result

# =============================================================================
# 6. CRITERION (REBALANCED)
# =============================================================================

class SegSetCriterion(SetCriterion):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def loss_tversky(self, outputs, targets, indices, num_masks):
        """
        FIX: Tversky più stabile con epsilon e clipping. 
        Salta calcolo se peso è 0.
        """
        # SKIP IF WEIGHT IS 0
        if self.weight_dict.get("loss_tversky", 0.0) == 0.0:
             return {"loss_tversky": torch.tensor(0.0, device=outputs["pred_masks"].device)}

        assert "pred_masks" in outputs
        src_idx = self._get_src_permutation_idx(indices)
        tgt_idx = self._get_tgt_permutation_idx(indices)
        src_masks = outputs["pred_masks"]
        src_masks = src_masks[src_idx]
        masks = [t["masks"] for t in targets]
        target_masks, valid = nested_tensor_from_tensor_list(masks).decompose()
        target_masks = target_masks.to(src_masks)
        target_masks = target_masks[tgt_idx]

        src_masks = src_masks[:, None]
        target_masks = target_masks[:, None]

        with torch.no_grad():
            point_coords = get_uncertain_point_coords_with_randomness(
                src_masks,
                lambda logits: calculate_uncertainty(logits),
                self.num_points,
                self.oversample_ratio,
                self.importance_sample_ratio,
            )
            point_labels = point_sample(
                target_masks,
                point_coords,
                align_corners=False,
            ).squeeze(1)

        point_logits = point_sample(
            src_masks,
            point_coords,
            align_corners=False,
        ).squeeze(1)

        # FIX: Calcolo più robusto
        src_probs = torch.sigmoid(point_logits).clamp(1e-7, 1 - 1e-7)
        target_labels = point_labels
        
        true_pos = (src_probs * target_labels).sum(dim=1)
        false_neg = (target_labels * (1 - src_probs)).sum(dim=1)
        false_pos = ((1 - target_labels) * src_probs).sum(dim=1)
        
        # FIX: Smoothing costante invece di epsilon microscopica
        smooth = 1.0
        
        # FIX: Priority to Recall (alpha=0.3, beta=0.7) - ACTUALLY APPLIED
        alpha = 0.3 
        beta = 0.7
        
        numerator = true_pos + smooth
        denominator = true_pos + alpha * false_pos + beta * false_neg + smooth
        
        tversky_index = numerator / denominator
        loss = 1.0 - tversky_index
        
        # FIX: Clipping per evitare esplosioni
        loss = loss.clamp(0, 2.0)
        
        return {"loss_tversky": loss.sum() / max(num_masks, 1)}

    def loss_masks(self, outputs, targets, indices, num_masks):
        """
        FIX 1: Safe Point Sampling & Robust Loss (User Provided Logic)
        """
        assert "pred_masks" in outputs
        src_idx = self._get_src_permutation_idx(indices)
        tgt_idx = self._get_tgt_permutation_idx(indices)
        src_masks = outputs["pred_masks"][src_idx]
        
        # 🔴 CHECK 1: Se non ci sono match, ritorna loss zero
        if src_masks.shape[0] == 0:
            return {"loss_mask": torch.tensor(0.0, device=outputs["pred_masks"].device)}
        
        masks = [t["masks"] for t in targets]
        target_masks, valid = nested_tensor_from_tensor_list(masks).decompose()
        target_masks = target_masks.to(src_masks)[tgt_idx]
        
        # 🔴 CHECK 2: Verifica che target_masks non sia vuoto
        if target_masks.shape[0] == 0:
            return {"loss_mask": torch.tensor(0.0, device=outputs["pred_masks"].device)}
        
        src_masks = src_masks[:, None]
        target_masks = target_masks[:, None]
        
        # 🔴 CHECK 3: NaN pre-sampling
        if torch.isnan(src_masks).any() or torch.isinf(src_masks).any():
            print(f"⚠️ NaN/Inf in src_masks at iter {get_event_storage().iter}")
            # FIX: Return Clean 0.0 Tensor (No Gradients) to avoid NaN propagation
            return {"loss_mask": torch.tensor(0.0, device=src_masks.device)}
        
        with torch.no_grad():
            point_coords = get_uncertain_point_coords_with_randomness(
                src_masks,
                lambda logits: calculate_uncertainty(logits),
                self.num_points,
                self.oversample_ratio,
                self.importance_sample_ratio,
            )
            point_labels = point_sample(target_masks, point_coords, align_corners=False).squeeze(1)
        
        point_logits = point_sample(src_masks, point_coords, align_corners=False).squeeze(1)
        
        # 🔴 CHECK 4: NaN post-sampling
        if torch.isnan(point_logits).any() or torch.isinf(point_logits).any():
            print(f"⚠️ NaN/Inf in point_logits at iter {get_event_storage().iter}")
            return {"loss_mask": torch.tensor(0.0, device=src_masks.device)}
        
        # 🔴 FIX: Clamp logits per stabilità numerica
        point_logits = point_logits.clamp(-10, 10)
        
        gamma = 2.0
        alpha = 0.7 # FIX: Reduced from 0.9 to 0.7 to reduce foreground bias
        
        # 🔴 FIX: Usa BCE stabile con label smoothing (Autocast Safe)
        # probs needed for p_t calculation
        probs = torch.sigmoid(point_logits) 
        ce_loss = F.binary_cross_entropy_with_logits(point_logits, point_labels, reduction='none')
        
        p_t = probs * point_labels + (1 - probs) * (1 - point_labels)
        loss = ce_loss * ((1 - p_t) ** gamma)
        
        if alpha >= 0:
            alpha_t = alpha * point_labels + (1 - alpha) * (1 - point_labels)
            loss = alpha_t * loss
        
        # 🔴 CHECK 5: NaN finale
        final_loss = loss.mean()
        if torch.isnan(final_loss) or torch.isinf(final_loss):
            print(f"⚠️ NaN/Inf in final focal loss at iter {get_event_storage().iter}")
            final_loss = torch.tensor(0.0, device=src_masks.device)

        # 🔴 FIX: Dice Loss (Added)
        numerator = 2 * (probs * point_labels).sum(1)
        denominator = probs.sum(1) + point_labels.sum(1)
        dice_loss = 1 - (numerator + 1.0) / (denominator + 1.0)
        dice_loss = dice_loss.mean()
        
        return {"loss_mask": final_loss, "loss_dice": dice_loss}

    def loss_skeleton_recall(self, outputs, targets, indices, num_masks):
        """
        FIX: Loss specifica per connettività. Salta se peso = 0.
        """
        # SKIP IF WEIGHT IS 0
        if self.weight_dict.get("loss_skeleton", 0.0) == 0.0:
             return {"loss_skeleton": torch.tensor(0.0, device=outputs["pred_masks"].device)}
             
        assert "pred_masks" in outputs
        src_idx = self._get_src_permutation_idx(indices)
        tgt_idx = self._get_tgt_permutation_idx(indices)
        src_masks = outputs["pred_masks"]
        src_masks = src_masks[src_idx]
        masks = [t["masks"] for t in targets]
        target_masks, valid = nested_tensor_from_tensor_list(masks).decompose()
        target_masks = target_masks.to(src_masks)
        target_masks = target_masks[tgt_idx]

        src_masks = src_masks[:, None]
        target_masks = target_masks[:, None]
        
        if src_masks.shape[-2:] != target_masks.shape[-2:]:
            target_masks = F.interpolate(
                target_masks, 
                size=src_masks.shape[-2:], 
                mode="bilinear", 
                align_corners=False
            )
        
        loss = compute_skeleton_recall_loss(src_masks, target_masks)
        return {"loss_skeleton": loss}

    def loss_confidence(self, outputs, targets, indices, num_masks):
        """
        Entropy penalty ONLY on foreground GT pixels.
        Forces confident predictions (probs near 0 or 1).
        """
        if self.weight_dict.get("loss_confidence", 0.0) == 0.0:
            return {"loss_confidence": torch.tensor(0.0, device=outputs["pred_masks"].device)}
        
        src_idx = self._get_src_permutation_idx(indices)
        tgt_idx = self._get_tgt_permutation_idx(indices)
        
        src_masks = outputs["pred_masks"][src_idx]
        masks = [t["masks"] for t in targets]
        target_masks, _ = nested_tensor_from_tensor_list(masks).decompose()
        target_masks = target_masks.to(src_masks)[tgt_idx]
        
        if src_masks.shape[0] == 0:
            return {"loss_confidence": torch.tensor(0.0, device=src_masks.device)}
        
        # Resize if needed
        if src_masks.shape[-2:] != target_masks.shape[-2:]:
            target_masks = F.interpolate(
                target_masks[:, None].float(), 
                size=src_masks.shape[-2:], 
                mode="nearest"
            ).squeeze(1)
        
        probs = torch.sigmoid(src_masks).clamp(1e-7, 1 - 1e-7)
        entropy = -(probs * torch.log(probs) + (1 - probs) * torch.log(1 - probs))
        
        # FIX: Calculate entropy on ALL pixels (weighted by foreground/background)
        # This penalizes uncertainty on both FG (miss) and BG (false positive)
        foreground_mask = target_masks > 0.5
        background_mask = ~foreground_mask
        
        fg_entropy = entropy[foreground_mask].mean() if foreground_mask.sum() > 0 else torch.tensor(0.0, device=src_masks.device)
        bg_entropy = entropy[background_mask].mean() if background_mask.sum() > 0 else torch.tensor(0.0, device=src_masks.device)
        
        # Weight: 0.7 FG (want confident cable) + 0.3 BG (penalize FP)
        return {"loss_confidence": 0.7 * fg_entropy + 0.3 * bg_entropy}

    def get_loss(self, loss, outputs, targets, indices, num_masks):
        if loss == 'tversky': return self.loss_tversky(outputs, targets, indices, num_masks)
        if loss == 'labels': return self.loss_labels(outputs, targets, indices, num_masks)
        if loss == 'masks': return self.loss_masks(outputs, targets, indices, num_masks)
        if loss == 'skeleton': return self.loss_skeleton_recall(outputs, targets, indices, num_masks)
        if loss == 'confidence': return self.loss_confidence(outputs, targets, indices, num_masks)
        return super().get_loss(loss, outputs, targets, indices, num_masks)

# =============================================================================
# 7. META ARCH
# =============================================================================

@META_ARCH_REGISTRY.register()
class SegMaskFormer(MaskFormer):
    @classmethod
    def from_config(cls, cfg):
        ret = super().from_config(cfg)
        matcher = HungarianMatcher(
            cost_class=cfg.MODEL.MASK_FORMER.CLASS_WEIGHT,
            cost_mask=cfg.MODEL.MASK_FORMER.MASK_WEIGHT,
            cost_dice=cfg.MODEL.MASK_FORMER.DICE_WEIGHT,
            num_points=cfg.MODEL.MASK_FORMER.TRAIN_NUM_POINTS,
        )
        
        # FIX: Mathematically balanced weights (CE: 2.0/log(100) ≈ 0.43)
        # UPDATE: Now respecting Config for Strategy B flexibility
        weight_dict = {
            "loss_ce": cfg.MODEL.MASK_FORMER.CLASS_WEIGHT,  # Was 0.43 hardcoded
            "loss_mask": cfg.MODEL.MASK_FORMER.MASK_WEIGHT, # Was 5.0 hardcoded
            "loss_dice": cfg.MODEL.MASK_FORMER.DICE_WEIGHT, # Ensure Dice is here
            "loss_confidence": 0.5,    # NEW: Entropy penalty (Keep hardcoded or add config?)
            "loss_tversky": 0.0,       # Enabled in Stage 2
            "loss_skeleton": 0.0       # Enabled in Stage 3
        }
        
        if cfg.MODEL.MASK_FORMER.DEEP_SUPERVISION:
            dec_layers = cfg.MODEL.MASK_FORMER.DEC_LAYERS
            aux_weight_dict = {}
            for i in range(dec_layers - 1):
                aux_weight_dict.update({k + f"_{i}": v for k, v in weight_dict.items()})
            weight_dict.update(aux_weight_dict)

        criterion = SegSetCriterion(
            ret["sem_seg_head"].num_classes,
            matcher=matcher,
            weight_dict=weight_dict,
            eos_coef=cfg.MODEL.MASK_FORMER.NO_OBJECT_WEIGHT,
            losses=["labels", "masks", "confidence", "tversky", "skeleton"],
            num_points=cfg.MODEL.MASK_FORMER.TRAIN_NUM_POINTS,
            oversample_ratio=cfg.MODEL.MASK_FORMER.OVERSAMPLE_RATIO,
            importance_sample_ratio=cfg.MODEL.MASK_FORMER.IMPORTANCE_SAMPLE_RATIO,
        )
        ret["criterion"] = criterion
        return ret

    def prepare_targets(self, targets, images):
        h_pad, w_pad = images.tensor.shape[-2:]
        new_targets = []
        
        for t in targets:
            if t.has("gt_masks"):
                gt_masks = t.gt_masks
                if isinstance(gt_masks, BitMasks):
                    gt_masks = gt_masks.tensor
            else:
                gt_masks = torch.zeros((0, h_pad, w_pad), device=t.gt_classes.device)

            padded_masks = torch.zeros((gt_masks.shape[0], h_pad, w_pad), 
                                     dtype=gt_masks.dtype, device=gt_masks.device)
            
            if gt_masks.shape[0] > 0:
                padded_masks[:, : gt_masks.shape[1], : gt_masks.shape[2]] = gt_masks
            
            if t.has("gt_polars"):
                current_polars = t.gt_polars
                current_valid = t.gt_polars_valid
            else:
                current_polars = torch.zeros((len(t.gt_classes), 2), device=padded_masks.device)
                current_valid = torch.zeros((len(t.gt_classes)), dtype=torch.bool, device=padded_masks.device)

            new_targets.append({
                "labels": t.gt_classes,
                "masks": padded_masks,
                "polar_coordinates": current_polars,
                "polar_valid": current_valid
            })
            
        return new_targets

# =============================================================================
# 8. STAGED TRAINING HOOK (The "Brain" of the Strategy)
# =============================================================================

class StagedTrainingHook(HookBase):
    def __init__(self, cfg):
        self.cfg = cfg
        self.stage = 0
        self._backbone_frozen = True
        
    def after_step(self):
        current_iter = self.trainer.iter
        wd = self.trainer.model.criterion.weight_dict
        
        # =================================================================
        # STAGE 1 (0-10k): Detection + Confidence, Backbone UNJAILED (Fully Trainable)
        # =================================================================
        # =================================================================
        # =================================================================
        # =================================================================
        # USER CURRICULUM: 4-Stage Strategy (60k Iterations / ~215 Epochs)
        # REFINED: Fast Adaptation (8k) -> Longer Optimization (45k)
        # =================================================================
        
        # STAGE 1 (0 -> 8,000): Head Adaptation (28 Epochs)
        # Accelerating this phase as head adapts quickly.
        if current_iter < 8000:
            if self.stage != 1:
                self.stage = 1
                print("\n🔔 STAGE 1 (0-8k): Head Adaptation (28 Epochs)")
                print("   - Backbone: Active")
                print("   - Losses: CE=1.0, Mask=5.0, Confidence=0.5")
                print("   - Tversky: 0.0, Skeleton: 0.0")
                if self._backbone_frozen:
                    self.trainer.model.backbone.requires_grad_(True)
                    self._backbone_frozen = False
            
            wd['loss_ce'] = 1.0
            wd['loss_mask'] = 5.0
            wd['loss_confidence'] = 0.5
            wd['loss_tversky'] = 0.0
            wd['loss_skeleton'] = 0.0
            
        # STAGE 2 (8,000 -> 25,000): Introduce Tversky (+60 Epochs)
        # Early intro of Tversky to target thin lines sooner.
        elif 8000 <= current_iter < 25000:
            if self.stage != 2:
                self.stage = 2
                print("\n🔔 STAGE 2 (8k-25k): Introduce Tversky (+60 Epochs)")
                print("   - Losses: Tversky Ramping 0.0 -> 2.0")
            
            progress = (current_iter - 8000) / 17000
            wd['loss_tversky'] = 2.0 * progress
            wd['loss_skeleton'] = 0.0
            
        # STAGE 3 (25,000 -> 45,000): Max Tversky + Skeleton Intro (+70 Epochs)
        # EXTENDED: Main optimization phase for structure.
        elif 25000 <= current_iter < 45000:
            if self.stage != 3:
                self.stage = 3
                print("\n🔔 STAGE 3 (25k-45k): Max Tversky + Skeleton Intro (+70 Epochs)")
                print("   - Losses: Tversky=2.0, Skeleton Ramping 0.0 -> 1.0")
                
            wd['loss_tversky'] = 2.0
            progress = (current_iter - 25000) / 20000
            wd['loss_skeleton'] = 1.0 * progress
            
        # STAGE 4 (45,000+): Convergence (+50 Epochs)
        # LR Decay.
        elif self.stage != 4:
            self.stage = 4
            print("\n🔔 STAGE 4 (45k+): Convergence (+50 Epochs)")
            print("   - Losses: All Max (Skel=1.0, Tversky=2.0)")
            wd['loss_ce'] = 1.0
            wd['loss_mask'] = 5.0
            wd['loss_confidence'] = 0.5
            wd['loss_tversky'] = 2.0
            wd['loss_skeleton'] = 1.0

# =============================================================================
# 9. VALIDATION HOOK
# =============================================================================

class ValidationLossHook(HookBase):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg.clone() # Clone to avoid modifying global frozen cfg
        self.cfg.defrost()     # Ensure mutable
        self.eval_period = 20 # Debug: frequent eval
        self.cfg.DATASETS.TRAIN = cfg.DATASETS.TEST
        self.cfg.SOLVER.IMS_PER_BATCH = 1
        self.cfg.DATALOADER.NUM_WORKERS = 0
        self._loader = iter(build_detection_train_loader(self.cfg, mapper=BackgroundReplacementMapper(self.cfg, is_train=False)))
    
    def after_step(self):
        if (self.trainer.iter + 1) % self.eval_period != 0:
            return

        torch.cuda.empty_cache()
        num_val_batches = 10 # More robust validation
        val_accumulators = {k: 0.0 for k in ["val_total", "val_ce", "val_mask", "val_tversky", "val_skel"]}
        
        with torch.no_grad():
            self.trainer.model.train() # FIX: Must be in train mode to compute losses, even for validation!
            for _ in range(num_val_batches):
                try:
                    try:
                        data = next(self._loader)
                    except StopIteration:
                        self._loader = iter(build_detection_train_loader(self.cfg, mapper=BackgroundReplacementMapper(self.cfg, is_train=False)))
                        data = next(self._loader)
                
                    loss_dict = self.trainer.model(data)
                    
                    val_accumulators["val_total"] += sum(loss_dict.values()).item()
                    val_accumulators["val_ce"] += loss_dict.get("loss_ce", torch.tensor(0.0)).item()
                    val_accumulators["val_mask"] += loss_dict.get("loss_mask", torch.tensor(0.0)).item()
                    val_accumulators["val_tversky"] += loss_dict.get("loss_tversky", torch.tensor(0.0)).item()
                    val_accumulators["val_skel"] += loss_dict.get("loss_skeleton", torch.tensor(0.0)).item()
                        
                except Exception as e:
                    pass
            
            # Log to Storage
            storage = get_event_storage()
            for k, v in val_accumulators.items():
                storage.put_scalar(f"validation/{k}", v / num_val_batches)
            
            # Print Table
            table = Table(title=f"Validation Results (Iter {self.trainer.iter})")
            table.add_column("Metric", style="cyan")
            table.add_column("Value", style="magenta")
            for k, v in val_accumulators.items():
                table.add_row(k, f"{v / num_val_batches:.4f}")
            Console().print(table)
            
            torch.cuda.empty_cache()
            
            # FIX: Restore train mode for next iteration!
            self.trainer.model.train()

# =============================================================================
# 9b. RICH LOSS PLOTTER HOOK (Restored)
# =============================================================================

class RichLossPlotterHook(HookBase):
    def __init__(self):
        super().__init__()
        self.history = {
            "Total Loss": [],
            "Val Total Loss": [],
            "Class (CE)": [],
            "Val Class": [],
            "Mask (Focal)": [],
            "Val Mask": [],
            "Mask (Tversky)": [],
            "Val Tversky": [],
            "Skeleton Recall": [],
            "Val Skeleton": []
        }

    def _strip_ansi(self, text):
        ansi_escape = re.compile(r'\x1B(?:[@-Z\\-_]|\[[0-?]*[ -/]*[@-~])')
        return ansi_escape.sub('', text)

    def after_step(self):
        storage = get_event_storage()
        
        if storage.iter % 20 == 0: # Log every 20 iters
            histories = storage.histories()
            
            def get_sum_of_keys(prefix):
                total = 0.0
                found = False
                for k, v in histories.items():
                    if k == prefix or k.startswith(f"{prefix}_"):
                        total += v.latest()
                        found = True
                return total if found else 0.0

            def get_val(name):
                return histories[name].latest() if name in histories else 0.0

            val_total = get_val("total_loss")
            val_total_eval = get_val("validation/val_total") 
            
            sum_ce    = get_sum_of_keys("loss_ce")
            val_ce    = get_val("validation/val_ce")

            sum_mask  = get_sum_of_keys("loss_mask")
            val_mask  = get_val("validation/val_mask")

            sum_dice  = get_sum_of_keys("loss_tversky")
            val_dice  = get_val("validation/val_tversky")

            sum_skel  = get_sum_of_keys("loss_skeleton")
            val_skel  = get_val("validation/val_skel")
            
            self.history["Total Loss"].append(val_total)
            if val_total_eval > 0: self.history["Val Total Loss"].append(val_total_eval)
            
            self.history["Class (CE)"].append(sum_ce)
            if val_ce > 0: self.history["Val Class"].append(val_ce)

            self.history["Mask (Focal)"].append(sum_mask)
            if val_mask > 0: self.history["Val Mask"].append(val_mask)
            
            self.history["Mask (Tversky)"].append(sum_dice)
            if val_dice > 0: self.history["Val Tversky"].append(val_dice)
            
            self.history["Skeleton Recall"].append(sum_skel)
            if val_skel > 0: self.history["Val Skeleton"].append(val_skel)

            COL_WIDTH = 65
            TOTAL_WIDTH = (COL_WIDTH * 2) + 3

            if "time" in histories:
                time_per_step = histories["time"].median(20)
                remaining_steps = self.trainer.max_iter - storage.iter
                eta_str = str(datetime.timedelta(seconds=int(time_per_step * remaining_steps)))
            else:
                eta_str = "..."

            print("\n")
            print(" Loss Summary ".center(TOTAL_WIDTH, "="))
            print(f"🚀 Iter: {storage.iter}/{self.trainer.max_iter} | ⏳ ETA: {eta_str}")
            print(f"📊 Train: Tot:{val_total:.3f} | Cls:{sum_ce:.3f} | Msk:{sum_mask:.3f} | Tvr:{sum_dice:.3f} | Skl:{sum_skel:.3f}")
            if val_total_eval > 0:
                print(f"📉 Val:   Tot:{val_total_eval:.3f} | Cls:{val_ce:.3f} | Msk:{val_mask:.3f} | Tvr:{val_dice:.3f} | Skl:{val_skel:.3f}")
            print("-" * TOTAL_WIDTH)

            # --- PLOTTING ---
            metrics_to_plot = [
                ("Total Loss",  self.history["Total Loss"], self.history["Val Total Loss"]),
                #("Polar Loss", self.history["Polar (Smooth L1)"], self.history["Val Polar"]),
                ("Class (CE)",  self.history["Class (CE)"], self.history["Val Class"]),
                ("Mask (Focal)",  self.history["Mask (Focal)"], self.history["Val Mask"]),
                ("Mask (Tversky)", self.history["Mask (Tversky)"], self.history["Val Tversky"]),
                ("Skeleton Recall", self.history["Skeleton Recall"], self.history["Val Skeleton"])
            ]

            def get_chart_block(series, title, color=None):
                data = series[-40:] if len(series) > 1 else [0]*40
                
                # Generazione contenuto grezzo (Grafico o Messaggio Attesa)
                lines = []
                if len(data) < 2 or (len(series) < 2 and max(data) == 0):
                    lines = ["(Waiting...)"] + [""]*4
                else:
                    cfg = {'height': 5, 'format': '{:6.2f}'}
                    if color: cfg['colors'] = [color]
                    try:
                        chart_str = asciichartpy.plot(data, cfg)
                        lines = chart_str.split('\n')
                    except:
                        lines = ["(Error plotting)"] + [""]*4

                # --- CENTRATURA TITOLO INTELLIGENTE ---
                # 1. Troviamo la riga più lunga del grafico (visibile, senza colori)
                content_width = 0
                for line in lines:
                    content_width = max(content_width, len(self._strip_ansi(line)))
                
                # Se il grafico è troppo stretto (meno del titolo), usiamo la lunghezza del titolo
                content_width = max(content_width, len(title))
                
                # 2. Centriamo il titolo rispetto al contenuto del grafico
                centered_title = title.center(content_width)
                
                # 3. Ritorniamo: Titolo Centrato, Riga Vuota, Grafico
                return [centered_title, ""] + lines

            print("\n")
            for label, train_series, val_series in metrics_to_plot:
                # Titolo della sezione (es. "Total Loss") centrato globalmente
                print(f" {label} ".center(TOTAL_WIDTH))
                print("\n")
                
                block_train = get_chart_block(train_series, "Train", asciichartpy.blue)
                block_val = get_chart_block(val_series, "Val", asciichartpy.red)
                
                # Pareggiamo le altezze
                max_h = max(len(block_val), len(block_train))
                block_train += [""] * (max_h - len(block_train))
                block_val += [""] * (max_h - len(block_val))
                
                for l_train, l_val in zip(block_train, block_val):
                    # 1. Calcoliamo la lunghezza visibile di SX
                    vis_len_train = len(self._strip_ansi(l_train))
                    
                    # 2. Calcoliamo il padding per fissare il separatore
                    # Il separatore sarà sempre al carattere COL_WIDTH
                    padding = " " * max(0, COL_WIDTH - vis_len_train)
                    
                    # 3. Stampa: SX + Padding + Separatore + DX
                    print(f"{l_train}{padding} | {l_val}")
                
                print("-" * TOTAL_WIDTH)
                print("") 

            print("=" * TOTAL_WIDTH + "\n")

# =============================================================================
# 10. TRAINER
# =============================================================================

class SegTrainer(DefaultTrainer):
    def build_hooks(self):
        hooks = super().build_hooks()
        hooks.insert(-1, StagedTrainingHook(self.cfg))
        hooks.insert(-1, ValidationLossHook(self.cfg))
        hooks.insert(-1, RichLossPlotterHook())
        return hooks

    def __init__(self, cfg):
        super().__init__(cfg)
        self.accumulation_steps = 4 # 1 * 4 = 4 Effective Batch
        print(f"⚡ Gradient Accumulation Enabled: {self.accumulation_steps} steps")

    def run_step(self):
        """
        Overwrite run_step for Gradient Accumulation
        """
        assert self.model.training, "[SegTrainer] model was changed to eval mode!"
        
        import time
        start = time.perf_counter()
        
        # 1. Fetch Data
        if not hasattr(self, "_data_loader_iter"):
             self._data_loader_iter = iter(self.data_loader)
             
        try:
            data = next(self._data_loader_iter)
        except StopIteration:
            self._data_loader_iter = iter(self.data_loader)
            data = next(self._data_loader_iter)
            
        data_time = time.perf_counter() - start
        
        # 2. Forward & Loss
        loss_dict = self.model(data)
        
        # 3. Sum Losses
        if isinstance(loss_dict, dict):
            losses = sum(loss_dict.values())
        else:
            losses = loss_dict
            
        # 4. Divide by Accumulation Steps
        losses = losses / self.accumulation_steps
        
        # 5. Backward (Accumulate Grads)
        losses.backward()
        
        # 6. Step & Zero Grad only every N steps
        if (self.iter + 1) % self.accumulation_steps == 0:
            self.optimizer.step()
            self.optimizer.zero_grad()
            
        # 7. Write metrics (Only write unscaled loss for logging accuracy)
        # We restore the full loss value for logging purposes
        if isinstance(loss_dict, dict):
            metrics_dict = {k: v.item() for k, v in loss_dict.items()}
            # FIX: Explicitly add total_loss for the plotter
            metrics_dict["total_loss"] = sum(v.item() for v in loss_dict.values())
        else:
            metrics_dict = {"total_loss": loss_dict.item()}
            
        metrics_dict["data_time"] = data_time
        
        # FIX: Use EventStorage directly instead of _write_metrics
        storage = get_event_storage()
        storage.put_scalars(**metrics_dict)
        
        # self.iter is managed by TrainerBase loop, do not increment manually!

    
    @classmethod
    def build_lr_scheduler(cls, cfg, optimizer):
        # LR Warmup (5k iter) + Cosine Decay
        def lr_lambda(iteration):
            warmup_iters = 5000
            max_iter = cfg.SOLVER.MAX_ITER
            if iteration < warmup_iters:
                # Linear warmup: 0 → 1
                return iteration / warmup_iters
            # Cosine decay: 1 → 0
            progress = (iteration - warmup_iters) / (max_iter - warmup_iters)
            return 0.5 * (1 + np.cos(np.pi * progress))
        return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
    
    @classmethod
    def build_train_loader(cls, cfg):
        return build_detection_train_loader(cfg, mapper=BackgroundReplacementMapper(cfg, is_train=True))

# =============================================================================
# 11. CONFIG SETUP
# =============================================================================

def setup_seg_config(output_dir):
    cfg = get_cfg()
    add_maskformer2_config(cfg)
    config_file = project_root / 'Mask2Former' / 'configs' / 'coco/instance-segmentation/maskformer2_R50_bs16_50ep.yaml'
    
    cfg.set_new_allowed(True)
    cfg.merge_from_file(str(config_file))
    cfg.set_new_allowed(False)

    # FIX: Gradient Clipping rilassato (Overrides 'full_model' from Mask2Former config)
    cfg.SOLVER.CLIP_GRADIENTS.CLIP_TYPE = "norm"
    cfg.SOLVER.CLIP_GRADIENTS.CLIP_VALUE = 1.0
    cfg.SOLVER.CLIP_GRADIENTS.ENABLED = True
    cfg.SOLVER.CLIP_GRADIENTS.NORM_TYPE = 2.0

    # 1. INPUT RESOLUTION (User Request: "Multiple of 12")
    # FIX: Emergency Resolution Reduction for OOM
    cfg.INPUT.MIN_SIZE_TRAIN = (480, 512, 544, 576) 
    cfg.INPUT.MAX_SIZE_TRAIN = 1333
    cfg.INPUT.MIN_SIZE_TEST = 672
    
    # Decoder Depth Upgrade (Curriculum Step 1)
    cfg.MODEL.MASK_FORMER.DEC_LAYERS = 9 # Deepening the decoder
    
    # Pad to multiple of 96
    cfg.MODEL.MASK_FORMER.SIZE_DIVISIBILITY = 96
    
    # --- RESTORED CONFIGURATIONS ---
    cfg.DATASETS.TRAIN = ("ttpla_train",)
    cfg.DATASETS.TEST = ("ttpla_val",)
    cfg.DATALOADER.NUM_WORKERS = 2 # Fix: Avoid WSL deadlock
    
    cfg.MODEL.META_ARCHITECTURE = "SegMaskFormer"
    cfg.MODEL.SEM_SEG_HEAD.NUM_CLASSES = 1
    cfg.MODEL.RETINANET.NUM_CLASSES = 1
    cfg.MODEL.MASK_FORMER.NUM_OBJECT_QUERIES = 100
    # -------------------------------
    
    # 2. BATCH SIZE
    cfg.SOLVER.IMS_PER_BATCH = 1  # Reduced to 1 due to OOM
    
    # Loss Weights
    cfg.MODEL.MASK_FORMER.NO_OBJECT_WEIGHT = 0.1
    cfg.MODEL.MASK_FORMER.CLASS_WEIGHT = 1.0       # Matching U-Net BCE weight (approx)
    cfg.MODEL.MASK_FORMER.MASK_WEIGHT = 5.0        # Reduced from 20.0 (U-Net has 1.0 but M2F needs higher)
    cfg.MODEL.MASK_FORMER.DICE_WEIGHT = 1.0        # Enabled
    # FIX: Tversky enabled in trainer, weight handled there? 
    # Actually, train_resnet custom trainer modifies weights in the Hook!
    # I should check StagedTrainingHook.
    
    cfg.MODEL.MASK_FORMER.SKELETON_WEIGHT = 0.0
    
    # Validation PeriodSOLVER
    cfg.SOLVER.BASE_LR = 0.0001
    cfg.SOLVER.WEIGHT_DECAY = 0.05
    cfg.SOLVER.MAX_ITER = 40000
    cfg.SOLVER.CHECKPOINT_PERIOD = 2000
    cfg.SOLVER.AMP.ENABLED = True # FIX: Enable AMP for memory savings
    
    # 5. MODEL
    cfg.MODEL.MASK_FORMER.TRAIN_NUM_POINTS = 12544
    cfg.MODEL.MASK_FORMER.OVERSAMPLE_RATIO = 3.0
    cfg.MODEL.MASK_FORMER.IMPORTANCE_SAMPLE_RATIO = 0.75
    
    # RESNET CONFIGURATION
    cfg.MODEL.RESNETS.DEPTH = 101
    # FIX: Use direct URL to avoid 403 Forbidden
    cfg.MODEL.WEIGHTS = "https://dl.fbaipublicfiles.com/detectron2/ImageNetPretrained/MSRA/R-101.pkl"
    # Remove Swin specific params that might cause warnings/errors if left
    
    cfg.INPUT.FORMAT = "RGB"
    cfg.MODEL.PIXEL_MEAN = [123.675, 116.280, 103.530]
    cfg.MODEL.PIXEL_STD = [58.395, 57.120, 57.375]
    
    cfg.OUTPUT_DIR = output_dir
    os.makedirs(cfg.OUTPUT_DIR, exist_ok=True)
    return cfg

# =============================================================================
# 12. MAIN
# =============================================================================

def main():
    os.system('cls' if os.name == 'nt' else 'clear')
    OUTPUT_DIR = "./models/output_m2f_resnet101" # Changed output dir
    cfg = setup_seg_config(OUTPUT_DIR)
    
    print("\n" + "="*85)
    print(f"🚀 STAGED TRAINING STRATEGY (RESNET-101)")
    print("="*85)
    print(f"   [Stage 1] 0-10k:   Detection + Confidence. Backbone Un-Frozen.")
    print(f"   [Stage 2] 10k-20k: + Soft Tversky (ramp).")
    print(f"   [Stage 3] 20k-30k: + Skeleton (ramp).")
    print(f"   [Stage 4] 30k-40k: Final optimization.")
    print("-" * 85)
    print(f"🔧 CONFIG:")
    print(f"   - Losses: CE=0.43, Mask=5.0, Confidence=0.5")
    print(f"   - LR: Warmup (5k) + Cosine Decay")
    print(f"   - Batch: 1 (Accumulation x4 = 4 Effective)")
    print("="*85)
    
    if torch.cuda.is_available():
        gpu_name = torch.cuda.get_device_name(0)
        t = torch.cuda.get_device_properties(0).total_memory
        r = torch.cuda.memory_reserved(0)
        a = torch.cuda.memory_allocated(0)
        free = (t - r + a) / 1024**3
        print(f"✅ GPU: {gpu_name} ({free:.2f} GB Free)")
    else:
        print("⚠️  GPU not found")
    
    with suppress_output():
        DatasetCatalog.clear()
        MetadataCatalog.clear()
        data_root = str(project_root / "data")
        
        def register_all():
            for split in ['train', 'val']:
                name = f"ttpla_{split}"
                if name in DatasetCatalog.list():
                    continue
                DatasetCatalog.register(name, lambda s=split: load_ttpla_dataset(data_root, s))
                MetadataCatalog.get(name).set(thing_classes=["cable"])
        
        register_all()
    
    print(f"📚 Dataset loaded")
    print(f"📂 Output: {cfg.OUTPUT_DIR}")
    print(f"⚙️  Workers: {cfg.DATALOADER.NUM_WORKERS} | Batch: {cfg.SOLVER.IMS_PER_BATCH}")
    print("-" * 85)
    print("⏳ Initializing model...")

    trainer = None
    try:
        with suppress_output():
            trainer = SegTrainer(cfg)
            # Resume from last checkpoint if available
            trainer.resume_or_load(resume=False)
            logging.getLogger("detectron2").setLevel(logging.WARNING)
            logging.getLogger("fvcore").setLevel(logging.WARNING)
    except Exception as e:
        print(f"\n❌ INITIALIZATION ERROR:\n{e}")
        import traceback
        traceback.print_exc()
        return

    # Print checkpoint info
    if trainer.start_iter > 0:
        print(f"↪️  Resuming from iteration {trainer.start_iter}")
    else:
        print("🆕 Starting fresh (no checkpoint found)")
    
    print("✅ Model ready! Resuming training...")
    trainer.train()
    
    print("\n" + "="*85)
    print("🎉 MARATHON TRAINING COMPLETED!")
    print(f"📁 Checkpoints saved in: {cfg.OUTPUT_DIR}")
    print("="*85)

if __name__ == "__main__":
    main()
