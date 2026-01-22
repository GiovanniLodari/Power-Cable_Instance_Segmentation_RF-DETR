"""
RF-DETR Seg Training - Configurazione corretta
"""
import os
from rfdetr import RFDETRSegPreview

# --- CONFIGURAZIONE ---
DATASET_DIR = "data/rf-detr_data_tiled"
OUTPUT_DIR = "src/experiments/rfdetr_v3"

os.makedirs(OUTPUT_DIR, exist_ok=True)

print("=" * 60)
print("🚀 RF-DETR Seg Training")
print("=" * 60)
print()  # Newline per separazione

model = RFDETRSegPreview(
    resolution=960, 
    num_classes=1, 
    num_queries=100,   # Ridotto per velocità
    num_select=60      # Ridotto di conseguenza
)

model.train(
    dataset_dir=DATASET_DIR,
    dataset_file="roboflow",
    output_dir=OUTPUT_DIR,

    #resume="src/experiments/rfdetr_v3/checkpoint0008.pth",
    weights="src/experiments/rfdetr_v3/checkpoint0008.pth",

    # Resolution e mask
    resolution=960,  # Divisibile per 24 e 64
    mask_downsample_ratio=1,
    mask_point_sample_ratio=8192,
    
    # CHIAVE: num_select < num_queries per evitare errore topk
    num_queries=100,
    num_select=60,
    num_classes = 1,
    
    # Scale (DISABILITATO per velocità - bottleneck principale)
    multi_scale=False,  # Con True: 85s/step, con False: ~3-4s/step
    expanded_scales=False,
    do_random_resize_via_padding=False,
    square_resize_div_64=True,
    
    # Training
    epochs=100,
    batch_size=4,
    grad_accum_steps=8, 
    warmup_epochs=2,
    checkpoint_interval=1,
    
    # LR
    freeze_encoder=False,
    lr=2e-4,
    lr_encoder=1e-5,
    lr_scheduler="step",
    lr_drop=80,
    clip_max_norm=0.5,
    
    # Matching
    set_cost_class=4.0,  # Era 1 -> Abbassato per più match
    set_cost_bbox=5.0,   # Era 1 -> Alzato (priorità posizione)
    set_cost_giou=2.0,   # Era 3 -> Bilanciato
    
    # Loss (Penalizza meno l'errore di classe per incoraggiare predizioni)
    cls_loss_coef=4.0,   
    bbox_loss_coef=5.0,  # Era 1 -> Alzato drasticamente
    giou_loss_coef=3.0,  # Era 3 -> Adattato
    
    mask_dice_loss_coef=8.0,
    mask_ce_loss_coef=5.0,
    
    # Altre
    amp=True,
    use_ema=True,
    device='cuda',
    segmentation_head=True

)

print("\n✅ Training completato!")
