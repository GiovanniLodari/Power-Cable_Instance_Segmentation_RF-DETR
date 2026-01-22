"""
RF-DETR Seg Training - Configurazione corretta
"""
import os
from rfdetr import RFDETRSegPreview

# --- CONFIGURAZIONE ---
DATASET_DIR = "data/rf-detr_data"
OUTPUT_DIR = "src/experiments/rfdetr_seg_v1"

os.makedirs(OUTPUT_DIR, exist_ok=True)

print("=" * 60)
print("🚀 RF-DETR Seg Training")
print("=" * 60)
print()  # Newline per separazione

model = RFDETRSegPreview(
    resolution=768, 
    num_classes=1, 
    num_queries=100,   # Ridotto per velocità
    num_select=80      # Ridotto di conseguenza
)

model.train(
    dataset_dir=DATASET_DIR,
    dataset_file="roboflow",
    output_dir=OUTPUT_DIR,
    
    # Resolution e mask
    resolution=768,  # Divisibile per 24 e 64
    mask_downsample_ratio=1,
    mask_point_sample_ratio=1024,  # Dimezzato: 2048 -> 1024 per velocità
    
    # CHIAVE: num_select < num_queries per evitare errore topk
    num_queries=100,
    num_select=80,
    
    # Scale (DISABILITATO per velocità - era il bottleneck principale)
    multi_scale=False,  # Con True: 85s/step, con False: ~3-4s/step
    expanded_scales=False,
    do_random_resize_via_padding=False,
    square_resize_div_64=True,
    
    # Training
    epochs=100,
    batch_size=1,
    grad_accum_steps=8,
    warmup_epochs=1,
    checkpoint_interval=1,
    
    # LR
    freeze_encoder=False,
    lr=5e-5,
    lr_encoder=3e-6,
    lr_scheduler="step",
    lr_drop=70,
    clip_max_norm=0.1,
    
    # Matching (Più permissivo sulla classe, severo sulla box)
    set_cost_class=0.5,  # Era 1 -> Abbassato per più match
    set_cost_bbox=5.0,   # Era 1 -> Alzato (priorità posizione)
    set_cost_giou=2.0,   # Era 3 -> Bilanciato
    
    # Loss (Penalizza meno l'errore di classe per incoraggiare predizioni)
    cls_loss_coef=1.0,   # Era 2.0 -> Dimezzato
    bbox_loss_coef=5.0,  # Era 1 -> Alzato drasticamente
    giou_loss_coef=2.0,  # Era 3 -> Adattato
    
    mask_dice_loss_coef=20.0,
    mask_ce_loss_coef=1.0,
    
    # Altre
    amp=True,
    use_ema=True,
    device='cuda',
    segmentation_head=True,

    # Resume
    #resume="src/experiments/rfdetr_seg_v1/checkpoint.pth",
)

print("\n✅ Training completato!")
