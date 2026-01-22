"""
RF-DETR Seg Inference Script
Richiede: pip install rfdetr supervision opencv-python
"""
import os
import cv2
import numpy as np
import torch
import supervision as sv
from pathlib import Path
from rfdetr import RFDETRSegPreview

# ========== CONFIGURAZIONE ==========
CHECKPOINT_PATH = "rfdetr_seg_best.pth"  # Checkpoint nella stessa cartella
INPUT_DIR = "input_images"               # Cartella con immagini da processare
OUTPUT_DIR = "output_predictions"        # Cartella output
CONFIDENCE_THRESHOLD = 0.5               # Threshold di confidenza
RESOLUTION = 768                         # Stessa resolution del training
# =====================================

def load_model(checkpoint_path):
    """Carica il modello RF-DETR Seg da checkpoint."""
    print(f"📂 Caricando modello da: {checkpoint_path}")
    
    model = RFDETRSegPreview(resolution=RESOLUTION)
    
    # Carica il checkpoint
    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    
    # Il checkpoint può avere diversi formati
    if 'model' in checkpoint:
        state_dict = checkpoint['model']
    elif 'model_ema' in checkpoint:
        state_dict = checkpoint['model_ema']
    else:
        state_dict = checkpoint
    
    # Carica i pesi (strict=False per gestire mismatch su num_classes)
    model.model.model.load_state_dict(state_dict, strict=False)
    model.model.model.eval()
    
    if torch.cuda.is_available():
        model.model.model.cuda()
        print("✅ Modello caricato su GPU")
    else:
        print("⚠️ GPU non disponibile, usando CPU")
    
    return model


def predict_image(model, image_path, conf_threshold=0.5):
    """Esegue predizione su una singola immagine."""
    # Usa il metodo predict built-in di RF-DETR
    detections = model.predict(image_path, threshold=conf_threshold)
    return detections


def visualize_and_save(image_path, detections, output_path):
    """Visualizza e salva le predizioni."""
    image = cv2.imread(str(image_path))
    
    # Usa supervision per annotare
    mask_annotator = sv.MaskAnnotator(opacity=0.5)
    box_annotator = sv.BoxAnnotator()
    label_annotator = sv.LabelAnnotator()
    
    # Crea labels con confidenza
    labels = [f"cable {conf:.2f}" for conf in detections.confidence]
    
    # Annota
    annotated = image.copy()
    if detections.mask is not None:
        annotated = mask_annotator.annotate(annotated, detections)
    annotated = box_annotator.annotate(annotated, detections)
    annotated = label_annotator.annotate(annotated, detections, labels=labels)
    
    # Salva
    cv2.imwrite(str(output_path), annotated)
    return annotated


def main():
    print("=" * 60)
    print("🚀 RF-DETR Seg Inference")
    print("=" * 60)
    
    # Setup directories
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    # Carica modello
    model = load_model(CHECKPOINT_PATH)
    
    # Trova immagini
    image_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff'}
    input_path = Path(INPUT_DIR)
    
    if not input_path.exists():
        print(f"❌ Cartella input non trovata: {INPUT_DIR}")
        print(f"   Crea la cartella e inserisci le immagini da processare")
        return
    
    image_files = [f for f in input_path.iterdir() 
                   if f.suffix.lower() in image_extensions]
    
    if not image_files:
        print(f"❌ Nessuna immagine trovata in: {INPUT_DIR}")
        return
    
    print(f"\n📸 Trovate {len(image_files)} immagini\n")
    
    # Processa immagini
    for i, img_path in enumerate(image_files, 1):
        print(f"[{i}/{len(image_files)}] Processando: {img_path.name}")
        
        try:
            # Predizione
            detections = predict_image(model, str(img_path), CONFIDENCE_THRESHOLD)
            
            # Salva
            output_path = Path(OUTPUT_DIR) / f"pred_{img_path.name}"
            visualize_and_save(img_path, detections, output_path)
            
            n_detections = len(detections) if detections else 0
            print(f"   ✅ {n_detections} oggetti trovati -> {output_path.name}")
            
        except Exception as e:
            print(f"   ❌ Errore: {e}")
    
    print(f"\n✅ Completato! Output salvato in: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
