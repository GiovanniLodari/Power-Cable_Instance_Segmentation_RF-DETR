import os
import torch
import cv2
import numpy as np
import pandas as pd
from pathlib import Path
from tqdm import tqdm
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
from rfdetr import RFDETRSegPreview
import math
import json
from pycocotools import mask as mask_utils

# ========== CONFIGURAZIONE ==========
CHECKPOINT_DIR = "src/experiments/rf_detr/checkpoints"
VALIDATION_IMAGES = "data/rf-detr_data/test"
VALIDATION_ANNOTATIONS = "data/rf-detr_data/test/_annotations.coco.json"
OUTPUT_CSV = "src/experiments/rf_detr/output/metrics_comparison.csv"
RESOLUTION = 1200
CONF_THRESHOLD = 0.5

NUM_QUERIES = 100
NUM_SELECT = 80
# =====================================

def calculate_polar_coordinates(mask_array):
    """Calcola rho e theta usando la regressione lineare sui punti della maschera."""
    mask_u8 = (mask_array > 0.5).astype(np.uint8)
    kernel = np.ones((3,3), np.uint8)
    mask_u8 = cv2.morphologyEx(mask_u8, cv2.MORPH_OPEN, kernel)
    
    points_yx = np.argwhere(mask_u8 > 0)
    points = points_yx[:, [1, 0]].astype(np.float32)

    if points.shape[0] < 20:
        return 0.0, 0.0, False

    [vx, vy, x0, y0] = cv2.fitLine(points, distType=cv2.DIST_L2, param=0, reps=0.01, aeps=0.01)
    vx, vy, x0, y0 = vx[0], vy[0], x0[0], y0[0]

    theta_normal_rad = math.atan2(vy, vx) + math.pi / 2
    theta_final = theta_normal_rad % (2 * math.pi)

    rho = x0 * math.cos(theta_final) + y0 * math.sin(theta_final)

    if rho < 0:
        rho = -rho
        theta_final = (theta_final + math.pi) % (2 * math.pi)

    return float(rho), float(theta_final), True

def theta_diff_score(theta_pred, theta_gt):
    """Similarità esponenziale - entrambi gli angoli devono essere in [0, 2π]."""
    # Normalizziamo entrambi in [0, π] per la simmetria della linea
    tp = theta_pred % np.pi
    tg = theta_gt % np.pi
    
    # Differenza minima gestendo la periodicità
    t = min(abs(tp - tg), np.pi - abs(tp - tg))
    return np.exp(-0.12 * t)

def calculate_angle_sim_score(pred_masks, gt_thetas, iou_matrix, iou_thresh=0.3):
    """
    Calcola l'angle score con matching migliorato.
    FIX 1: Soglia IoU abbassata a 0.3
    FIX 2: Evita match duplicati
    FIX 3: Diagnostica dettagliata
    """
    scores = []
    matched_gts = set()
    debug_info = []
    
    for i, pred_mask in enumerate(pred_masks):
        best_gt_idx = np.argmax(iou_matrix[i]) if iou_matrix.shape[1] > 0 else -1
        
        # Verifica che il GT non sia già stato matchato
        if best_gt_idx >= 0 and best_gt_idx not in matched_gts and iou_matrix[i][best_gt_idx] >= iou_thresh:
            _, theta_p, ok_p = calculate_polar_coordinates(pred_mask)
            theta_g = gt_thetas[best_gt_idx]
            
            if ok_p and theta_g is not None:
                score = theta_diff_score(theta_p, theta_g)
                scores.append(score)
                matched_gts.add(best_gt_idx)
                
                # Diagnostica
                debug_info.append({
                    'pred_theta_deg': np.rad2deg(theta_p % np.pi),
                    'gt_theta_deg': np.rad2deg(theta_g % np.pi),
                    'score': score,
                    'iou': iou_matrix[i][best_gt_idx]
                })
    
    # Stampa i 5 peggiori match (solo se ci sono problemi)
    if debug_info and len(debug_info) >= 5:
        worst_5 = sorted(debug_info, key=lambda x: x['score'])[:5]
        if worst_5[0]['score'] < 0.95:   # Solo se c'è qualcosa di sospetto
            print("\n🔴 5 peggiori match angolari:")
            for item in worst_5:
                print(f"IoU={item['iou']:.2f} | Pred={item['pred_theta_deg']:.1f}° | GT={item['gt_theta_deg']:.1f}° | Score={item['score']:.3f}")
    
    return np.mean(scores) if scores else 0.0

def load_model_for_eval(checkpoint_path):
    print(f"🔧 Caricamento: {Path(checkpoint_path).name}")
    model = RFDETRSegPreview(
        resolution=RESOLUTION,
        num_queries=NUM_QUERIES, 
        num_select=NUM_SELECT,
        num_classes=91
    )
    
    try:
        internal_model = model.model.model
    except AttributeError:
        internal_model = model

    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    state_dict = checkpoint.get('model_ema', checkpoint.get('model', checkpoint))
    new_state_dict = {k.replace('module.', ''): v for k, v in state_dict.items()}
    
    internal_model.load_state_dict(new_state_dict, strict=True) 
    internal_model.eval()
    
    if torch.cuda.is_available():
        internal_model.cuda()
    return model

def evaluate_single_checkpoint(model, coco_gt, image_dir):
    """Esegue inferenza su tutto il validation set e calcola metriche."""
    results = []
    angle_scores_list = []
    
    image_ids = coco_gt.getImgIds()
    cat_ids = coco_gt.getCatIds()
    target_category_id = cat_ids[0] if cat_ids else 1
    
    prediction_id_counter = 0
    for img_id in tqdm(image_ids, desc="Evaluating", leave=False):
        img_info = coco_gt.loadImgs(img_id)[0]
        file_name = img_info['file_name']
        img_h = img_info.get('height', RESOLUTION)
        img_w = img_info.get('width', RESOLUTION)
        img_path = os.path.join(image_dir, file_name)
        
        detections = model.predict(img_path, threshold=0.01)
        
        if not detections:
            continue

        # Preparazione maschere GT con normalizzazione theta
        ann_ids = coco_gt.getAnnIds(imgIds=img_id)
        anns = coco_gt.loadAnns(ann_ids)
        gt_masks = []
        gt_thetas = []

        for ann in anns:
            mask = coco_gt.annToMask(ann)
            gt_masks.append(mask)
            
            if 'polar_coordinates' in ann and len(ann['polar_coordinates']) > 0:
                theta_raw = ann['polar_coordinates'][0]['theta']
                
                # FIX: Se il JSON ha theta in gradi, decommentare questa riga:
                # theta_raw = np.deg2rad(theta_raw)
            
            # Normalizza in [0, 2π] come fa calculate_polar_coordinates
            theta_normalized = theta_raw % (2 * np.pi)
            gt_thetas.append(theta_normalized)
        else:
            gt_thetas.append(None)
        
        # Calcolo Angle Score
        high_conf_indices = [i for i, c in enumerate(detections.confidence) if c > CONF_THRESHOLD]
        
        if high_conf_indices and gt_masks:
            valid_pred_masks = [detections.mask[i] for i in high_conf_indices]

            # Calcolo IoU
            ious = np.zeros((len(valid_pred_masks), len(gt_masks)))
            for r, pm in enumerate(valid_pred_masks):
                for c, gm in enumerate(gt_masks):
                    intersection = np.logical_and(pm, gm).sum()
                    union = np.logical_or(pm, gm).sum()
                    ious[r, c] = intersection / union if union > 0 else 0

            angle_s = calculate_angle_sim_score(valid_pred_masks, gt_thetas, ious)
            if angle_s > 0:   # Solo se abbiamo match validi
                angle_scores_list.append(angle_s)

        # Formattazione per COCOEval e salvataggio (SPOSTATO FUORI dal blocco angle score)
        for i in range(len(detections.confidence)):
            rle = mask_utils.encode(np.asfortranarray(detections.mask[i].astype(np.uint8)))
            rle['counts'] = rle['counts'].decode('utf-8')
            
            # Calcola bbox dal RLE
            bbox = mask_utils.toBbox(rle).tolist() # [x, y, w, h]
            area = float(mask_utils.area(rle))
            
            # Calcolo rho e theta per la predizione
            rho_p, theta_p, ok_p = calculate_polar_coordinates(detections.mask[i])

            res = {
                'image_id': int(img_id),
                'category_id': target_category_id, 
                'segmentation': rle,
                'bbox': bbox,
                'score': float(detections.confidence[i]),
                'lines': [float(rho_p), float(theta_p)] if ok_p else [0.0, 0.0],
                'area': area,
                'height': int(img_h),
                'width': int(img_w),
                'id': prediction_id_counter
            }
            results.append(res)
            prediction_id_counter += 1

    # Salvataggio Predizioni JSON (con category_id = 0 come richiesto)
    output_json_path = os.path.join(os.path.dirname(OUTPUT_CSV), "predictions.json")
    # Se vuoi salvare un file per ogni checkpoint, usa un nome dinamico, ma qui sovrascriviamo o salviamo l'ultimo
    # Per ora salviamo tutto quello accumulato (che è solo per QUESTO checkpoint in questa funzione?)
    # Se vogliamo salvare i risultati di QUESTO checkpoint:
    
    # Creiamo una copia per il salvataggio con category_id = 0
    results_to_save = []
    for r in results:
        r_save = r.copy()
        r_save['category_id'] = 0
        results_to_save.append(r_save)
        
    with open(output_json_path, 'w') as f:
        json.dump(results_to_save, f)
    # print(f"Salvate predizioni in {output_json_path}") # Opzionale, evito spam loop

    # COCO Evaluation
    coco_dt = coco_gt.loadRes(results) if results else coco_gt.loadRes([])
    
    # 1. Valutazione Segmentation
    coco_eval_segm = COCOeval(coco_gt, coco_dt, 'segm')
    coco_eval_segm.evaluate()
    coco_eval_segm.accumulate()
    coco_eval_segm.summarize()
    
    ap50_segm = coco_eval_segm.stats[1]
    ar50_segm = coco_eval_segm.stats[7] # AR @ maxDets=100 solitamente è index 8 (AR 50:95 | all | 100). Index 7 è AR 50:95 | large 
    # Wait, COCO stats:
    # 0: AP 50:95 all
    # 1: AP 50 all
    # 2: AP 75 all
    # 3: AP small
    # 4: AP medium
    # 5: AP large
    # 6: AR 50:95 all 1
    # 7: AR 50:95 all 10
    # 8: AR 50:95 all 100 <-- Questo è quello che di solito si usa per AR
    
    #ar5095_segm = coco_eval_segm.stats[8]

    # 2. Valutazione BBox
    coco_eval_bbox = COCOeval(coco_gt, coco_dt, 'bbox')
    coco_eval_bbox.evaluate()
    coco_eval_bbox.accumulate()
    coco_eval_bbox.summarize()
    
    ap50_bbox = coco_eval_bbox.stats[1]
    ar5095_bbox = coco_eval_bbox.stats[7]
    
    avg_angle_score = np.mean(angle_scores_list) if angle_scores_list else 0.0
    
    print(f"   📏 Immagini con angle score valido: {len(angle_scores_list)}/{len(image_ids)}")

    return ap50_segm, ar50_segm, ap50_bbox, ar5095_bbox, avg_angle_score

def main():
    checkpoints = sorted(list(Path(CHECKPOINT_DIR).glob("*.pth")))

    if not checkpoints:
        print("Nessun checkpoint trovato!")
        return

    print(f"Trovati {len(checkpoints)} checkpoints.")

    coco_gt = COCO(VALIDATION_ANNOTATIONS)

    metrics_data = []

    for ckpt in checkpoints:
        print(f"\n--- Valutando: {ckpt.name} ---")
        try:
            model = load_model_for_eval(ckpt)

            ap50_segm, ar5095_segm, ap50_bbox, ar5095_bbox, angle_score = evaluate_single_checkpoint(model, coco_gt, VALIDATION_IMAGES)

            lds = ap50_segm + ar5095_segm + (2 * angle_score)

            print(f"📊 Risultati {ckpt.name}:")
            print(f" [SEGM] AP@50: {ap50_segm:.4f} | AR@50:95: {ar5095_segm:.4f}")
            print(f" [BBOX] AP@50: {ap50_bbox:.4f} | AR@50:95: {ar5095_bbox:.4f}")
            print(f" Angle Score: {angle_score:.4f}")
            print(f" LDS: {lds:.4f}")

            metrics_data.append({
                "checkpoint": ckpt.name,
                "AP_50_SEGM": ap50_segm,
                "AR_50_95_SEGM": ar5095_segm,
                "AP_50_BBOX": ap50_bbox,
                "AR_50_95_BBOX": ar5095_bbox,
                "angle_score": angle_score,
                "LDS": lds
            })

            del model
            torch.cuda.empty_cache()

        except Exception as e:
            print(f"Errore su {ckpt.name}: {e}")

    df = pd.DataFrame(metrics_data)
    df = df.sort_values(by="LDS", ascending=False)
    df.to_csv(OUTPUT_CSV, index=False)
    print(f"\n✅ Salvato report comparativo in: {OUTPUT_CSV}")
    print("Miglior checkpoint:")
    print(df.iloc[0])

if __name__ == "__main__":
    main()