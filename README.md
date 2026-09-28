# Segmentazione di istanze di cavi elettrici con RF-DETR

Segmentazione di istanze di **cavi elettrici** in immagini aeree con [RF-DETR](https://github.com/roboflow/rf-detr) (`RFDETRSegPreview`). Il repository contiene gli script di addestramento, l'inferenza e una valutazione che oltre alle metriche COCO misura quanto bene il modello stima **l'orientamento** dei cavi.

## Come funziona

- **Modello**: RF-DETR Seg con una sola classe (il cavo), 100 query e 60-80 predizioni per immagine.
- **Addestramento in due fasi**:
  - [`train_epoch_0_to_11.py`](src/experiments/rf_detr/train_epoch_0_to_11.py): risoluzione 768 sul dataset intero;
  - [`train_epoch_11_to_45.py`](src/experiments/rf_detr/train_epoch_11_to_45.py): riparte da un checkpoint precedente (`checkpoint0008.pth`) e continua a risoluzione 960 su una versione a **tessere** (*tiled*) del dataset, con pesi diversi per le loss di classe, box e maschera.
- **Valutazione** ([`inference.py`](src/experiments/rf_detr/inference.py)): per ogni checkpoint in `checkpoints/` calcola sul test set
  - AP@50 e AR@50:95 sulle maschere e sui box (COCO);
  - un **angle score**: per ogni cavo predetto, abbinato a un cavo reale con IoU ≥ 0,3, si stima la retta della maschera (regressione lineare sui punti) e si confronta l'angolo con quello reale, con similarità `exp(-0,12·Δθ)`;
  - il **LDS** (Line Detection Score) = AP@50 + AR@50:95 + 2 · angle score.

  I checkpoint vengono ordinati per LDS in un CSV.
- [`run_lds_eval.py`](src/experiments/utils/run_lds_eval.py): calcola AP@50, AR@50, angle score, differenza di rho e LDS partendo da un file di predizioni in formato COCO.
- [`rfdetr_seg_inference.py`](src/experiments/rf_detr/rfdetr_seg_inference.py): esegue il modello su una cartella di immagini e salva le immagini con le maschere predette sovrapposte.

## Dati

I dati non sono nel repository. Gli script si aspettano un dataset in formato COCO/Roboflow (cartelle `train`, `valid`, `test`, ciascuna con `_annotations.coco.json`) in:

| Percorso | Uso |
|---|---|
| `data/rf-detr_data` | Dataset intero (fase 1 e test) |
| `data/rf-detr_data_tiled` | Dataset a tessere (fase 2) |

## Installazione

Serve una GPU CUDA.

```bash
python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

## Uso

Gli script vanno lanciati dalla radice del repository.

```bash
# fase 1
python src/experiments/rf_detr/train_epoch_0_to_11.py

# fase 2 (da un checkpoint precedente, vedi `weights=` nello script)
python src/experiments/rf_detr/train_epoch_11_to_45.py

# confronto dei checkpoint sul test set (scrive output/metrics_comparison.csv)
python src/experiments/rf_detr/inference.py

# LDS di un file di predizioni
python src/experiments/utils/run_lds_eval.py data/test/test.json predictions.json
```
