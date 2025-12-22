#!/bin/bash
set -e  # Interrompi lo script se c'è un errore

echo "🚀 Inizio configurazione ambiente di sviluppo..."

# 1. Verifica compilatore
if ! command -v g++ &> /dev/null; then
    echo "❌ G++ non trovato! Installalo con: sudo apt install build-essential"
    exit 1
fi

# 2. Gestione Virtual Environment
echo "🐍 Configurazione Virtual Environment (wsl_venv)..."

if [ -d "wsl_venv" ]; then
    echo "⚠ La cartella 'wsl_venv' esiste già."
    read -p "Vuoi ricrearla da zero? (y/n) " -n 1 -r
    echo
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        rm -rf wsl_venv
        python3 -m venv wsl_venv
        echo "✓ Vecchio venv rimosso e ricreato."
    else
        echo "✓ Utilizzo venv esistente."
    fi
else
    python3 -m venv wsl_venv
    echo "✓ Nuovo venv creato."
fi

# ATTIVAZIONE VENV
source wsl_venv/bin/activate

# Aggiornamento pip
pip install --upgrade pip

# 3. Installazione PyTorch
echo "⬇️ Verifica/Installazione PyTorch 2.5.1 (CUDA 12.1)..."
pip install torch==2.5.1 torchvision==0.20.1 torchaudio==2.5.1 --index-url https://download.pytorch.org/whl/cu121

# 4. Dipendenze preliminari (AGGIUNTO WHEEL E SETUPTOOLS QUI)
echo "⬇️ Installazione strumenti di build e dipendenze..."
# 'wheel' è essenziale per l'errore 'invalid command bdist_wheel'
pip install wheel setuptools
pip install opencv-python numpy scipy matplotlib pillow cython ninja

# 5. Installazione Detectron2
echo "⬇️ Compilazione e Installazione Detectron2..."
export FORCE_CUDA=1

# Usiamo --no-build-isolation. Ora che abbiamo 'wheel' e 'torch' installati,
# questo comando funzionerà.
python -m pip install --no-build-isolation 'git+https://github.com/facebookresearch/detectron2.git'

# 6. Finalizzazione Mask2Former
echo "⬇️ Configurazione Mask2Former..."

if [ -d "Mask2Former" ]; then
    cd Mask2Former
    
    echo "📦 Installazione requisiti Mask2Former..."
    # Ignoriamo le dipendenze nei txt che potrebbero andare in conflitto, fidandoci di quelle installate
    pip install -r requirements.txt
    
    echo "⚙️ Compilazione estensioni CUDA (MultiScaleDeformableAttention)..."
    pip install --no-build-isolation -e .
    
    cd ..
    echo "✓ Mask2Former configurato correttamente."
else
    echo "❌ ERRORE CRITICO: Cartella 'Mask2Former' non trovata in $(pwd)."
    exit 1
fi

echo "--------------------------------------------------------"
echo "✅ Installazione completata con successo!"
echo "👉 Per attivare l'ambiente: source wsl_venv/bin/activate"
echo "--------------------------------------------------------"