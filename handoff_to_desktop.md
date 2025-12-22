
# Handoff: Laptop (Training) -> Desktop (Inference)

## 1. Setup on Laptop (Already Done)

- **Training**: Running (Round 2, ~60k iterations).
- **Checkpoints**: Automatically saving to `/home/giova/VisionTransformer/shared_checkpoints` every 2500 steps.
- **Git**: I created a `.gitignore` to exclude heavy weights. You should push the code to a private repo.

---

## 2. Setup on Desktop (User Action)

1.  **VPN**: Install **Tailscale Desktop App** on both PC (Laptop) and PC (Desktop). It is much easier than CLI.
    - Ensure they can see each other (Ping).
2.  **Code**: 
    - Laptop: `git init`, `git add .`, `git commit`, `git push`.
    - Desktop: `git clone <repo_url>`.
3.  **Weights**: 
    - The code will *expect* weights in `shared_checkpoints/`.
    - **Transfer**: Copy the latest `.pth` file from Laptop to Desktop using Google Drive, SMB Share, or Tailscale Drop (Right click -> Send).
    - Place it in `shared_checkpoints/` on the Desktop.

---

## 3. Prompt for Desktop Agent

**Copy and Paste this to the Agent on the Desktop PC:**

```markdown
# Role: Inference & Visualization Agent

You are working on the Desktop PC to visualize training progress from the Laptop.
The Laptop is training a PointRend model on 4K images.

## Context
- **Codebase**: Cloned from Git.
- **Weights**: Located in `shared_checkpoints/` (Synced manually via Drive/SMB).
- **Goal**: Run inference on standard test images to verify "Smart Crop" and "Tiling" performance.

## Action Plan
1. **Check Files**: Verify `safe_inference_smart_crop.py` and `ultra_safe_inference.py` are present (via Git).
2. **Check Weights**: Look for the latest `model_00XXXXX.pth` in `shared_checkpoints/`.
3. **Run Inference**:

### Option A: Ultra Safe (Tiling) - Real world test
```bash
python src/experiments/point_rend/ultra_safe_inference.py --weights shared_checkpoints/model_LATEST.pth
```

### Option B: Smart Crop (Oracle) - Debug test
```bash
python src/experiments/point_rend/safe_inference_smart_crop.py --weights shared_checkpoints/model_LATEST.pth
```
```
