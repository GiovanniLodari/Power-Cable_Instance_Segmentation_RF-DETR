
import os
import sys
import numpy as np
import cv2
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from scipy.ndimage import distance_transform_edt
from tqdm import tqdm
import matplotlib.pyplot as plt
import albumentations as A
from albumentations.pytorch import ToTensorV2

# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../")))
from src.experiments.seg_former.train import CustomUnetPlusPlus, Config

# =============================================================================
# 1. Refinement CNN Head
# =============================================================================
class RefinementCNN(nn.Module):
    def __init__(self, in_channels, hidden_dim=64):
        super().__init__()
        self.net = nn.Sequential(
            # Layer 1
            nn.Conv2d(in_channels, hidden_dim, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(hidden_dim),
            nn.ReLU(inplace=True),
            # Layer 2
            nn.Conv2d(hidden_dim, hidden_dim, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(hidden_dim),
            nn.ReLU(inplace=True),
            # Layer 3
            nn.Conv2d(hidden_dim, hidden_dim, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(hidden_dim),
            nn.ReLU(inplace=True),
            # Output Layer (1 Channel Energy Map)
            nn.Conv2d(hidden_dim, 1, kernel_size=1, padding=0),
            nn.Sigmoid() # Energy map 0..1
        )
        
    def forward(self, x):
        return self.net(x)

# =============================================================================
# 2. Cascaded Model Wrapper
# =============================================================================
class CascadedCableModel(nn.Module):
    def __init__(self, base_model_path, device):
        super().__init__()
        self.device = device
        
        # Load Base Model (U-Net)
        print(f"Loading base model from {base_model_path}...")
        self.base_model = CustomUnetPlusPlus(
                encoder_name=Config.ENCODER,
                encoder_weights=Config.ENCODER_WEIGHTS,
                in_channels=3,
                classes=1,
                encoder_depth=5,
                decoder_channels=(256, 128, 64, 32, 16),
                activation=None,
                decoder_attention_type="scse"
        )
        
        # Load Weights
        state_dict = torch.load(base_model_path, map_location=device)
        new_state_dict = {}
        for k, v in state_dict.items():
            if k.startswith("_orig_mod."):
                 new_state_dict[k[10:]] = v
            else:
                 new_state_dict[k] = v
        self.base_model.load_state_dict(new_state_dict)
        
        # Freeze Base Model
        self.base_model.eval()
        for param in self.base_model.parameters():
            param.requires_grad = False
            
        # Refinement Head
        # Input channels: Decoder output channels (32) due to depth adjustment
        self.refiner = RefinementCNN(in_channels=32, hidden_dim=64)
        
        # We might also want to upsample features if they are subsampled?
        # Unet++ output is usually subsampled by factor 4 for MiT encoder in this config?
        # Let's check: segmentation_head has upsampling=4.
        # This means decoder tensor is 1/4th size. 
        # We need to upsample it BEFORE or AFTER refinement? 
        # Doing it AFTER is cheaper (conv on smaller map). 
        # We will upsample the output of refiner.

    def forward(self, x):
        with torch.no_grad():
            features = self.base_model.encoder(x)
            decoder_output = self.base_model.decoder(features)
            
            # Base Segmentation Output (just for visualization/comparision if needed)
            base_masks = self.base_model.segmentation_head(decoder_output)
            
        # Pass decoder features to Refiner
        energy_map = self.refiner(decoder_output)
        
        # Upsample Result to Original Input Size
        energy_map = nn.functional.interpolate(energy_map, size=x.shape[2:], mode='bilinear', align_corners=False)
        
        return energy_map, base_masks

# =============================================================================
# 3. Energy Map Dataset
# =============================================================================
class EnergyMapDataset(Dataset):
    def __init__(self, root_dir, json_path=None, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        
        # Load COCO JSON if no mask images are present
        if json_path is None:
             json_path = os.path.join(root_dir, "train.json")
             
        from pycocotools.coco import COCO
        print(f"Loading annotations from {json_path}...")
        self.coco = COCO(json_path)
        self.img_ids = self.coco.getImgIds()
        print(f"Found {len(self.img_ids)} images in JSON.")

    def __len__(self):
        return len(self.img_ids)
        
    def __getitem__(self, idx):
        img_id = self.img_ids[idx]
        img_info = self.coco.loadImgs(img_id)[0]
        file_name = img_info['file_name']
        
        # Full path
        img_path = os.path.join(self.root_dir, file_name)
        
        # Read Image
        image = cv2.imread(img_path)
        if image is None:
             # Fallback logic if nested
             img_path = os.path.join(self.root_dir, os.path.basename(file_name))
             image = cv2.imread(img_path)
             if image is None:
                 raise FileNotFoundError(f"Could not read {file_name}")
                 
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Generate Mask from Annotations
        ann_ids = self.coco.getAnnIds(imgIds=img_id)
        anns = self.coco.loadAnns(ann_ids)
        
        mask = np.zeros((image.shape[0], image.shape[1]), dtype=np.uint8)
        for ann in anns:
            # decode mask
            if 'segmentation' in ann:
                 m = self.coco.annToMask(ann)
                 mask = np.maximum(mask, m) # Combine
        
        # Apply Transforms
        if self.transform:
            augmented = self.transform(image=image, mask=mask)
            image = augmented['image']
            mask = augmented['mask']
            
        # Convert Mask to Energy Map (Distance Transform)
        # Mask is Tensor (H, W) or (1, H, W). Need numpy.
        mask_np = mask.numpy()
        if mask_np.ndim == 3:
             mask_np = mask_np[0] # remove channel
             
        # Distance Transform
        # edt computes distance to BACKGROUND (0). 
        # So we want distance inside the cable (1).
        dist = distance_transform_edt(mask_np > 0)
        
        # Normalize (0 to 1)
        if dist.max() > 0:
            dist = dist / dist.max()
            
        # To Tensor
        dist_tensor = torch.tensor(dist, dtype=torch.float32).unsqueeze(0) # (1, H, W)
        
        return image, dist_tensor, mask # Retain mask for viz

# =============================================================================
# 4. Training Loop
# =============================================================================
def train_refiner():
    # Config
    EPOCHS = 15
    LR = 1e-3
    BATCH_SIZE = 4
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    
    # Paths
    BASE_MODEL = "models/segformer_mit_b5_sota/best_model.pth"
    DATA_DIR = "data/train" # Adjust to where training images are
    SAVE_DIR = "src/experiments/seg_former_refinement/output"
    os.makedirs(SAVE_DIR, exist_ok=True)
    
    # Model
    model = CascadedCableModel(BASE_MODEL, DEVICE).to(DEVICE)
    
    # Optimizer (Only Refiner params)
    optimizer = optim.Adam(model.refiner.parameters(), lr=LR)
    criterion = nn.MSELoss()
    
    # Data
    transform = A.Compose([
        A.Resize(height=Config.TRAIN_CROP_SIZE, width=Config.TRAIN_CROP_SIZE),
        A.Normalize(),
        ToTensorV2()
    ])
    
    # Note: We need a dataset that actually works with the file structure.
    # If standard TTPLA dataset class exists and works, use it?
    # But it likely returns (img, mask). We need (img, energy).
    # Using our custom Dataset class defined above.
    train_dataset = EnergyMapDataset(DATA_DIR, transform=transform)
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=2)
    
    print("Starting Refinement Training...")
    
    for epoch in range(EPOCHS):
        model.train() # Set refiner to train (base is frozen manually)
        epoch_loss = 0
        
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{EPOCHS}")
        for images, energy_targets, _ in pbar:
            images = images.to(DEVICE)
            energy_targets = energy_targets.to(DEVICE)
            
            optimizer.zero_grad()
            
            # Forward
            energy_pred, _ = model(images)
            
            # Loss
            loss = criterion(energy_pred, energy_targets)
            
            # Backward
            loss.backward()
            optimizer.step()
            
            epoch_loss += loss.item()
            pbar.set_postfix({"Loss": loss.item()})
            
        print(f"Epoch {epoch+1} Avg Loss: {epoch_loss / len(train_loader):.6f}")
        
        # Save Checkpoint
        torch.save(model.refiner.state_dict(), os.path.join(SAVE_DIR, f"refiner_ep{epoch+1}.pth"))
        
        # Save Debug Visualization
        save_debug_image(model, train_dataset, DEVICE, epoch, SAVE_DIR)

def save_debug_image(model, dataset, device, epoch, save_dir):
    model.eval()
    # Pick random sample
    idx = np.random.randint(0, len(dataset))
    image, energy_gt, mask_gt = dataset[idx]
    
    input_tensor = image.unsqueeze(0).to(device)
    
    with torch.no_grad():
        energy_pred, base_pred = model(input_tensor)
        
    # Unpack
    energy_pred = energy_pred.squeeze().cpu().numpy()
    energy_gt = energy_gt.squeeze().cpu().numpy()
    base_pred = torch.sigmoid(base_pred).squeeze().cpu().numpy()
    
    # Denormalize Image for viz
    # Approx denorm
    img_np = image.permute(1, 2, 0).cpu().numpy()
    img_np = (img_np * 0.229 + 0.485) # Simple un-normalization
    img_np = np.clip(img_np, 0, 1)
    
    plt.figure(figsize=(15, 5))
    
    plt.subplot(1, 4, 1)
    plt.imshow(img_np)
    plt.title("Input Image")
    plt.axis('off')
    
    plt.subplot(1, 4, 2)
    plt.imshow(base_pred, cmap='gray')
    plt.title("Frozen U-Net Mask")
    plt.axis('off')
    
    plt.subplot(1, 4, 3)
    plt.imshow(energy_gt, cmap='jet')
    plt.title("GT Energy Map (Dist)")
    plt.axis('off')
    
    plt.subplot(1, 4, 4)
    plt.imshow(energy_pred, cmap='jet')
    plt.title(f"Pred Energy Map (Ep {epoch+1})")
    plt.axis('off')
    
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, f"viz_ep{epoch+1}.png"))
    plt.close()

if __name__ == "__main__":
    train_refiner()
