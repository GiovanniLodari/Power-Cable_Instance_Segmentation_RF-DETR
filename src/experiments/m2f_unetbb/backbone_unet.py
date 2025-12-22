
import torch
import torch.nn as nn
import segmentation_models_pytorch as smp
from detectron2.modeling import BACKBONE_REGISTRY, Backbone, ShapeSpec

@BACKBONE_REGISTRY.register()
class UNetPPBackbone(Backbone):
    """
    Mask2Former Backbone wrapper for a pre-trained U-Net++ model.
    Extracts features from the U-Net++ DECODER layers (P3, P4) and Encoder (P5).
    """
    def __init__(self, cfg, input_shape):
        super().__init__()
        
        with open("/tmp/backbone_debug.txt", "w") as f:
            f.write("UNetPPBackbone instantiated!\n")
            f.write(f"Config depth: {cfg.MODEL.RESNETS.DEPTH}\n")
        
        encoder_name = cfg.MODEL.RESNETS.DEPTH
        if hasattr(encoder_name, "__int__") and encoder_name == 50:
             encoder_name = "resnet50"
        elif hasattr(encoder_name, "__int__") and encoder_name == 101:
             encoder_name = "resnet101"
        else:
             encoder_name = "resnet50"
             
        print(f"Building U-Net++ Backbone with encoder: {encoder_name}")
        self.unet = smp.UnetPlusPlus(
            encoder_name=encoder_name,
            encoder_weights=None, # FIX: Don't download ImageNet, we load custom weights later!
            in_channels=3,
            classes=1,
            activation=None
        )
        
        # -------------------------------------------------------------
        # MEMORY OPTIMIZATION: Prune High-Res Decoder Layers
        # -------------------------------------------------------------
        # Problem: Standard U-Net++ computes all intermediate nodes up to full resolution.
        # Mask2Former only needs P3 (Stride 8), P4 (Stride 16), P5 (Stride 32).
        # We replace the decoder's forward method to skip ANY node with Stride < 8.
        # Logic: depth_idx correlates with resolution (0=Low, N=High).
        # We determined that computing only (depth_idx + layer_idx <= 1) preserves P3/P4.
        
        import types
        def pruned_forward(decoder_self, features):
            features = features[1:]  # remove first skip
            features = features[::-1]  # reverse

            dense_x = {}
            for layer_idx in range(len(decoder_self.in_channels) - 1):
                for depth_idx in range(decoder_self.depth - layer_idx):
                    
                    # PRUNING CONDITION
                    if depth_idx + layer_idx > 1:
                        continue

                    if layer_idx == 0:
                        output = decoder_self.blocks[f"x_{depth_idx}_{depth_idx}"](
                            features[depth_idx], features[depth_idx + 1]
                        )
                        dense_x[f"x_{depth_idx}_{depth_idx}"] = output
                    else:
                        dense_l_i = depth_idx + layer_idx
                        cat_features = [
                            dense_x[f"x_{idx}_{dense_l_i}"]
                            for idx in range(depth_idx + 1, dense_l_i + 1)
                        ]
                        cat_features = torch.cat(
                            cat_features + [features[dense_l_i + 1]], dim=1
                        )
                        dense_x[f"x_{depth_idx}_{dense_l_i}"] = decoder_self.blocks[
                            f"x_{depth_idx}_{dense_l_i}"
                        ](dense_x[f"x_{depth_idx}_{dense_l_i - 1}"], cat_features)
            
            # Pruned decoder doesn't produce final output
            return None 

        print("Applying Pruned Forward Logic to U-Net++ Decoder (Saving VRAM)...")
        self.unet.decoder.forward = types.MethodType(pruned_forward, self.unet.decoder)
        # -------------------------------------------------------------
        
        # Dictionary to store hooked features
        self._hooked_features = {}
        
        def get_activation(name):
            def hook(model, input, output):
                self._hooked_features[name] = output
            return hook

        # 1. Register hooks on ALL decoder blocks to introspect shapes
        self.hook_handles = []
        for name, module in self.unet.decoder.blocks.named_children():
            # module is a DecoderBlock
            handle = module.register_forward_hook(get_activation(name))
            self.hook_handles.append(handle)
            
        print("Introspecting U-Net++ structure for feature extraction...")
        # 2. Run dummy pass to find P3, P4 candidates
        # FIX: Run on CPU to avoid distributed training device conflicts in __init__
        dummy_input = torch.zeros(1, 3, 256, 256) 
        self.unet.eval()
        
        with torch.no_grad():
            # FIX: Manually run encoder/decoder to avoid segmentation head crash on None
            features_dummy = self.unet.encoder(dummy_input)
            _ = self.unet.decoder(features_dummy)
            
        # 3. Analyze shapes to map Stride -> Block Name
        # We need:
        # P3: Stride 8 (Input 256 -> Output 32)
        # P4: Stride 16 (Input 256 -> Output 16)
        # P5: Stride 32 (Input 256 -> Output 8) -> From Encoder
        
        stride_map = {} # stride -> list of (block_name, channels, dense_index)
        
        for name, feat in self._hooked_features.items():
            _, c, h, w = feat.shape
            stride = 256 // h
            
            # Parse dense index from name "x_{i}_{j}"
            # i is the dense depth (column), higher is better (more processed)
            parts = name.split('_')
            dense_idx = int(parts[1])
            
            if stride not in stride_map: stride_map[stride] = []
            stride_map[stride].append((name, c, dense_idx))
            
        # Select best blocks
        # We want the highest dense_idx for each stride
        self.selected_layers = {} # name -> stride
        
        # P3 (Stride 8)
        if 8 in stride_map:
            best_p3 = sorted(stride_map[8], key=lambda x: x[2])[-1]
            self.selected_layers["p3"] = best_p3
            print(f"Selected P3 (Stride 8): {best_p3[0]} (Channels: {best_p3[1]})")
        else:
            print("WARNING: Could not find Stride 8 decoder block. Fallback likely needed.")

        # P4 (Stride 16)
        if 16 in stride_map:
            best_p4 = sorted(stride_map[16], key=lambda x: x[2])[-1]
            self.selected_layers["p4"] = best_p4
            print(f"Selected P4 (Stride 16): {best_p4[0]} (Channels: {best_p4[1]})")
        else:
             print("WARNING: Could not find Stride 16 decoder block.")

        # P5 (Stride 32) -> Encoder 5
        # Encoder features are [f0, f1, f2, f3, f4, f5]
        # f5 is stride 32
        enc_channels = self.unet.encoder.out_channels
        p5_channels = enc_channels[5]
        print(f"Selected P5 (Stride 32): Encoder Stage 5 (Channels: {p5_channels})")
        
        # 4. Cleanup temporary hooks and register ONLY specific ones
        for h in self.hook_handles: h.remove()
        self.hook_handles = []
        self._hooked_features = {} # Clear cache
        
        # Register permanent hooks
        if "p3" in self.selected_layers:
            name = self.selected_layers["p3"][0]
            self.unet.decoder.blocks.get_submodule(name).register_forward_hook(get_activation("p3"))
            
        if "p4" in self.selected_layers:
            name = self.selected_layers["p4"][0]
            self.unet.decoder.blocks.get_submodule(name).register_forward_hook(get_activation("p4"))
            
        # Define Output Specs
        self._out_features = ["p3", "p4", "p5"]
        self._out_feature_channels = {
            "p3": self.selected_layers["p3"][1],
            "p4": self.selected_layers["p4"][1],
            "p5": p5_channels
        }
        self._out_feature_strides = {"p3": 8, "p4": 16, "p5": 32}
        
        with open("/tmp/backbone_debug.txt", "a") as f:
            f.write(f"Backbone initialized with features: {self._out_features}\n")
        
    def forward(self, x):
        # Clear previous hooks
        self._hooked_features = {}
        
        # Run Encoder
        features = self.unet.encoder(x)
        
        # Run Decoder (hooks will capture P3, P4)
        # We don't need the final mask, just the pass
        # SMP decoder forward requires the feature list
        _ = self.unet.decoder(features)
        
        # Assemble dictionary
        out = {
            "p3": self._hooked_features.get("p3"),
            "p4": self._hooked_features.get("p4"),
            "p5": features[5] # Direct from encoder
        }
        
        return out

    def output_shape(self):
        return {
            name: ShapeSpec(
                channels=self._out_feature_channels[name], stride=self._out_feature_strides[name]
            )
            for name in self._out_features
        }
