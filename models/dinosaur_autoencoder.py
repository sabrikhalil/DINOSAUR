import torch
import torch.nn as nn
from models.dino_vit_encoder import DINO_ViT_Encoder
from models.slot_attention import SlotAttention
from models.mlp_decoder import MLPDecoder
from models.transformer_decoder import TransformerDecoder

class DINOSAUR(nn.Module):
    """
    DINOSAUR autoencoder.
    Processes an image through a frozen DINO ViT encoder to get patch features,
    groups them into slots using Slot Attention,
    and decodes the slots to reconstruct the DINO features.
    
    Input:
      image: (B, 3, H, W)
    Outputs:
      recon: (B, N, D_feat) where N is number of patches,
      alpha: (B, K, N, 1) decoder alpha maps (for segmentation visualization).
    """
    def __init__(self, num_slots=10, slot_dim=256, num_iterations=3, feat_dim=768, num_tokens=196):
        super(DINOSAUR, self).__init__()
        self.encoder = DINO_ViT_Encoder()  # frozen DINO ViT, returns (B, N, feat_dim)
        self.pre_slot_mlp = nn.Sequential(
            nn.Linear(feat_dim, feat_dim),
            nn.ReLU(),
            nn.Linear(feat_dim, feat_dim)
        )
        self.slot_attention = SlotAttention(num_slots=num_slots, in_dim=feat_dim, slot_dim=slot_dim, num_iterations=num_iterations)
        self.decoder = TransformerDecoder(num_layers=4, num_heads=8, d_model=feat_dim, hidden_dim=feat_dim*4, num_tokens=num_tokens, slot_dim=slot_dim)

        ### CRITICAL ISSUE : TARGET AND RECON ARE IN DIFFERENT SCALE , TWO SOLUTIONS: 

        ### 1. Add a Un-Noramlization layer to change output scale from 0-1 to whatever is the scale of target 
        self.un_normalize = nn.Linear(feat_dim, feat_dim)  # Initialized near identity.
        nn.init.eye_(self.un_normalize.weight)
        nn.init.zeros_(self.un_normalize.bias)


        # 2. Normalize the target (Non trainable way otherwise collapse !) 
        # Load precomputed stats 
        stats = torch.load("dino_stats.pth")
        self.register_buffer("dino_mean", torch.tensor(stats["mean"]))
        self.register_buffer("dino_std", torch.tensor(stats["std"]))
    
    def forward(self, x):
        # x: (B, 3, H, W)
        feat = self.encoder(x)  # (B, N, feat_dim)

        # Standardize features
        feat_normalized = (feat - self.dino_mean) / self.dino_std

        feat_ = self.pre_slot_mlp(feat_normalized)  # Add this before slot_attention -- 
        slots = self.slot_attention(feat_)  # (B, num_slots, slot_dim)
        recon, alpha = self.decoder(slots, feat_)  # recon: (B, num_tokens, feat_dim), alpha: (B, num_slots, num_tokens, 1)
        #recon = self.un_normalize(recon) # unnormalize the reconstruction to be same scale as target
        return recon, alpha, feat_normalized # Return normalized features as target

# Quick test function
if __name__ == "__main__":
    model = DINOSAUR(num_slots=10, slot_dim=64, num_iterations=3, feat_dim=768, num_tokens=196)
    x = torch.randn(2, 3, 224, 224)  # (B, 3, H, W)
    recon, alpha, feat = model(x)
    print("Reconstruction shape:", recon.shape)  # Expected: (2, 196, 768)
    print("Alpha shape:", alpha.shape)           # Expected: (2, 10, 196, 1)
