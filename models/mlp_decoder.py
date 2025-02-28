import torch
import torch.nn as nn
import torch.nn.functional as F

class MLPDecoder(nn.Module):
    """
    MLP Decoder for feature reconstruction.
    For each slot, we broadcast to N tokens, add positional encodings,
    then decode to reconstruct DINO features.
    
    Inputs:
      slots: (B, K, D_slot)
    Outputs:
      recon: (B, N, D_feat)
    """
    def __init__(self, num_slots=10, slot_dim=64, num_tokens=196, feat_dim=768, hidden_dim=128):
        super(MLPDecoder, self).__init__()
        self.num_slots = num_slots
        self.num_tokens = num_tokens  # number of tokens/patches
        self.feat_dim = feat_dim
        
        # Positional embeddings: (1, num_tokens, hidden_dim)
        self.pos_emb = nn.Parameter(torch.randn(1, num_tokens, hidden_dim))
        
        # Shared MLP applied to each slot token.
        self.mlp = nn.Sequential(
            nn.Linear(slot_dim + hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, feat_dim + 1),  # last dim: feat reconstruction + alpha
            nn.LayerNorm(feat_dim +1)  # Add this
        )
        
    def forward(self, slots):
        """
        slots: (B, K, slot_dim)
        Returns:
          recon: (B, num_tokens, feat_dim)
        """
        B, K, D_slot = slots.shape
        # Broadcast each slot to num_tokens:
        slots = slots.unsqueeze(2).expand(B, K, self.num_tokens, D_slot)  # (B, K, num_tokens, slot_dim)
        # Repeat positional embeddings for each slot:
        pos_emb = self.pos_emb.unsqueeze(1).expand(B, K, self.num_tokens, -1)  # (B, K, num_tokens, hidden_dim)
        # Concatenate slot and positional embedding:
        tokens = torch.cat([slots, pos_emb], dim=-1)  # (B, K, num_tokens, slot_dim + hidden_dim)
        # Apply MLP token-wise:
        out = self.mlp(tokens)  # (B, K, num_tokens, feat_dim + 1)
        # Split into reconstructed features and alpha mask logits.
        recon_tokens = out[..., :self.feat_dim]  # (B, K, num_tokens, feat_dim)
        alpha_logits = out[..., -1:]           # (B, K, num_tokens, 1)
        
        # Compute softmax over slots for each token:
        alpha = F.softmax(alpha_logits, dim=1)   # (B, K, num_tokens, 1)
        
        # Weighted sum over slots:
        recon = torch.sum(alpha * recon_tokens, dim=1)  # (B, num_tokens, feat_dim)
        return recon, alpha

# Quick test function
if __name__ == "__main__":
    B, K, slot_dim = 2, 10, 64
    slots = torch.randn(B, K, slot_dim)
    decoder = MLPDecoder(num_slots=K, slot_dim=slot_dim, num_tokens=196, feat_dim=768, hidden_dim=128)
    recon, alpha = decoder(slots)
    print("Reconstruction shape:", recon.shape)  # Expected: (2, 196, 768)
    print("Alpha shape:", alpha.shape)           # Expected: (2, 10, 196, 1)
