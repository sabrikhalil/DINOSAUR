import torch
import torch.nn as nn
import torch.nn.functional as F

class TransformerDecoder(nn.Module):
    def __init__(self, num_layers=4, num_heads=8, d_model=768, 
                 hidden_dim=3072, num_tokens=196, slot_dim=256):
        super().__init__()
        self.d_model = d_model
        self.num_tokens = num_tokens
        
        # Learned BOS token
        self.bos = nn.Parameter(torch.randn(1, 1, d_model))
        
        # Slot projections
        self.slot_proj = nn.Linear(slot_dim, d_model)
        self.slot_norm = nn.LayerNorm(d_model)
        
        # Positional embeddings
        self.pos_emb = nn.Parameter(torch.randn(1, num_tokens, d_model))
        
        # Transformer layers with pre-normalization
        self.layers = nn.ModuleList([
            TransformerDecoderLayer(d_model, num_heads, hidden_dim) 
            for _ in range(num_layers)
        ])
        
        # Output projection
        self.out = nn.Linear(d_model, d_model)

    def forward(self, slots, target_features):
        B = slots.size(0)
        
        # Project slots to decoder dimension
        slots = self.slot_norm(self.slot_proj(slots))  # (B, K, d_model)
        
        # Prepare shifted inputs with BOS token.
        # target_features: (B, N, d_model) with N=196.
        shifted = torch.cat([self.bos.expand(B, -1, -1), target_features[:, :-1]], dim=1)  # (B, 196, d_model)
        x = shifted + self.pos_emb  # (B, 196, d_model)
        
        # Autoregressive decoding: pass through each layer.
        cross_attns = []
        for layer in self.layers:
            x, attn = layer(x, slots)  # x: (B, L, d_model)
            cross_attns.append(attn)
        
        # Final projection.
        recon = self.out(x)  # (B, L, d_model)
        
        # Process attention from last layer.
        alpha = cross_attns[-1]  # Expected shape: (B, L, K)
        K = alpha.shape[-1]
        # Reshape alpha to (B, K, sqrt(num_tokens), sqrt(num_tokens))
        alpha = alpha.transpose(1, 2)  # (B, K, L)
        grid_size = int(self.num_tokens ** 0.5)
        alpha = alpha.reshape(B, K, grid_size, grid_size)
        
        return recon, alpha

class TransformerDecoderLayer(nn.Module):
    def __init__(self, d_model, num_heads, hidden_dim):
        super().__init__()
        # Self-attention
        self.self_attn = nn.MultiheadAttention(d_model, num_heads)
        self.norm1 = nn.LayerNorm(d_model)
        
        # Cross-attention
        self.cross_attn = nn.MultiheadAttention(d_model, num_heads)
        self.norm2 = nn.LayerNorm(d_model)
        
        # Feed-forward network
        self.mlp = nn.Sequential(
            nn.Linear(d_model, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, d_model)
        )
        self.norm3 = nn.LayerNorm(d_model)

    def forward(self, x, slots):
        # x: (B, L, d_model) and slots: (B, K, d_model)
        # Transpose to (L, B, d_model) for multihead attention.
        x = x.transpose(0, 1)       # (L, B, d_model)
        slots = slots.transpose(0, 1)  # (K, B, d_model)
        L = x.size(0)
        # Create causal mask for self-attention (shape: L x L)
        causal_mask = torch.triu(torch.ones(L, L, dtype=torch.bool, device=x.device), diagonal=1)
        
        # Pre-norm self-attention: apply layer norm before self-attention.
        x_norm = self.norm1(x)
        self_attn_out, _ = self.self_attn(x_norm, x_norm, x_norm, attn_mask=causal_mask, is_causal=True)
        x = x + self_attn_out
        
        # Pre-norm cross-attention: apply layer norm before cross-attention.
        x_norm = self.norm2(x)

        # Cross-attention with un-averaged weights
        cross_attn_out, attn_weights = self.cross_attn(
            query=x_norm, 
            key=slots, 
            value=slots, 
            average_attn_weights=False  # Keep all heads
        )
        # attn_weights shape: (B, num_heads, L, K)
        
        # Average over attention heads
        attn_weights = attn_weights.mean(dim=1)  # (B, L, K)

        x = x + cross_attn_out
        
        # Pre-norm FFN: apply layer norm before feed-forward.
        x_norm = self.norm3(x)
        ffn_out = self.mlp(x_norm)
        x = x + ffn_out
        
        # Transpose back to (B, L, d_model)
        x = x.transpose(0, 1)

        
        return x, attn_weights
