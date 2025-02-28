import torch
import torch.nn as nn
import torch.nn.functional as F

class SlotAttention(nn.Module):
    """
    Slot Attention module.
    Groups encoder features (B, N, D_in) into a set of slots (B, K, D_slot)
    """
    def __init__(self, num_slots=6, in_dim=768, slot_dim=256, num_iterations=3):
        super(SlotAttention, self).__init__()
        self.num_slots = num_slots
        self.slot_dim = slot_dim
        self.num_iterations = num_iterations
        
        # Learnable initial slots: (1, num_slots, slot_dim) --- ERROR -- All slots should be initialized by same variable (So, they learn something different)
        #self.slots_mu = nn.Parameter(torch.randn(1, num_slots, slot_dim))
        #self.slots_logsigma = nn.Parameter(torch.zeros(1, num_slots, slot_dim))

        # Learned slot initialization parameters.
        self.slots_mu = nn.Parameter(torch.randn(1, 1, slot_dim))
        
        # FIX 2: Initialize sigma to small positive values.
        self.slots_sigma = nn.Parameter(torch.zeros(1, 1, slot_dim))
        nn.init.xavier_uniform_(self.slots_sigma)  # Ensures controlled variance.
        
        # Linear maps for attention
        self.project_q = nn.Linear(slot_dim, slot_dim)
        self.project_k = nn.Linear(in_dim, slot_dim)
        self.project_v = nn.Linear(in_dim, slot_dim)
        
        # GRU for slot update
        self.gru = nn.GRUCell(slot_dim, slot_dim)
        self.mlp = nn.Sequential(
            nn.Linear(slot_dim, 4 * slot_dim),  ## Hidden space 4 times bigger 
            nn.ReLU(),
            nn.Linear(4 * slot_dim, slot_dim)
        )
        
        self.norm_input = nn.LayerNorm(in_dim)
        self.norm_slots = nn.LayerNorm(slot_dim)
        self.norm_pre_ff = nn.LayerNorm(slot_dim)
        
    def forward(self, inputs):
        """
        inputs: (B, N, D_in)
        returns: slots: (B, num_slots, slot_dim)
        """
        B, N, D_in = inputs.shape
        inputs = self.norm_input(inputs)
        k = self.project_k(inputs)  # (B, N, slot_dim)
        v = self.project_v(inputs)  # (B, N, slot_dim)
        
        # Initialize slots: sample from Gaussian using learned parameters.
        slots = self.slots_mu + self.slots_sigma * torch.randn(B, self.num_slots, self.slot_dim, device=inputs.device)
        
        for _ in range(self.num_iterations):
            slots_prev = slots
            slots_norm = self.norm_slots(slots)
            q = self.project_q(slots_norm)  # (B, num_slots, slot_dim)
            q = q * (self.slot_dim ** -0.5)
            
            # Compute attention
            attn_logits = torch.einsum("bid,bjd->bij", k, q)  # (B, N, num_slots)
            attn = F.softmax(attn_logits, dim=-1)  # (B, N, num_slots)
            attn = attn + 1e-8
            attn_norm = attn / torch.sum(attn, dim=1, keepdim=True)
            
            updates = torch.einsum("bnd,bnk->bkd", v, attn_norm)  # (B, num_slots, slot_dim)
            
            # Slot update using GRU.
            slots = self.gru(
                updates.reshape(-1, self.slot_dim),
                slots_prev.reshape(-1, self.slot_dim)
            )
            slots = slots.reshape(B, self.num_slots, self.slot_dim)
            slots = slots + self.mlp(self.norm_pre_ff(slots))
            
        return slots

# Quick test function
if __name__ == "__main__":
    B, N, D_in = 2, 196, 768   # example: 2 images, 196 patches, feature dim 768
    x = torch.randn(B, N, D_in)
    slot_attn = SlotAttention(num_slots=10, in_dim=D_in, slot_dim=64, num_iterations=3)
    slots = slot_attn(x)
    print("Slots shape:", slots.shape)  # Expected: (2, 10, 64)
