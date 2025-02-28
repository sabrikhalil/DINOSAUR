import torch
import torch.nn as nn
import timm

class DINO_ViT_Encoder(nn.Module):
    """
    Frozen DINO ViT encoder.
    Uses a DINO-pretrained Vision Transformer.
    Input: images (B, 3, H, W)
    Output: patch features (B, N, D_feat) where N is the number of patches (excludes the CLS token).
    """
    def __init__(self, model_name="vit_base_patch16_224", patch_size=16, pretrained=True):
        super(DINO_ViT_Encoder, self).__init__()
        # Create the model without default pretrained weights
        self.model = timm.create_model(model_name, pretrained=False)
        
        if pretrained:
            # URL for DINO-pretrained ViT-B/16 weights (from the official DINO repo)
            url = "https://dl.fbaipublicfiles.com/dino/dino_vitbase16_pretrain/dino_vitbase16_pretrain.pth"
            state_dict = torch.hub.load_state_dict_from_url(url, map_location="cpu", check_hash=True)
            # Remove potential "module." prefix in state_dict keys
            new_state_dict = {}
            for k, v in state_dict.items():
                new_key = k.replace("module.", "") if k.startswith("module.") else k
                new_state_dict[new_key] = v
            self.model.load_state_dict(new_state_dict, strict=False)
        
        # Freeze the parameters
        for param in self.model.parameters():
            param.requires_grad = False
        
        # Remove classification head if present
        if hasattr(self.model, "fc"):
            self.model.fc = nn.Identity()
        elif hasattr(self.model, "head"):
            self.model.head = nn.Identity()

    def forward(self, x):
        """
        Forward pass.
        x: (B, 3, H, W)
        Returns:
          features: (B, N, D_feat) where N is the number of patch tokens (CLS token removed if present).
        """
        features = self.model.forward_features(x)  # Expected shape: (B, 197, D_feat)
        # If the first token is the CLS token, remove it.
        if features.shape[1] == 197:
            features = features[:, 1:, :]  # Now shape: (B, 196, D_feat)
        return features

# Quick test function
if __name__ == "__main__":
    encoder = DINO_ViT_Encoder()
    x = torch.randn(2, 3, 224, 224)  # (batch size, channels, height, width)
    feat = encoder(x)
    print("Encoder output shape:", feat.shape)  # Expected: (2, 196, D_feat)
