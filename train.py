import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import LambdaLR
import torchvision
import wandb
import torch.nn.functional as F
import random
import numpy as np

from models.dinosaur_autoencoder import DINOSAUR
from data.custom_dataset import get_dataloader

def combined_loss(recon, target):
    # MSE loss between reconstructed features and target features.
    #print("Target features stats:")
    #print("Mean:", target.mean().item(), "Std:", target.std().item())
    #print("Recon features stats:")
    #print("Mean:", recon.mean().item(), "Std:", recon.std().item())
    return nn.MSELoss()(recon, target)

def train_epoch(model, train_loader, val_loader, optimizer, criterion, device, current_epoch, start_global_step, log_interval=100):
    model.train()
    running_loss = 0.0
    global_step = start_global_step
    
    for epoch_step, batch in enumerate(train_loader):
        global_step += 1
        batch = batch.to(device)
        optimizer.zero_grad()
        
        recon, _, target = model(batch)

        loss = criterion(recon, target)
        loss.backward()
        grad_norm = nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        
        running_loss += loss.item()
        
        wandb.log({
            "train_loss": loss.item(),
            "grad_norm": grad_norm,
            "lr": optimizer.param_groups[0]['lr'],
            "global_step": global_step,
            "epoch": current_epoch + 1
        })
        print(f"Epoch {current_epoch+1} Step {epoch_step+1}, Global Step {global_step}, Loss: {loss.item():.4f}")
        
        if global_step % log_interval == 0:
            val_loss = validate_epoch(model, val_loader, criterion, device)
            wandb.log({"val_loss": val_loss, "global_step": global_step})
            print(f"--> Validation Loss at Global Step {global_step}: {val_loss:.4f}")
            save_val_examples(model, val_loader, device, current_epoch, global_step)
    
    avg_loss = running_loss / len(train_loader)
    return avg_loss, global_step

def validate_epoch(model, dataloader, criterion, device, max_batches=50):
    model.eval()
    running_loss = 0.0
    with torch.no_grad():
        for i, batch in enumerate(dataloader):
            if i >= max_batches:
                break
            batch = batch.to(device)
            recon, _, target = model(batch)
            loss = criterion(recon, target)
            running_loss += loss.item()
    model.train()
    return running_loss / max_batches




def save_val_examples(model, dataloader, device, epoch, global_step, num_examples=3, blend_alpha=0.5):
    model.eval()
    output_dir = os.path.join("data", "validation_output")
    os.makedirs(output_dir, exist_ok=True)
    
    with torch.no_grad():
        # Randomly sample images from the validation dataset.
        dataset = dataloader.dataset
        indices = random.sample(range(len(dataset)), num_examples)
        samples = [dataset[i] for i in indices]
        batch = torch.stack(samples, dim=0).to(device)  # (B, 3, H, W)
        
        # Get model outputs.
        recon, alpha, target = model(batch)
        # alpha is expected to have shape either:
        # Option A: (B, K, num_tokens, 1) or Option B: (B, K, grid, grid)
        print("Raw alpha shape:", alpha.shape)
        
        # If there is a singleton dimension, squeeze it.
        if alpha.shape[-1] == 1:
            alpha = alpha.squeeze(-1)
        print("Alpha shape after squeeze:", alpha.shape)
        
        # If alpha is 3D ([B, K, num_tokens]), reshape it to a grid.
        if len(alpha.shape) == 3:
            B, K, num_tokens = alpha.shape
            grid_size = int(math.sqrt(num_tokens))
            if grid_size * grid_size != num_tokens:
                print("Warning: num_tokens is not a perfect square!")
            alpha_grid = alpha.reshape(B, K, grid_size, grid_size)
        elif len(alpha.shape) == 4:
            # Assume already in grid form.
            alpha_grid = alpha
            B, K, grid_h, grid_w = alpha_grid.shape
            grid_size = grid_h  # assume square grid
        else:
            raise ValueError("Unexpected alpha shape: {}".format(alpha.shape))
        print("Alpha grid shape:", alpha_grid.shape)
        
        # Upsample the alpha grid using bilinear interpolation to get per-pixel maps.
        up_alpha = F.interpolate(alpha_grid, size=(batch.shape[2], batch.shape[3]),
                                 mode='bilinear', align_corners=False)  # (B, K, H, W)
        # Now compute per-pixel segmentation by taking argmax over the slot (K) dimension.
        segmentation = torch.argmax(up_alpha, dim=1)  # (B, H, W)
        
        # --- Create a Color Palette for Each Slot ---
        palette = np.array([
            [1.0, 0.0, 0.0],    # red
            [0.0, 1.0, 0.0],    # green
            [0.0, 0.0, 1.0],    # blue
            [1.0, 1.0, 0.0],    # yellow
            [1.0, 0.0, 1.0],    # magenta
            [0.0, 1.0, 1.0],    # cyan
        ], dtype=np.float32)
        palette = torch.tensor(palette, device=device)  # (K, 3)
        
        # Map each pixel in the segmentation to its corresponding color.
        seg_color = palette[segmentation]  # (B, H, W, 3)
        seg_color = seg_color.permute(0, 3, 1, 2)  # (B, 3, H, W)
        
        # --- Create Colored Overlay ---
        overlay = (1 - blend_alpha) * batch + blend_alpha * seg_color
        combined = torch.cat([batch, overlay], dim=3)  # (B, 3, H, 2*W)
        grid = torchvision.utils.make_grid(combined.cpu(), nrow=1)
        wandb.log({"combined_input_overlay": wandb.Image(grid)}, step=global_step)
        local_path = os.path.join(output_dir, f"epoch_{epoch+1}_step_{global_step}_combined.png")
        torchvision.utils.save_image(grid, local_path)
        
        # --- Visualize Raw Attention Maps for a Single Example ---
        # For the first image in the batch, visualize all K attention maps.
        # We'll assume alpha_grid is in shape (B, K, grid, grid) with grid ~ 14 (or as given).
        attn_maps = alpha_grid[0]  # (K, grid, grid)
        attn_maps_norm = []
        for i in range(attn_maps.shape[0]):
            attn = attn_maps[i]
            attn = (attn - attn.min()) / (attn.max() - attn.min() + 1e-6)
            attn_maps_norm.append(attn.unsqueeze(0))
        attn_maps_norm = torch.cat(attn_maps_norm, dim=0)  # (K, grid, grid)
        # Create a grid of attention maps (e.g. 2 rows x 3 columns if K=6).
        attn_grid = torchvision.utils.make_grid(attn_maps_norm.unsqueeze(1), nrow=3)
        wandb.log({"attention_maps": wandb.Image(attn_grid)}, step=global_step)
        attn_path = os.path.join(output_dir, f"epoch_{epoch+1}_step_{global_step}_attn.png")
        torchvision.utils.save_image(attn_grid, attn_path)
    
    model.train()





def main():
    wandb.init(project="DINOSAUR", config={
        "learning_rate": 4e-4,
        "batch_size": 64,
        "num_epochs": 500,
        "num_slots": 6,
        "slot_dim": 256,
        "resolution": 224,
        "grad_clip": 1.0
    })
    config = wandb.config
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    model = DINOSAUR(num_slots=config.num_slots, slot_dim=config.slot_dim, num_iterations=3, feat_dim=768, num_tokens=196).to(device)
    
    optimizer = optim.AdamW(
        model.parameters(),
        lr=config.learning_rate,
        betas=(0.9, 0.999),
        eps=1e-8,
        weight_decay=0.0
    )
    
    # Learning rate scheduler: decay LR so that at epoch 100, lr ~ 1e-4.
    scheduler = LambdaLR(optimizer, lr_lambda=lambda epoch: (0.25)**(epoch / 1000))
    
    # Path to VOC dataset (update this path based on your dataset extraction)
    voc_root = os.path.join("data", "VOCdevkit", "VOC2012")
    train_loader = get_dataloader(root_dir=voc_root, split='train', batch_size=config.batch_size, shuffle=True, num_workers=4, resolution=config.resolution)
    val_loader = get_dataloader(root_dir=voc_root, split='val', batch_size=config.batch_size, shuffle=False, num_workers=4, resolution=config.resolution)
    
    start_epoch, global_step = 0, 0
    CHECKPOINT_DIR = "checkpoints"
    os.makedirs(CHECKPOINT_DIR, exist_ok=True)
    checkpoint_files = sorted(os.listdir(CHECKPOINT_DIR))
    if checkpoint_files: 
        latest_ckpt = os.path.join(CHECKPOINT_DIR, checkpoint_files[-1])
        try:
            checkpoint = torch.load(latest_ckpt, map_location=device)
            model.load_state_dict(checkpoint['model'])
            optimizer.load_state_dict(checkpoint['optimizer'])
            scheduler.load_state_dict(checkpoint['scheduler'])
            start_epoch = checkpoint['epoch'] + 1
            global_step = checkpoint['global_step']
            print(f"Resuming from epoch {start_epoch}, step {global_step}")
        except Exception as e:
            print(f"Error loading checkpoint: {e}")
    else:
        print("No checkpoint found. Starting fresh training.")
    
    for epoch in range(start_epoch, config.num_epochs):
        print(f"Epoch {epoch+1}/{config.num_epochs}")
        train_loss, global_step = train_epoch(model, train_loader, val_loader, optimizer, combined_loss, device, epoch, global_step, log_interval=100)
        print(f"Epoch {epoch+1} complete. Average Train Loss: {train_loss:.4f}")
        wandb.log({"epoch_train_loss": train_loss, "epoch": epoch+1})
        
        scheduler.step()
        
        # Save checkpoints every 10 epochs
        if (epoch + 1) % 10 == 0:
            checkpoint = {
                'epoch': epoch,
                'global_step': global_step,
                'model': model.state_dict(),
                'optimizer': optimizer.state_dict(),
                'scheduler': scheduler.state_dict()
            }
            ckpt_path = os.path.join(CHECKPOINT_DIR, f"epoch_{epoch+1}.pth")
            torch.save(checkpoint, ckpt_path)
            #wandb.save(ckpt_path)
            print(f"Checkpoint saved at epoch {epoch+1}.")
    
    print("Training complete.")

if __name__ == "__main__":
    main()
