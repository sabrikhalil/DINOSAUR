import os
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms

class CustomVOC2012Dataset(Dataset):
    """
    Custom dataset for Pascal VOC 2012.
    Expects images under 'JPEGImages' in the VOC2012 folder.
    """
    def __init__(self, root_dir, split="train", transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.split = split
        
        # Construct the path to the split file.
        split_file = os.path.join(root_dir, "ImageSets", "Main", f"{split}.txt")
        if not os.path.exists(split_file):
            expected_dir = os.path.join(root_dir, "ImageSets", "Main")
            available_files = os.listdir(expected_dir) if os.path.exists(expected_dir) else "Directory not found"
            raise FileNotFoundError(
                f"Expected split file not found at {split_file}.\n"
                f"Please ensure the dataset is correctly extracted.\n"
                f"Contents of the expected directory ({expected_dir}): {available_files}"
            )
            
        with open(split_file, "r") as f:
            self.image_ids = [line.strip() for line in f.readlines()]
        
        self.images_dir = os.path.join(root_dir, "JPEGImages")
    
    def __len__(self):
        return len(self.image_ids)
    
    def __getitem__(self, idx):
        img_id = self.image_ids[idx]
        img_path = os.path.join(self.images_dir, f"{img_id}.jpg")
        image = Image.open(img_path).convert("RGB")
        
        if self.transform:
            image = self.transform(image)
        
        # For unsupervised training, we only need the image.
        # (image shape: (C, H, W))
        return image

def get_transforms(split, resolution=224):
    normalize = transforms.Normalize(
        mean=[0.485, 0.456, 0.406], 
        std=[0.229, 0.224, 0.225]
    )
    if split == "train":
        transform = transforms.Compose([
            transforms.Resize((resolution, resolution)),  # (H, W)
            transforms.RandomHorizontalFlip(),
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
            transforms.ToTensor(),
            #normalize 
        ])
    else:  # validation
        transform = transforms.Compose([
            transforms.Resize((resolution, resolution)),
            transforms.ToTensor(),
            #normalize  
        ])
    return transform

def get_dataloader(root_dir, split="train", batch_size=32, shuffle=True, num_workers=4, resolution=224):
    transform = get_transforms(split, resolution)
    dataset = CustomVOC2012Dataset(root_dir=root_dir, split=split, transform=transform)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers)
    return dataloader

# Quick test
if __name__ == "__main__":
    # Adjust the path to where the VOC dataset was extracted.
    # For example, if you downloaded with download_dataset.py, you might need:
    # voc_root = os.path.join(os.path.dirname(__file__), "PASCAL_VOC2012", "VOCdevkit", "VOC2012")
    voc_root = os.path.join(os.path.dirname(__file__),  "VOCdevkit", "VOC2012")
    loader = get_dataloader(voc_root, split="train", batch_size=4, shuffle=False)
    for batch in loader:
        print("Batch shape:", batch.shape)  # Expected: (batch_size, 3, resolution, resolution)
        break
