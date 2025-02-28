# DINOSAUR: Unsupervised Object Segmentation via Slot Attention & Transformer Decoder

PyTorch implementation of DINOSAUR (Bridging the Gap to real-world object-centric learning), combining DINO-pretrained ViT features with Slot Attention and a Transformer decoder for unsupervised object segmentation.

![epoch_989_step_89000_combined](https://github.com/user-attachments/assets/8b6a72b9-3fee-45a8-840e-e9bb70613f44)


## Key Features
- 🦖 **DINO ViT Encoder**: Frozen DINO-pretrained Vision Transformer for feature extraction.
- 🎰 **Slot Attention**: Iteratively groups image patches into object-centric slots.
- 🔄 **Transformer Decoder**: Autoregressively reconstructs features with cross-attention to slots.
- 🎨 **Segmentation Visualization**: Generates per-slot attention maps for object localization.

## Installation
```bash
git clone https://github.com/yourusername/dinosaur.git
cd dinosaur
pip install -r requirements.txt
```

## Setup

1. **Clone the repository and install dependencies:**

   ```bash
   git clone https://github.com/yourusername/DINOSAUR
   cd DINOSAUR
   pip install -r requirements.txt
   ```

2. **Download the dataset:** 
   
   ```bash 
   python data/download_dataset.py
   ```

3. **Run the training script:** 
 
   ```bash 
   python train.py 
   ```



