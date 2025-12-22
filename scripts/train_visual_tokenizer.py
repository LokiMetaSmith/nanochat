"""
Training script for the Visual Tokenizer (UniTok/VQGAN).
"""

import argparse
import os
import time
from tqdm import tqdm
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
import pyarrow.parquet as pq
from PIL import Image
import io
import wandb
import numpy as np

from nanochat.unitok import UniTok, VQLoss, Discriminator
from nanochat.common import print0

class ParquetImageDataset(Dataset):
    def __init__(self, parquet_file, image_size=256):
        self.table = pq.read_table(parquet_file)
        self.images = self.table['image']
        self.image_size = image_size

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img_bytes = self.images[idx].as_py()
        img = Image.open(io.BytesIO(img_bytes)).convert('RGB')
        img = img.resize((self.image_size, self.image_size), Image.BILINEAR)
        img = np.array(img).astype(np.float32) / 127.5 - 1.0 # [-1, 1]
        return torch.from_numpy(img).permute(2, 0, 1) # (C, H, W)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_path", type=str, required=True, help="Path to parquet file")
    parser.add_argument("--output_dir", type=str, default="out/tokenizer_vision")
    parser.add_argument("--image_size", type=int, default=256)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--lr", type=float, default=4.5e-6)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--save_every", type=int, default=1000)
    parser.add_argument("--wandb", action="store_true")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    device = torch.device(args.device)

    # Data
    dataset = ParquetImageDataset(args.data_path, args.image_size)
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, num_workers=4, pin_memory=True)

    # Model
    model = UniTok().to(device)
    discriminator = Discriminator().to(device)

    # Loss
    vqloss = VQLoss(disc_start=0).to(device) # Start disc immediately for simplicity or tune

    # Optimizers
    opt_g = torch.optim.Adam(model.parameters(), lr=args.lr, betas=(0.5, 0.9))
    opt_d = torch.optim.Adam(discriminator.parameters(), lr=args.lr, betas=(0.5, 0.9))

    if args.wandb:
        wandb.init(project="nanochat-vision-tokenizer", config=args)

    global_step = 0
    for epoch in range(args.epochs):
        pbar = tqdm(dataloader)
        for images in pbar:
            images = images.to(device)

            # 1. Update Generator (VQVAE)
            rec, diff = model(images)

            # Reconstruction Loss + GAN Loss (Generator part)
            # We need to compute logits_fake manually if we want to pass them,
            # but VQLoss implementation expects to compute them inside if discriminator is passed.
            # Let's verify VQLoss signature:
            # forward(codebook_loss, inputs, reconstructions, optimizer_idx, global_step, ...)

            loss_g, log_g = vqloss(diff, images, rec, 0, global_step,
                                   last_layer=model.decoder.net[-1].weight,
                                   discriminator=discriminator)

            opt_g.zero_grad()
            loss_g.backward()
            opt_g.step()

            # 2. Update Discriminator
            # We need fresh reconstructions for D update strictly speaking, but reusing is often fine.
            # For strict correctness, detach rec.

            logits_real = discriminator(images.detach())
            logits_fake = discriminator(rec.detach())

            loss_d, log_d = vqloss(diff, images, rec, 1, global_step,
                                   discriminator=discriminator,
                                   logits_real=logits_real,
                                   logits_fake=logits_fake)

            opt_d.zero_grad()
            loss_d.backward()
            opt_d.step()

            global_step += 1

            # Logging
            if global_step % 10 == 0:
                desc = f"Ep {epoch} | G: {loss_g.item():.4f} | D: {loss_d.item():.4f}"
                pbar.set_description(desc)
                if args.wandb:
                    wandb.log({**log_g, **log_d})

            if global_step % args.save_every == 0:
                torch.save(model.state_dict(), os.path.join(args.output_dir, "visual_tokenizer.pt"))

    torch.save(model.state_dict(), os.path.join(args.output_dir, "visual_tokenizer_final.pt"))
    print("Training complete.")

if __name__ == "__main__":
    main()
