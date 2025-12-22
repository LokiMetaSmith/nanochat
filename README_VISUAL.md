# Visual Token (UniTok) Integration Guide

This guide explains how to train NanoChat using native discrete image tokens, enabling the model to generate images autoregressively.

## 1. Prepare Vision Data

First, prepare a dataset of images. You can use a local folder or download CIFAR-10.

```bash
# Download CIFAR-10 and convert to Parquet
uv run python scripts/prepare_vision_data.py --download_cifar --output_file data/cifar_vision.parquet --image_size 64
```

*Note: We use 64x64 images for this example to ensure it runs quickly. The `UniTok` configuration assumes a downsampling factor of 16, resulting in a 4x4 grid (16 tokens).*

## 2. Train the Visual Tokenizer (UniTok)

Train the VQ-VAE/VQGAN tokenizer to learn a discrete codebook for images.

```bash
# Train tokenizer
uv run python scripts/train_visual_tokenizer.py \
    --data_path data/cifar_vision.parquet \
    --output_dir out/tokenizer_vision \
    --image_size 64 \
    --batch_size 16 \
    --epochs 5 \
    --save_every 500
```

This will produce `out/tokenizer_vision/visual_tokenizer.pt`.

## 3. Train GPT with Visual Tokens

Train the multimodal GPT model. It will use the trained tokenizer to encode images into tokens on-the-fly and learn to predict them.

```bash
# Train GPT using the specific configuration
bash run.sh -m scripts.base_train configs/unitok.json
```

*Configuration (`configs/unitok.json`):*
- `use_visual_tokens`: true
- `visual_tokenizer_path`: "out/tokenizer_vision/visual_tokenizer.pt"
- `visual_vocab_size`: 1024 (Matches default in tokenizer script)

## 4. Generate Visual Content

Generate text and images from the trained model.

```bash
# Generate
uv run python scripts/generate_visual.py \
    --checkpoint out/base_checkpoints/d4 \
    --prompt "Generate an image" \
    --use_visual_tokens \
    --visual_tokenizer_path out/tokenizer_vision/visual_tokenizer.pt \
    --max_tokens 100 \
    --device cpu
```

The script will output generated text and save any generated image segments to `out/generation/`.

## Architecture Overview

- **UniTok**: A Vector Quantized GAN (VQGAN) that compresses images into discrete indices.
- **GPT Integration**:
    - The vocabulary is expanded: `[Text Tokens | Visual Tokens]`.
    - Images are encoded to indices, shifted to the visual token range, and prepended to the text sequence.
    - The model predicts visual tokens autoregressively.
    - `GPT.decode_mixed_tokens` handles converting the mixed stream back into text and image tensors.
