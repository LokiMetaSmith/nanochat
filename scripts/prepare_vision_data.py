"""
Script to prepare vision data (e.g. ImageNet, CIFAR) for Tokenizer training.
Converts a folder of images into Parquet files with 'image' (bytes) column.
"""

import os
import argparse
import glob
import io
import pyarrow as pa
import pyarrow.parquet as pq
from PIL import Image
from tqdm import tqdm
import random

def process_folder(input_dir, output_file, image_size=256, max_images=None):
    """
    Reads images from input_dir (recursively), resizes them, and saves to a single parquet file.
    """
    image_extensions = ['*.jpg', '*.jpeg', '*.png', '*.webp']
    files = []
    for ext in image_extensions:
        files.extend(glob.glob(os.path.join(input_dir, '**', ext), recursive=True))

    print(f"Found {len(files)} images in {input_dir}")

    if max_images is not None and len(files) > max_images:
        random.shuffle(files)
        files = files[:max_images]
        print(f"Sampled {len(files)} images")

    data = []
    for fpath in tqdm(files):
        try:
            with Image.open(fpath) as img:
                img = img.convert('RGB')
                img = img.resize((image_size, image_size), Image.LANCZOS)

                # Save to bytes
                buf = io.BytesIO()
                img.save(buf, format='PNG')
                img_bytes = buf.getvalue()

                data.append({'image': img_bytes, 'filepath': fpath})
        except Exception as e:
            print(f"Error processing {fpath}: {e}")

    if not data:
        print("No valid images found.")
        return

    # Create PyArrow Table
    table = pa.Table.from_pylist(data)

    # Save to Parquet
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    pq.write_table(table, output_file)
    print(f"Saved {len(data)} images to {output_file}")

def download_cifar10(output_dir):
    """
    Downloads CIFAR-10 as a fallback if no local folder provided.
    """
    from torchvision.datasets import CIFAR10
    print("Downloading CIFAR-10...")
    dataset = CIFAR10(root=output_dir, download=True, train=True)

    # Save images to a folder
    img_dir = os.path.join(output_dir, 'cifar10_images')
    os.makedirs(img_dir, exist_ok=True)

    print("Extracting images...")
    for i, (img, label) in enumerate(tqdm(dataset)):
        img.save(os.path.join(img_dir, f"{i}_{label}.png"))

    return img_dir

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_dir", type=str, help="Path to folder containing images")
    parser.add_argument("--output_file", type=str, default="data/vision_data.parquet", help="Output parquet file")
    parser.add_argument("--download_cifar", action="store_true", help="Download CIFAR-10 if input_dir is not set")
    parser.add_argument("--image_size", type=int, default=256)
    parser.add_argument("--max_images", type=int, default=None)

    args = parser.parse_args()

    if args.input_dir:
        process_folder(args.input_dir, args.output_file, args.image_size, args.max_images)
    elif args.download_cifar:
        # Use a temp dir for download
        tmp_dir = "data/cifar_tmp"
        img_dir = download_cifar10(tmp_dir)
        process_folder(img_dir, args.output_file, args.image_size, args.max_images)
    else:
        print("Please provide --input_dir or --download_cifar")
