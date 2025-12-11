"""
Data Collector for NanoBot Telemetry.

This script acts as a bridge between the robot (ESP32) and the training loop.
It reads telemetry data (Serial/Mock), buffers it, and writes it to Parquet files.

Usage:
    python scripts/collect_telemetry.py --mock --data_dir data/telemetry
    python scripts/collect_telemetry.py --port /dev/ttyUSB0 --baud 115200
"""

import time
import os
import json
import argparse
import random
import io
import pyarrow as pa
import pyarrow.parquet as pq
import numpy as np
from PIL import Image

def generate_mock_data(step):
    """
    Generate synthetic robot data.
    """
    # Text log
    text = f"Step {step}: Moving to target. Status: OK."

    # Image (Noise)
    # 64x64 random RGB image
    img_array = np.random.randint(0, 255, (64, 64, 3), dtype=np.uint8)
    img = Image.fromarray(img_array)
    img_byte_arr = io.BytesIO()
    img.save(img_byte_arr, format='JPEG')
    image_bytes = img_byte_arr.getvalue()

    # Sensors (16 floats)
    sensors = [random.random() for _ in range(16)]

    # Surface (32 floats)
    surface = [random.random() for _ in range(32)]

    return {
        "text": text,
        "image": image_bytes,
        "sensors": sensors,
        "surface": surface,
        "timestamp": time.time()
    }

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mock", action="store_true", help="Generate mock data")
    parser.add_argument("--port", type=str, default="/dev/ttyUSB0", help="Serial port")
    parser.add_argument("--baud", type=int, default=115200, help="Baud rate")
    parser.add_argument("--data_dir", type=str, default="data/telemetry", help="Output directory")
    parser.add_argument("--buffer_size", type=int, default=100, help="Rows per parquet file")
    parser.add_argument("--interval", type=float, default=0.1, help="Sampling interval (s)")
    args = parser.parse_args()

    os.makedirs(args.data_dir, exist_ok=True)
    print(f"Starting collector. Output: {args.data_dir}")

    buffer = []
    file_count = 0
    step = 0

    try:
        while True:
            if args.mock:
                data = generate_mock_data(step)
                time.sleep(args.interval)
            else:
                # TODO: Implement Serial reading
                # ser.readline() -> parse JSON
                print("Serial not implemented yet. Use --mock.")
                break

            buffer.append(data)
            step += 1

            if len(buffer) >= args.buffer_size:
                # Write to Parquet
                filename = os.path.join(args.data_dir, f"telemetry_{int(time.time())}_{file_count}.parquet")

                # Define Schema
                # text: string
                # image: binary
                # sensors: list<float>
                # surface: list<float>
                # timestamp: double

                table = pa.Table.from_pylist(buffer)
                pq.write_table(table, filename)
                print(f"Wrote {filename} ({len(buffer)} rows)")

                buffer = []
                file_count += 1

    except KeyboardInterrupt:
        print("Stopping collector...")

if __name__ == "__main__":
    main()
