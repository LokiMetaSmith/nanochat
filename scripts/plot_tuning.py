import json
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import argparse
import os

def plot_tuning_results(history_file="tuning_history.json", output_dir="tuning_plots"):
    if not os.path.exists(history_file):
        print(f"Error: {history_file} not found.")
        return

    with open(history_file, "r") as f:
        data = json.load(f)

    if not data:
        print("No data found in history file.")
        return

    os.makedirs(output_dir, exist_ok=True)
    df = pd.DataFrame(data)

    # Convert config dict to columns
    config_df = pd.json_normalize(df['config'])
    df = pd.concat([df.drop('config', axis=1), config_df], axis=1)

    # 1. Throughput vs Batch Size (if available)
    throughput_data = df[df['metric_name'] == 'throughput']
    if not throughput_data.empty:
        plt.figure(figsize=(10, 6))
        sns.scatterplot(data=throughput_data, x='device_batch_size', y='metric_value', hue='compile', style='compile_dynamic', s=100)
        plt.title('Throughput vs Batch Size')
        plt.xlabel('Batch Size')
        plt.ylabel('Tokens/sec')
        plt.grid(True)
        plt.savefig(f"{output_dir}/throughput_batch_size.png")
        plt.close()

    # 2. Loss Landscape (Matrix LR vs Embedding LR)
    loss_data = df[df['metric_name'] == 'loss']
    if not loss_data.empty:
        # Filter successful runs
        loss_data = loss_data[loss_data['status'] == 'success']

        if 'matrix_lr' in loss_data.columns and 'embedding_lr' in loss_data.columns:
            plt.figure(figsize=(10, 8))
            pivot_table = loss_data.pivot_table(index='embedding_lr', columns='matrix_lr', values='metric_value', aggfunc='min')
            sns.heatmap(pivot_table, annot=True, fmt=".4f", cmap="viridis_r")
            plt.title('Loss Landscape: Matrix LR vs Embedding LR')
            plt.xlabel('Matrix LR')
            plt.ylabel('Embedding LR')
            plt.savefig(f"{output_dir}/loss_landscape_lr.png")
            plt.close()

        # 3. Loss vs Trial ID (Convergence over time)
        plt.figure(figsize=(12, 6))
        # Add trial index
        loss_data['trial_id'] = range(len(loss_data))
        sns.lineplot(data=loss_data, x='trial_id', y='metric_value', marker='o')
        plt.title('Loss Improvement over Trials')
        plt.xlabel('Trial Sequence')
        plt.ylabel('Loss')
        plt.grid(True)
        plt.savefig(f"{output_dir}/loss_history.png")
        plt.close()

        # 4. Layer Decay Impact
        if 'layer_lr_decay' in loss_data.columns and loss_data['layer_lr_decay'].nunique() > 1:
             plt.figure(figsize=(8, 6))
             sns.boxplot(data=loss_data, x='layer_lr_decay', y='metric_value')
             plt.title('Impact of Layer LR Decay on Loss')
             plt.xlabel('Layer Decay Factor')
             plt.ylabel('Loss')
             plt.grid(True)
             plt.savefig(f"{output_dir}/loss_layer_decay.png")
             plt.close()

    print(f"Plots saved to {output_dir}/")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Visualize tuning results")
    parser.add_argument("--history", type=str, default="tuning_history.json", help="Path to history JSON")
    parser.add_argument("--output", type=str, default="tuning_plots", help="Output directory")
    args = parser.parse_args()

    try:
        plot_tuning_results(args.history, args.output)
    except Exception as e:
        print(f"Error generating plots: {e}")
