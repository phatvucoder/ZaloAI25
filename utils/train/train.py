#!/usr/bin/env python3

import os
import sys
import yaml
import argparse
from pathlib import Path

# Add project root to Python path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

try:
    from ultralytics import YOLO
except ImportError:
    print("Error: ultralytics package not found. Please install with: pip install ultralytics")
    sys.exit(1)


def load_config(config_path):
    """Load configuration from YAML file."""
    try:
        with open(config_path, 'r') as f:
            return yaml.safe_load(f)
    except FileNotFoundError:
        print(f"Error: Config file not found: {config_path}")
        sys.exit(1)
    except yaml.YAMLError as e:
        print(f"Error: Invalid YAML in config file: {e}")
        sys.exit(1)


def main():
    parser = argparse.ArgumentParser(description='Train YOLO model with configuration file')
    parser.add_argument('--model', type=str, required=True,
                        help='Path to model file (.pt) or model name (yolov8n, yolov8s, etc.)')
    parser.add_argument('--config', type=str, default='configs/train.yaml',
                        help='Path to training configuration file')

    args = parser.parse_args()

    # Load configuration
    config = load_config(args.config)

    # Validate that dataset.yaml exists
    data_path = config.get('data')
    if data_path and not os.path.exists(data_path):
        print(f"Warning: Dataset file not found: {data_path}")
        print("Please make sure you have created the dataset first using:")
        print("python -m utils.data.build_yolo --config configs/data.yaml")

    # Initialize model
    try:
        print(f"Loading model: {args.model}")
        model = YOLO(args.model)
    except Exception as e:
        print(f"Error: Failed to load model: {e}")
        sys.exit(1)

    # Print training parameters
    print(f"Training with parameters:")
    for key, value in config.items():
        print(f"  {key}: {value}")

    # Start training with kwargs
    try:
        print("\nStarting training...")
        results = model.train(**config)

        print("Training completed successfully!")
        print(f"Results saved to: runs/train/{config.get('name', 'exp')}")

        # Print final metrics
        if hasattr(results, 'results_dict'):
            print("\nFinal training metrics:")
            for metric, value in results.results_dict.items():
                print(f"  {metric}: {value:.4f}")

    except Exception as e:
        print(f"Error during training: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()