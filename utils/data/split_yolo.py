#!/usr/bin/env python3

import os
import yaml
import argparse
import random
from pathlib import Path


def load_config(config_path):
    """Load configuration from YAML file."""
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Configuration file not found: {config_path}")

    with open(config_path, 'r') as f:
        return yaml.safe_load(f)


def validate_arguments(method, ratio):
    """Validate command line arguments."""
    valid_methods = ['random_ratio', 'ood50']
    if method not in valid_methods:
        raise ValueError(f"Invalid method '{method}'. Must be one of: {valid_methods}")

    if ratio is not None:
        if not isinstance(ratio, (list, tuple)):
            raise ValueError("Ratio must be a list or tuple")

        if len(ratio) not in [2, 3]:
            raise ValueError("Ratio must have 2 elements (train, val) or 3 elements (train, val, test)")

        if not all(isinstance(x, (int, float)) for x in ratio):
            raise ValueError("All ratio elements must be numbers")

        if not all(x > 0 for x in ratio):
            raise ValueError("All ratio elements must be positive")

        if abs(sum(ratio) - 100) > 0.001:
            raise ValueError(f"Ratio elements must sum to 100, got {sum(ratio)}")


def collect_image_paths(images_dir):
    """Collect all image file paths from the images directory."""
    images_dir = Path(images_dir)
    if not images_dir.exists():
        raise FileNotFoundError(f"Images directory not found: {images_dir}")

    image_paths = []
    supported_exts = ['jpg', 'jpeg', 'png']

    for ext in supported_exts:
        image_paths.extend(images_dir.glob(f"*.{ext}"))
        image_paths.extend(images_dir.glob(f"*.{ext.upper()}"))

    if not image_paths:
        raise ValueError(f"No image files found in {images_dir}")

    return sorted([str(p.resolve()) for p in image_paths])


def parse_video_id_from_filename(filename):
    """Extract video_id from filename by removing frame suffix."""
    if '_frame_' in filename:
        video_id = filename.split('_frame_')[0]
        return video_id
    return filename


def split_random_ratio(image_paths, ratio):
    """Split image paths randomly based on ratio."""
    random.shuffle(image_paths)

    if len(ratio) == 2:
        train_ratio, val_ratio = ratio
        test_ratio = 0
    else:
        train_ratio, val_ratio, test_ratio = ratio

    total_count = len(image_paths)
    train_count = int(total_count * train_ratio / 100)
    val_count = int(total_count * val_ratio / 100)
    test_count = total_count - train_count - val_count

    splits = {
        'train': image_paths[:train_count],
        'val': image_paths[train_count:train_count + val_count]
    }

    if test_count > 0:
        splits['test'] = image_paths[train_count + val_count:]

    return splits


def split_ood50(image_paths, reversed_flag=False):
    """Split image paths based on video_id suffix (0 for train, 1 for val, or reversed)."""
    train_paths = []
    val_paths = []

    for image_path in image_paths:
        filename = Path(image_path).stem
        video_id = parse_video_id_from_filename(filename)

        if video_id.endswith('_0'):
            split = 'val' if reversed_flag else 'train'
        elif video_id.endswith('_1'):
            split = 'train' if reversed_flag else 'val'
        else:
            # Default to train if no clear suffix
            split = 'train'

        if split == 'train':
            train_paths.append(image_path)
        else:
            val_paths.append(image_path)

    splits = {
        'train': train_paths,
        'val': val_paths
    }

    return splits


def save_split_files(splits, output_dir):
    """Save split files to text files with absolute paths."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    for split_name, paths in splits.items():
        output_file = output_dir / f"{split_name}.txt"

        with open(output_file, 'w') as f:
            for path in paths:
                f.write(f"{path}\n")

    # Print summary
    for split_name, paths in splits.items():
        print(f"{split_name.capitalize()}: {len(paths)} images")

    print(f"\nSplit files saved to: {output_dir}")


def main():
    parser = argparse.ArgumentParser(description='Create train/val/test splits for YOLO dataset')
    parser.add_argument('--config', default='configs/data.yaml',
                       help='Path to configuration file')
    parser.add_argument('--method', choices=['random_ratio', 'ood50'], default='ood50',
                       help='Splitting method: random_ratio or ood50')
    parser.add_argument('--ratio', nargs='+', type=float,
                       help='Split ratios as list of percentages (e.g., 70 30 or 70 20 10)')
    parser.add_argument('--reversed', action='store_true',
                       help='Reverse train/val assignment for ood50 method')

    args = parser.parse_args()

    try:
        # Load configuration
        config = load_config(args.config)

        # Validate arguments
        validate_arguments(args.method, args.ratio)

        # Get paths from config
        split_dir = Path(config['paths']['split_dir'])
        images_dir = split_dir / 'images'
        splits_output_dir = split_dir / 'splits'

        print(f"Loading image paths from: {images_dir}")
        image_paths = collect_image_paths(images_dir)
        print(f"Found {len(image_paths)} images")

        # Perform splitting
        if args.method == 'random_ratio':
            if args.ratio is None:
                raise ValueError("Ratio must be specified when using random_ratio method")
            print(f"Splitting with random ratios: {args.ratio}")
            splits = split_random_ratio(image_paths, args.ratio)

        elif args.method == 'ood50':
            print(f"Splitting with ood50 method (reversed={args.reversed})")
            splits = split_ood50(image_paths, args.reversed)

        # Save split files
        save_split_files(splits, splits_output_dir)

        return 0

    except Exception as e:
        print(f"Error: {e}")
        return 1


if __name__ == "__main__":
    exit(main())