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


def validate_arguments(method, ratio, reversed_flag=False, annotations_path=None):
    """Validate command line arguments."""
    valid_methods = ['random_ratio', 'ood50', 'ood50flex']
    if method not in valid_methods:
        raise ValueError(f"Invalid method '{method}'. Must be one of: {valid_methods}")

    # --reversed flag only valid for ood50 method
    if reversed_flag and method != 'ood50':
        raise ValueError(f"--reversed flag is only valid with ood50 method, not with {method}")

    # --annotations path required for ood50flex method
    if method == 'ood50flex' and not annotations_path:
        raise ValueError("--annotations path is required when using ood50flex method")

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


def load_annotation_counts(annotation_file):
    """Load annotation file and return dict of {video_id: annotation_count}."""
    if not os.path.exists(annotation_file):
        raise FileNotFoundError(f"Annotation file not found: {annotation_file}")

    import json
    with open(annotation_file, 'r') as f:
        annotations = json.load(f)

    annotation_counts = {}
    for video_data in annotations:
        video_id = video_data.get('video_id')
        if not video_id:
            continue

        # Count all bounding boxes across all annotation intervals
        total_bboxes = 0
        video_annotations = video_data.get('annotations', [])

        for annotation_interval in video_annotations:
            bboxes = annotation_interval.get('bboxes', [])
            total_bboxes += len(bboxes)

        annotation_counts[video_id] = total_bboxes

    return annotation_counts


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


def split_ood50flex(image_paths, annotation_counts):
    """Split image paths based on annotation count comparison between _0 and _1 video pairs."""
    # Group video IDs by base name (e.g., 'Backpack_0' + 'Backpack_1' -> 'Backpack')
    video_pairs = {}
    unpaired_videos = []

    # Extract unique video IDs from image paths
    video_ids_from_images = set()
    for image_path in image_paths:
        filename = Path(image_path).stem
        video_id = parse_video_id_from_filename(filename)
        video_ids_from_images.add(video_id)

    # Build video pairs
    for video_id in video_ids_from_images:
        if video_id.endswith('_0'):
            base_name = video_id[:-2]  # Remove '_0' suffix
            if base_name not in video_pairs:
                video_pairs[base_name] = {'_0': video_id, '_1': None}
            else:
                video_pairs[base_name]['_0'] = video_id
        elif video_id.endswith('_1'):
            base_name = video_id[:-2]  # Remove '_1' suffix
            if base_name not in video_pairs:
                video_pairs[base_name] = {'_0': None, '_1': video_id}
            else:
                video_pairs[base_name]['_1'] = video_id
        else:
            # Videos without _0 or _1 suffix go to train by default
            unpaired_videos.append(video_id)

    # Determine train/val assignment based on annotation counts
    train_videos = set()
    val_videos = set()

    # Process paired videos
    for base_name, pair in video_pairs.items():
        video_0 = pair['_0']
        video_1 = pair['_1']

        count_0 = annotation_counts.get(video_0, 0) if video_0 else 0
        count_1 = annotation_counts.get(video_1, 0) if video_1 else 0

        # Video with MORE annotations goes to train, fewer to val
        if count_0 >= count_1:
            if video_0:
                train_videos.add(video_0)
            if video_1:
                val_videos.add(video_1)
        else:
            if video_1:
                train_videos.add(video_1)
            if video_0:
                val_videos.add(video_0)

    # Unpaired videos (missing _0 or _1) go to train by default
    train_videos.update(unpaired_videos)

    # Assign image paths based on video assignment
    train_paths = []
    val_paths = []

    for image_path in image_paths:
        filename = Path(image_path).stem
        video_id = parse_video_id_from_filename(filename)

        if video_id in train_videos:
            train_paths.append(image_path)
        elif video_id in val_videos:
            val_paths.append(image_path)
        else:
            # Default to train for any unspecified videos
            train_paths.append(image_path)

    splits = {
        'train': train_paths,
        'val': val_paths
    }

    # Print assignment summary
    print("ood50flex assignment summary:")
    for base_name, pair in video_pairs.items():
        video_0 = pair['_0']
        video_1 = pair['_1']
        count_0 = annotation_counts.get(video_0, 0) if video_0 else 0
        count_1 = annotation_counts.get(video_1, 0) if video_1 else 0

        if count_0 >= count_1:
            train_vid = video_0 if video_0 else "None"
            val_vid = video_1 if video_1 else "None"
        else:
            train_vid = video_1 if video_1 else "None"
            val_vid = video_0 if video_0 else "None"

        print(f"  {base_name}: train={train_vid} (ann={max(count_0, count_1)}), val={val_vid} (ann={min(count_0, count_1)})")

    print(f"  Unpaired videos: {len(unpaired_videos)} → train")

    return splits


def generate_dataset_yaml(splits, split_dir, config):
    """Generate YOLO dataset.yaml file with train/val paths and class information."""
    split_dir = Path(split_dir)

    # split_dir is the dataset root where images/, labels/, and splits/ folders exist
    dataset_root = split_dir

    # Relative paths from dataset root to split files
    train_path = "splits/train.txt" if "train" in splits else None
    val_path = "splits/val.txt" if "val" in splits else None
    test_path = "splits/test.txt" if "test" in splits else None

    # Extract class information from config
    class_mapping = config.get('class_mapping', {})

    # Create class names dictionary sorted by class ID
    class_names = {}
    for class_name, class_info in class_mapping.items():
        class_id = class_info.get('new_id')
        new_name = class_info.get('new_name', class_name.lower())
        if class_id is not None:
            class_names[class_id] = new_name

    # Sort by class ID to ensure consistent ordering
    sorted_classes = sorted(class_names.items())

    # Create dataset.yaml content
    dataset_yaml = {
        'path': str(dataset_root.resolve()),  # Absolute path to dataset root
        'train': train_path,
        'val': val_path,
        'test': test_path if test_path else None,
        'nc': len(sorted_classes),
        'names': {class_id: class_name for class_id, class_name in sorted_classes}
    }

    # Remove None values
    dataset_yaml = {k: v for k, v in dataset_yaml.items() if v is not None}

    # Write dataset.yaml file in the dataset root (split_dir), not in splits subdirectory
    yaml_file = dataset_root / 'dataset.yaml'
    with open(yaml_file, 'w') as f:
        yaml.dump(dataset_yaml, f, default_flow_style=False, sort_keys=False)

    print(f"Dataset configuration saved to: {yaml_file}")
    print(f"Classes: {dataset_yaml['nc']}")
    print(f"Class names: {list(dataset_yaml['names'].values())}")

    return yaml_file


def save_split_files(splits, splits_output_dir, config=None):
    """Save split files to text files with absolute paths."""
    splits_output_dir = Path(splits_output_dir)
    splits_output_dir.mkdir(parents=True, exist_ok=True)

    for split_name, paths in splits.items():
        output_file = splits_output_dir / f"{split_name}.txt"

        with open(output_file, 'w') as f:
            for path in paths:
                f.write(f"{path}\n")

    # Print summary
    for split_name, paths in splits.items():
        print(f"{split_name.capitalize()}: {len(paths)} images")

    print(f"\nSplit files saved to: {splits_output_dir}")

    # Generate dataset.yaml if config is provided
    # dataset.yaml should be saved in the dataset root (parent of splits directory)
    if config is not None:
        dataset_root = splits_output_dir.parent  # Go up one level from splits/ to dataset root
        generate_dataset_yaml(splits, dataset_root, config)


def main():
    parser = argparse.ArgumentParser(description='Create train/val/test splits for YOLO dataset')
    parser.add_argument('--config', default='configs/data.yaml',
                       help='Path to configuration file')
    parser.add_argument('--method', choices=['random_ratio', 'ood50', 'ood50flex'], default='ood50',
                       help='Splitting method: random_ratio, ood50, or ood50flex')
    parser.add_argument('--ratio', nargs='+', type=float,
                       help='Split ratios as list of percentages (e.g., 70 30 or 70 20 10)')
    parser.add_argument('--reversed', action='store_true',
                       help='Reverse train/val assignment for ood50 method')
    parser.add_argument('--annotations',
                       help='Path to annotations JSON file (required for ood50flex method, e.g., dataset/raw/train/annotations/annotations.json)')

    args = parser.parse_args()

    try:
        # Load configuration
        config = load_config(args.config)

        # Validate arguments
        validate_arguments(args.method, args.ratio, args.reversed, args.annotations)

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

        elif args.method == 'ood50flex':
            # Load annotation counts for intelligent splitting from provided path
            annotation_file = Path(args.annotations)
            print(f"Loading annotation counts from: {annotation_file}")
            annotation_counts = load_annotation_counts(annotation_file)
            print(f"Loaded annotation counts for {len(annotation_counts)} videos")

            print(f"Splitting with ood50flex method (annotation-based comparison)")
            splits = split_ood50flex(image_paths, annotation_counts)

        # Save split files and generate dataset.yaml
        save_split_files(splits, splits_output_dir, config)

        return 0

    except Exception as e:
        print(f"Error: {e}")
        return 1


if __name__ == "__main__":
    exit(main())