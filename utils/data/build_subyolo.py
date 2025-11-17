#!/usr/bin/env python3

import os
import yaml
import argparse
import shutil
from pathlib import Path
from tqdm import tqdm


def load_config(config_path):
    """Load configuration from YAML file."""
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)


def validate_arguments(annotated_params, non_annotated_params):
    """Validate sampling parameters."""
    for param_name, params in [('annotated', annotated_params), ('non_annotated', non_annotated_params)]:
        if len(params) != 2:
            raise ValueError(f"{param_name} must have exactly 2 elements: [keep_frames, interval_frames]")

        keep_frames, interval_frames = params

        if not isinstance(keep_frames, int) or not isinstance(interval_frames, int):
            raise ValueError(f"{param_name} parameters must be integers")

        if keep_frames < 0 or interval_frames < 0:
            raise ValueError(f"{param_name} parameters must be non-negative")

        if interval_frames == 0 and keep_frames > 0:
            raise ValueError(f"{param_name}: interval_frames cannot be 0 when keep_frames > 0")


def validate_source_dataset(source_dirs):
    """Validate that source dataset exists and has required structure."""
    for dir_name, dir_path in source_dirs.items():
        if not os.path.exists(dir_path):
            raise FileNotFoundError(f"Source {dir_name} directory not found: {dir_path}")

        if not os.path.isdir(dir_path):
            raise NotADirectoryError(f"Source {dir_name} path is not a directory: {dir_path}")


def calculate_sample_frames(frame_indices, keep_frames, interval_frames):
    """
    Calculate which frames to keep based on [keep, interval] logic.

    Args:
        frame_indices: Sorted list of available frame numbers
        keep_frames: Number of frames to keep in each interval (K)
        interval_frames: Size of each interval (N)

    Returns:
        List of frame numbers to keep
    """
    if keep_frames == 0 or interval_frames == 0:
        return []

    selected_frames = []

    # Group frames into intervals of size N
    for i in range(0, len(frame_indices), interval_frames):
        interval = frame_indices[i:i + interval_frames]

        # Select K frames equidistant within this interval
        if len(interval) >= keep_frames:
            # Calculate equidistant positions
            if keep_frames == 1:
                selected_frames.append(interval[0])
            else:
                step = (len(interval) - 1) / (keep_frames - 1)
                for j in range(keep_frames):
                    idx = int(j * step)
                    selected_frames.append(interval[idx])
        else:
            # If interval has fewer frames than K, keep all of them
            selected_frames.extend(interval)

    return selected_frames


def analyze_yolo_dataset(images_dir, labels_dir):
    """
    Analyze existing YOLO dataset to identify annotated vs non-annotated frames.

    Returns:
        Tuple of (annotated_files, non_annotated_files, frame_mapping)
    """
    annotated_files = []
    non_annotated_files = []
    frame_mapping = {}

    # Get all image files
    image_files = []
    for ext in ['jpg', 'jpeg', 'png']:
        image_files.extend(Path(images_dir).glob(f"*.{ext}"))

    for img_path in tqdm(image_files, desc="Analyzing dataset"):
        # Parse filename to extract video_id and frame_number
        filename = img_path.stem

        # Extract video_id and frame_number using the same pattern as build_yolo.py
        if '_frame_' in filename:
            video_id, frame_part = filename.split('_frame_', 1)
            frame_number = int(frame_part)

            # Check if corresponding label file exists and has content
            label_path = Path(labels_dir) / f"{filename}.txt"

            # Determine if annotated
            is_annotated = False
            if label_path.exists():
                with open(label_path, 'r') as f:
                    content = f.read().strip()
                    is_annotated = bool(content)

            # Organize by video
            if video_id not in frame_mapping:
                frame_mapping[video_id] = {'annotated': [], 'non_annotated': []}

            if is_annotated:
                annotated_files.append(img_path)
                frame_mapping[video_id]['annotated'].append(frame_number)
            else:
                non_annotated_files.append(img_path)
                frame_mapping[video_id]['non_annotated'].append(frame_number)

    return annotated_files, non_annotated_files, frame_mapping


def create_subset(frame_mapping, annotated_params, non_annotated_params,
                  source_dirs, target_dirs, config):
    """
    Create subset based on sampling parameters.

    Args:
        frame_mapping: Dictionary mapping video_id to frame lists
        annotated_params: [keep_frames, interval_frames] for annotated frames
        non_annotated_params: [keep_frames, interval_frames] for non-annotated frames
        source_dirs: Dictionary with paths to source images/labels
        target_dirs: Dictionary with paths to target images/labels
        config: Configuration dictionary
    """
    total_selected = 0

    # Create target directories
    for dir_path in target_dirs.values():
        os.makedirs(dir_path, exist_ok=True)

    # Create splits directory
    splits_dir = target_dirs['images'].parent / 'splits'
    os.makedirs(splits_dir, exist_ok=True)

    # Process each video
    for video_id, frames_data in tqdm(frame_mapping.items(), desc="Processing videos"):
        # Sort frames within each category
        annotated_frames = sorted(frames_data['annotated'])
        non_annotated_frames = sorted(frames_data['non_annotated'])

        # Calculate which frames to keep
        selected_annotated = calculate_sample_frames(
            annotated_frames, annotated_params[0], annotated_params[1]
        )
        selected_non_annotated = calculate_sample_frames(
            non_annotated_frames, non_annotated_params[0], non_annotated_params[1]
        )

        # Copy selected files
        for frame_num in selected_annotated + selected_non_annotated:
            filename_base = f"{video_id}_frame_{frame_num:06d}"

            # Copy image
            for ext in ['jpg', 'jpeg', 'png']:
                src_img = Path(source_dirs['images']) / f"{filename_base}.{ext}"
                if src_img.exists():
                    dst_img = Path(target_dirs['images']) / src_img.name
                    shutil.copy2(src_img, dst_img)
                    break

            # Copy label
            src_label = Path(source_dirs['labels']) / f"{filename_base}.txt"
            if src_label.exists():
                dst_label = Path(target_dirs['labels']) / src_label.name
                shutil.copy2(src_label, dst_label)

        total_selected += len(selected_annotated) + len(selected_non_annotated)

    return total_selected


def main():
    """Main function to create subset from existing YOLO dataset."""
    parser = argparse.ArgumentParser(description='Create subset from existing YOLO dataset')
    parser.add_argument('--config', required=True, help='Path to data.yaml config file')
    parser.add_argument('--annotated', nargs=2, type=int, metavar=('KEEP', 'INTERVAL'),
                       help='Annotated frames: keep K frames every N frames [keep, interval]')
    parser.add_argument('--non_annotated', nargs=2, type=int, metavar=('KEEP', 'INTERVAL'),
                       help='Non-annotated frames: keep K frames every N frames [keep, interval]')

    args = parser.parse_args()

    try:
        # Validate arguments
        validate_arguments(args.annotated, args.non_annotated)

        # Load configuration
        config = load_config(args.config)

        # Construct source and target paths
        source_base = Path(config['paths']['target_dir']) / config['paths']['target_name']
        source_dirs = {
            'images': source_base / 'images',
            'labels': source_base / 'labels'
        }

        target_base = Path(config['paths']['target_subset_dir']) / config['paths']['target_subset_name']
        target_dirs = {
            'images': target_base / 'images',
            'labels': target_base / 'labels'
        }

        # Validate source dataset
        validate_source_dataset(source_dirs)

        # Analyze existing dataset
        print("Analyzing existing YOLO dataset...")
        annotated_files, non_annotated_files, frame_mapping = analyze_yolo_dataset(
            source_dirs['images'], source_dirs['labels']
        )

        print(f"Found {len(annotated_files)} annotated frames and {len(non_annotated_files)} non-annotated frames")

        # Create subset
        print("Creating subset...")
        total_selected = create_subset(
            frame_mapping, args.annotated, args.non_annotated,
            source_dirs, target_dirs, config
        )

        print(f"\nSubset creation complete!")
        print(f"Selected {total_selected} frames")
        print(f"Output directory: {target_base.absolute()}")
        print(f"Sampling parameters:")
        print(f"  Annotated: keep {args.annotated[0]} frames every {args.annotated[1]} frames")
        print(f"  Non-annotated: keep {args.non_annotated[0]} frames every {args.non_annotated[1]} frames")

    except Exception as e:
        print(f"Error: {e}")
        return 1

    return 0


if __name__ == "__main__":
    exit(main())