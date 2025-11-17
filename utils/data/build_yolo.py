#!/usr/bin/env python3

import os
import json
import cv2
import yaml
import argparse
import shutil
import threading
from pathlib import Path
from tqdm import tqdm

NUMPY_AVAILABLE = False
CONCURRENT_AVAILABLE = False

# Check for optional dependencies
try:
    import numpy
    NUMPY_AVAILABLE = True
except ImportError:
    pass

try:
    from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor, as_completed
    CONCURRENT_AVAILABLE = True
except ImportError:
    pass


def load_config(config_path):
    """Load configuration from YAML file."""
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)


def find_video_file(video_dir, video_extensions):
    """Find video file in directory based on supported extensions."""
    if not os.path.exists(video_dir):
        return None

    for ext in video_extensions:
        for file in os.listdir(video_dir):
            if file.lower().endswith(ext.lower()):
                return os.path.join(video_dir, file)
    return None


def get_class_mapping(config, video_id):
    """Extract class mapping from video ID and configuration."""
    base_name = video_id.rsplit('_', 1)[0]
    class_info = config['class_mapping'].get(base_name)

    if not class_info:
        raise ValueError(f"No class mapping found for video_id: {video_id}")

    return class_info


def extract_annotations(video_record):
    """Extract annotations from video record into a frame-to-boxes dictionary."""
    annotations_dict = {}

    for interval in video_record.get('annotations', []):
        for bbox_data in interval.get('bboxes', []):
            frame_num = int(bbox_data['frame'])
            box = (int(bbox_data['x1']), int(bbox_data['y1']),
                   int(bbox_data['x2']), int(bbox_data['y2']))

            if frame_num not in annotations_dict:
                annotations_dict[frame_num] = []
            annotations_dict[frame_num].append(box)

    return annotations_dict


def get_video_fps(cap):
    """Safely detect video FPS with fallback."""
    try:
        fps = cap.get(cv2.CAP_PROP_FPS)
        if fps <= 0 or fps != fps:  # Check for invalid or NaN values
            return None
        return fps
    except Exception:
        return None


def calculate_frame_interval(original_fps, target_fps):
    """Calculate frame sampling interval based on target FPS."""
    if target_fps is None:
        return 1  # No sampling when FPS is None (original FPS)

    if target_fps == 0:
        return float('inf')  # Return infinite interval to skip all frames

    if target_fps < 0:
        raise ValueError("FPS cannot be negative")

    if original_fps is None or original_fps <= 0:
        return 1  # Can't calculate without original FPS, use no sampling

    if target_fps >= original_fps:
        return 1  # Target FPS is higher or equal to original, no sampling needed

    interval = int(original_fps / target_fps)
    return max(1, interval)  # Ensure minimum interval of 1


def calculate_hybrid_frame_indices(total_frames, original_fps, annotated_fps, non_annotated_fps,
                                 annotated_frames_set, ensure_annotated_frames=True):
    """
    Calculate which frames to extract based on hybrid FPS configuration.

    Args:
        total_frames: Total number of frames in the video
        original_fps: Original video FPS (may be None)
        annotated_fps: Target FPS for annotated frames (None = original)
        non_annotated_fps: Target FPS for non-annotated frames
        annotated_frames_set: Set of frame numbers that have annotations
        ensure_annotated_frames: If True, never skip annotated frames

    Returns:
        Sorted list of frame numbers to extract
    """
    selected_frames = set()

    # Always include annotated frames if required
    if ensure_annotated_frames:
        selected_frames.update(annotated_frames_set)

    # Calculate sampling intervals
    annotated_interval = calculate_frame_interval(original_fps, annotated_fps)
    non_annotated_interval = calculate_frame_interval(original_fps, non_annotated_fps)

    # Sample annotated frames at the specified FPS (if not already included)
    if not ensure_annotated_frames:
        annotated_frames_list = sorted(annotated_frames_set)
        for i, frame_num in enumerate(annotated_frames_list):
            if i % annotated_interval == 0:
                selected_frames.add(frame_num)

    # Sample non-annotated frames
    if non_annotated_interval != float('inf'):  # Only sample if not skipping completely
        for frame_num in range(0, total_frames, non_annotated_interval):
            if frame_num not in annotated_frames_set:
                selected_frames.add(frame_num)

    return sorted(selected_frames)


def convert_to_yolo_format(box, img_width, img_height):
    """Convert bounding box coordinates to YOLO format."""
    x1, y1, x2, y2 = float(box[0]), float(box[1]), float(box[2]), float(box[3])

    # Calculate center and dimensions
    x_center = (x1 + x2) / 2.0
    y_center = (y1 + y2) / 2.0
    width = x2 - x1
    height = y2 - y1

    # Normalize to [0, 1]
    x_center_norm = x_center / img_width
    y_center_norm = y_center / img_height
    width_norm = width / img_width
    height_norm = height / img_height

    return x_center_norm, y_center_norm, width_norm, height_norm


def convert_boxes_to_yolo_vectorized(boxes, video_width, video_height, dtype):
    """Vectorized YOLO coordinate conversion for multiple boxes."""
    import numpy as np

    box_array = np.array(boxes, dtype=dtype)
    x1, y1, x2, y2 = box_array[:, 0], box_array[:, 1], box_array[:, 2], box_array[:, 3]

    # Calculate center coordinates and dimensions
    x_center = (x1 + x2) / 2.0 / video_width
    y_center = (y1 + y2) / 2.0 / video_height
    width = (x2 - x1) / video_width
    height = (y2 - y1) / video_height

    return x_center, y_center, width, height


def write_yolo_labels(file_path, class_id, boxes, video_width, video_height,
                     use_numpy=False, vectorized_conversion=False, dtype="float32"):
    """Write YOLO format labels to file."""
    with open(file_path, 'w') as f:
        if not boxes:
            return  # Empty file for frames with no annotations

        if use_numpy and vectorized_conversion:
            x_center, y_center, width, height = convert_boxes_to_yolo_vectorized(
                boxes, video_width, video_height, dtype)

            for i in range(len(x_center)):
                f.write(f"{class_id} {x_center[i]:.6f} {y_center[i]:.6f} {width[i]:.6f} {height[i]:.6f}\n")
        else:
            for box in boxes:
                x_center_norm, y_center_norm, width_norm, height_norm = convert_to_yolo_format(
                    box, video_width, video_height)
                f.write(f"{class_id} {x_center_norm:.6f} {y_center_norm:.6f} {width_norm:.6f} {height_norm:.6f}\n")


def process_frames_video(video_record, output_dirs, config, class_info,
                        use_numpy=False, vectorized_conversion=False, dtype="float32",
                        print_sub_tqdm=False, batch_size=32):
    """Process video frames - either all frames or annotated frames only."""
    video_id = video_record['video_id']
    class_id = class_info['new_id']

    # Construct video path
    video_dir = os.path.join(config['paths']['train_data'], 'samples', video_id)
    video_path = find_video_file(video_dir, config['files']['video_ext'])

    if not video_path:
        print(f"Warning: Video file not found for {video_id}")
        return

    # Extract annotations
    annotations_dict = extract_annotations(video_record)

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Error: Could not open video: {video_path}")
        return

    try:
        video_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        video_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        # Determine extraction mode
        video_config = config.get('video', {})
        extraction_mode = video_config.get('extraction_mode', 'annotated_only')

        # Common processing parameters
        process_params = (cap, video_id, annotations_dict, output_dirs, config,
                         class_id, video_width, video_height, use_numpy,
                         vectorized_conversion, dtype, print_sub_tqdm, batch_size)

        if extraction_mode == 'legacy':
            # Legacy: use extract_all_frames setting
            if video_config.get('extract_all_frames', False):
                _process_all_frames(*process_params)
            else:
                _process_annotated_frames(*process_params)
        elif extraction_mode == 'all':
            _process_all_frames(*process_params)
        elif extraction_mode == 'annotated_only':
            _process_annotated_frames(*process_params)
        elif extraction_mode == 'non_annotated_only':
            _process_non_annotated_frames(*process_params)
        elif extraction_mode == 'hybrid':
            hybrid_fps = video_config.get('hybrid_fps', {})
            if hybrid_fps.get('enabled', False):
                _process_hybrid_frames(*process_params)
            else:
                print(f"Warning: hybrid mode enabled but hybrid_fps.disabled for {video_id}. Using annotated_only.")
                _process_annotated_frames(*process_params)
        else:
            print(f"Warning: Unknown extraction mode '{extraction_mode}' for {video_id}. Using annotated_only.")
            _process_annotated_frames(*process_params)
    finally:
        cap.release()


def _process_all_frames(cap, video_id, annotations_dict, output_dirs, config,
                       class_id, video_width, video_height, use_numpy,
                       vectorized_conversion, dtype, print_sub_tqdm, batch_size):
    """Process all frames from video."""
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    pbar = None
    if print_sub_tqdm:
        pbar = tqdm(total=total_frames, desc=f"Processing {video_id}")

    processed_frames = 0

    for frame_num in range(total_frames):
        ret, frame = cap.read()
        if not ret:
            break

        # Save frame and labels
        boxes = annotations_dict.get(frame_num, [])
        _save_frame_and_labels(frame_num, video_id, frame, boxes, output_dirs, config,
                             class_id, video_width, video_height, use_numpy,
                             vectorized_conversion, dtype)

        processed_frames += 1

        # Update progress
        if print_sub_tqdm and pbar:
            pbar.update(1)
            if processed_frames % batch_size == 0:
                pbar.set_postfix({"batch": f"{processed_frames//batch_size} ({processed_frames}/{total_frames})"})

    if pbar:
        pbar.close()


def _process_annotated_frames(cap, video_id, annotations_dict, output_dirs, config,
                             class_id, video_width, video_height, use_numpy,
                             vectorized_conversion, dtype, print_sub_tqdm, batch_size):
    """Process only annotated frames."""
    total_frames = len(annotations_dict)

    pbar = None
    if print_sub_tqdm:
        pbar = tqdm(total=total_frames, desc=f"Processing {video_id} (annotated)")

    processed_frames = 0

    for frame_num, boxes in annotations_dict.items():
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_num)
        ret, frame = cap.read()
        if not ret:
            continue

        # Save frame and labels
        _save_frame_and_labels(frame_num, video_id, frame, boxes, output_dirs, config,
                             class_id, video_width, video_height, use_numpy,
                             vectorized_conversion, dtype)

        processed_frames += 1

        # Update progress
        if print_sub_tqdm and pbar:
            pbar.update(1)
            if processed_frames % batch_size == 0:
                pbar.set_postfix({"batch": f"{processed_frames//batch_size} ({processed_frames}/{total_frames})"})

    if pbar:
        pbar.close()


def _save_frame_and_labels(frame_num, video_id, frame, boxes, output_dirs, config,
                           class_id, video_width, video_height, use_numpy,
                           vectorized_conversion, dtype):
    """Save a single frame and its YOLO labels."""
    file_basename = f"{video_id}_frame_{frame_num:06d}"
    image_path = os.path.join(output_dirs['images'], f"{file_basename}.{config['files']['image_ext'][0]}")
    label_path = os.path.join(output_dirs['labels'], f"{file_basename}.txt")

    cv2.imwrite(image_path, frame)
    write_yolo_labels(label_path, class_id, boxes, video_width, video_height,
                    use_numpy, vectorized_conversion, dtype)


def _process_hybrid_frames(cap, video_id, annotations_dict, output_dirs, config,
                          class_id, video_width, video_height, use_numpy,
                          vectorized_conversion, dtype, print_sub_tqdm, batch_size):
    """Process frames using hybrid FPS configuration."""
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    original_fps = get_video_fps(cap)

    # Get hybrid FPS configuration
    hybrid_config = config.get('video', {}).get('hybrid_fps', {})
    annotated_fps = hybrid_config.get('annotated_fps')
    non_annotated_fps = hybrid_config.get('non_annotated_fps', 3)
    ensure_annotated_frames = hybrid_config.get('ensure_annotated_frames', True)

    # Get set of annotated frames
    annotated_frames_set = set(annotations_dict.keys())

    # Calculate which frames to process
    frames_to_process = calculate_hybrid_frame_indices(
        total_frames, original_fps, annotated_fps, non_annotated_fps,
        annotated_frames_set, ensure_annotated_frames
    )

    pbar = None
    if print_sub_tqdm:
        annotated_count = len([f for f in frames_to_process if f in annotated_frames_set])
        non_annotated_count = len(frames_to_process) - annotated_count
        desc = f"Processing {video_id} (hybrid: {annotated_count} ann, {non_annotated_count} empty)"
        pbar = tqdm(total=len(frames_to_process), desc=desc)

    processed_frames = 0
    last_frame_pos = None

    for frame_num in frames_to_process:
        # Only seek if frame position is far from current position
        if last_frame_pos is None or abs(frame_num - last_frame_pos) > 5:
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_num)

        ret, frame = cap.read()
        if not ret:
            # If reading failed, try seeking explicitly
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_num)
            ret, frame = cap.read()
            if not ret:
                continue

        last_frame_pos = frame_num

        # Save frame and labels
        boxes = annotations_dict.get(frame_num, [])
        _save_frame_and_labels(frame_num, video_id, frame, boxes, output_dirs, config,
                             class_id, video_width, video_height, use_numpy,
                             vectorized_conversion, dtype)

        processed_frames += 1

        # Update progress
        if print_sub_tqdm and pbar:
            pbar.update(1)
            if processed_frames % batch_size == 0:
                pbar.set_postfix({"batch": f"{processed_frames//batch_size} ({processed_frames}/{len(frames_to_process)})"})

    if pbar:
        pbar.close()

    # Print processing statistics
    original_annotated = len(annotated_frames_set)
    original_total = total_frames
    final_annotated = len([f for f in frames_to_process if f in annotated_frames_set])
    final_total = len(frames_to_process)

    print(f"{video_id} hybrid processing:")
    print(f"  Original: {original_annotated}/{original_total} frames ({original_annotated/total_frames*100:.1f}% annotated)")
    print(f"  Final: {final_annotated}/{final_total} frames ({final_annotated/final_total*100:.1f}% annotated)")
    print(f"  Size reduction: {(1 - final_total/original_total)*100:.1f}%")


def _process_non_annotated_frames(cap, video_id, annotations_dict, output_dirs, config,
                                 class_id, video_width, video_height, use_numpy,
                                 vectorized_conversion, dtype, print_sub_tqdm, batch_size):
    """Process only frames without annotations (empty frames)."""
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    # Create set of annotated frame numbers for quick lookup
    annotated_frames_set = set(annotations_dict.keys())

    # Generate list of non-annotated frames
    non_annotated_frames = [frame_num for frame_num in range(total_frames)
                           if frame_num not in annotated_frames_set]

    if not non_annotated_frames:
        print(f"Warning: {video_id} has no non-annotated frames to process")
        return

    pbar = None
    if print_sub_tqdm:
        pbar = tqdm(total=len(non_annotated_frames), desc=f"Processing {video_id} (non-annotated)")

    processed_frames = 0

    for frame_num in non_annotated_frames:
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_num)
        ret, frame = cap.read()
        if not ret:
            continue

        # Save frame without labels (empty labels file)
        file_basename = f"{video_id}_frame_{frame_num:06d}"
        image_path = os.path.join(output_dirs['images'], f"{file_basename}.{config['files']['image_ext'][0]}")
        label_path = os.path.join(output_dirs['labels'], f"{file_basename}.txt")

        cv2.imwrite(image_path, frame)

        # Create empty label file
        with open(label_path, 'w') as f:
            pass  # Empty file for frames without annotations

        processed_frames += 1

        # Update progress
        if print_sub_tqdm and pbar:
            pbar.update(1)
            if processed_frames % batch_size == 0:
                pbar.set_postfix({"batch": f"{processed_frames//batch_size} ({processed_frames}/{len(non_annotated_frames)})"})

    if pbar:
        pbar.close()

    # Print processing statistics
    original_annotated = len(annotated_frames_set)
    original_total = total_frames
    final_non_annotated = processed_frames

    print(f"{video_id} non-annotated-only processing:")
    print(f"  Original: {original_annotated}/{original_total} frames ({original_annotated/total_frames*100:.1f}% annotated)")
    print(f"  Processed: {final_non_annotated} non-annotated frames ({final_non_annotated/original_total*100:.1f}% of total)")


def copy_object_images_task(config, original_class, class_info):
    """Task to copy object images for a single class."""
    video_id = f"{original_class}_0"
    source_dir = os.path.join(config['paths']['train_data'], 'samples', video_id, 'object_images')
    base_dir = os.path.join(config['paths']['target_dir'], config['paths']['target_name'])
    target_dir = os.path.join(base_dir, 'objects', class_info['new_name'])

    if not os.path.exists(source_dir):
        return

    os.makedirs(target_dir, exist_ok=True)

    # Copy matching image files
    for img_file in os.listdir(source_dir):
        if any(img_file.lower().endswith(ext.lower()) for ext in config['files']['image_ext']):
            src_path = os.path.join(source_dir, img_file)
            dst_path = os.path.join(target_dir, img_file)
            if not os.path.exists(dst_path):
                shutil.copy2(src_path, dst_path)


class BaseProcessor:
    """Base class with common functionality for all processors."""

    def __init__(self, config):
        self.config = config

    def copy_object_images(self):
        """Copy object reference images using threading."""
        if not self.config['processing']['copy_object_images']:
            return

        threads = []
        for original_class, class_info in self.config['class_mapping'].items():
            thread = threading.Thread(target=copy_object_images_task,
                                    args=(self.config, original_class, class_info))
            thread.start()
            threads.append(thread)

        for thread in threads:
            thread.join()


class NormalModeProcessor(BaseProcessor):
    """Simple, compatible processor for most systems."""

    def __init__(self, config):
        super().__init__(config)
        self.normal_config = config.get('normal_mode', {})
        self.max_io_threads = self.normal_config.get('max_io_threads', 4)

    def process_videos(self, video_records, output_dirs):
        """Sequential video processing with basic optimizations."""
        print("Running in NORMAL mode - maximum compatibility")

        for video_record in tqdm(video_records, desc="Processing videos"):
            try:
                video_id = video_record['video_id']
                class_info = get_class_mapping(self.config, video_id)

                process_frames_video(video_record, output_dirs, self.config, class_info)

            except Exception as e:
                print(f"Error processing video {video_record.get('video_id', 'unknown')}: {e}")


def turbo_video_worker(video_record, output_dirs, config, batch_size, use_numpy,
                      vectorized_conversion, dtype, print_sub_tqdm):
    """Standalone worker function for turbo mode video processing to avoid pickling issues."""
    try:
        video_id = video_record['video_id']
        class_info = get_class_mapping(config, video_id)

        process_frames_video(video_record, output_dirs, config, class_info,
                           use_numpy, vectorized_conversion, dtype,
                           print_sub_tqdm, batch_size)

    except Exception as e:
        print(f"Error in turbo worker for {video_record.get('video_id', 'unknown')}: {e}")


class TurboModeProcessor(BaseProcessor):
    """High-performance processor using advanced techniques."""

    def __init__(self, config):
        super().__init__(config)

        if not NUMPY_AVAILABLE or not CONCURRENT_AVAILABLE:
            raise RuntimeError("Turbo mode requires NumPy and concurrent.futures")

        # Import required modules
        import numpy as np
        from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed
        self.np = np
        self.ProcessPoolExecutor = ProcessPoolExecutor
        self.ThreadPoolExecutor = ThreadPoolExecutor
        self.as_completed = as_completed

        # Load turbo mode settings
        turbo_config = config.get('turbo_mode', {})
        self.max_workers = turbo_config.get('max_workers')
        self.use_threading = turbo_config.get('use_threading', True)
        self.batch_size = turbo_config.get('batch_size', 32)
        self.use_numpy = turbo_config.get('use_numpy', True)
        self.vectorized_conversion = turbo_config.get('vectorized_conversion', True)
        self.dtype = getattr(np, turbo_config.get('dtype', 'float32'))
        self.max_file_workers = turbo_config.get('max_file_workers', 8)
        self.print_sub_tqdm = turbo_config.get('print_sub_tqdm', False)

    def process_videos(self, video_records, output_dirs):
        """Parallel video processing with advanced optimizations."""
        print("Running in TURBO mode - maximum performance")

        # Determine optimal number of workers
        if self.max_workers is None:
            import multiprocessing
            max_workers = min(multiprocessing.cpu_count(), len(video_records))
        else:
            max_workers = min(self.max_workers, len(video_records))

        print(f"Using {max_workers} parallel workers")
        print(f"Batch size: {self.batch_size}")
        print(f"NumPy optimizations: {self.use_numpy and self.vectorized_conversion}")

        executor_class = self.ThreadPoolExecutor if self.use_threading else self.ProcessPoolExecutor

        with executor_class(max_workers=max_workers) as executor:
            futures = []
            for video_record in video_records:
                future = executor.submit(turbo_video_worker, video_record, output_dirs,
                                      self.config, self.batch_size, self.use_numpy,
                                      self.vectorized_conversion, self.dtype, self.print_sub_tqdm)
                futures.append(future)

            # Track progress
            with tqdm(total=len(futures), desc="Processing videos") as pbar:
                for future in self.as_completed(futures):
                    try:
                        future.result()
                        pbar.update(1)
                    except Exception as e:
                        print(f"Error processing video: {e}")

    def copy_object_images(self):
        """High-performance parallel file copying."""
        if not self.config['processing']['copy_object_images']:
            return

        with self.ThreadPoolExecutor(max_workers=min(self.max_file_workers,
                                                     len(self.config['class_mapping']))) as executor:
            futures = []
            for original_class, class_info in self.config['class_mapping'].items():
                future = executor.submit(copy_object_images_task, self.config,
                                        original_class, class_info)
                futures.append(future)

            # Wait for all copies to complete
            for future in self.as_completed(futures):
                try:
                    future.result()
                except Exception as e:
                    print(f"Error copying object images: {e}")


def create_directories(config):
    """Create output directories for the dataset."""
    base_dir = os.path.join(config['paths']['target_dir'], config['paths']['target_name'])

    directories = [
        os.path.join(base_dir, 'images'),
        os.path.join(base_dir, 'labels'),
        os.path.join(base_dir, 'splits')
    ]

    if config['processing']['copy_object_images']:
        directories.append(os.path.join(base_dir, 'objects'))

    for directory in directories:
        os.makedirs(directory, exist_ok=True)

    return {
        'images': os.path.join(base_dir, 'images'),
        'labels': os.path.join(base_dir, 'labels'),
        'splits': os.path.join(base_dir, 'splits')
    }


def main():
    """Main function to run the YOLO dataset building process."""
    parser = argparse.ArgumentParser(description='Build YOLO dataset from raw drone footage')
    parser.add_argument('--config', default='configs/data.yaml',
                       help='Path to configuration file')
    parser.add_argument('--mode', choices=['normal', 'turbo'], default='normal',
                       help='Processing mode: normal (compatible) or turbo (max speed)')

    args = parser.parse_args()

    # Validate mode requirements
    if args.mode == 'turbo':
        if not NUMPY_AVAILABLE:
            print("Warning: NumPy not available. Falling back to normal mode.")
            args.mode = 'normal'
        elif not CONCURRENT_AVAILABLE:
            print("Warning: concurrent.futures not available. Falling back to normal mode.")
            args.mode = 'normal'

    try:
        config = load_config(args.config)

        annotations_file = os.path.join(config['paths']['train_data'], 'annotations', 'annotations.json')

        if not os.path.exists(annotations_file):
            print(f"Error: Annotations file not found at {annotations_file}")
            return

        output_dirs = create_directories(config)

        print("Loading annotations...")
        with open(annotations_file, 'r') as f:
            video_records = json.load(f)

        # Select processor based on mode
        if args.mode == 'turbo':
            processor = TurboModeProcessor(config)
        else:
            processor = NormalModeProcessor(config)

        print("Processing videos...")
        processor.process_videos(video_records, output_dirs)

        print("Copying object images...")
        processor.copy_object_images()

        print(f"\nDataset generation complete!")
        output_path = os.path.join(config['paths']['target_dir'], config['paths']['target_name'])
        print(f"Output directory: {os.path.abspath(output_path)}")
        print(f"Mode used: {args.mode}")

    except Exception as e:
        print(f"Error: {e}")
        return


if __name__ == "__main__":
    main()