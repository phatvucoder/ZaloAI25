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
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)


def convert_to_yolo(box, img_w, img_h):
    x1, y1, x2, y2 = float(box[0]), float(box[1]), float(box[2]), float(box[3])
    img_w, img_h = float(img_w), float(img_h)

    dw = 1.0 / img_w
    dh = 1.0 / img_h

    x_center = (x1 + x2) / 2.0
    y_center = (y1 + y2) / 2.0
    width = x2 - x1
    height = y2 - y1

    return (x_center * dw, y_center * dh, width * dw, height * dh)




def process_video(config, video_record, output_dirs):
    video_id = video_record['video_id']
    video_dir = os.path.join(config['paths']['train_data'], 'samples', video_id)

    # Find video file
    video_path = None
    for ext in config['files']['video_ext']:
        for file in os.listdir(video_dir):
            if file.lower().endswith(ext.lower()):
                video_path = os.path.join(video_dir, file)
                break
        if video_path:
            break

    if not video_path:
        print(f"Warning: No video file found for {video_id}")
        return

    # Find class mapping to get correct class_id
    class_info = None
    base_name = video_id.rsplit('_', 1)[0]
    for original_class, info in config['class_mapping'].items():
        if original_class == base_name:
            class_info = info
            break

    if not class_info:
        print(f"Warning: No class mapping found for {video_id}")
        return

    new_class_id = class_info['new_id']

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Error: Could not open video {video_path}")
        return

    video_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    video_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)

    frame_skip = 1
    if config['video']['fps']:
        frame_skip = max(1, int(fps / config['video']['fps']))

    # Check if we should extract all frames or just annotated frames
    extract_all_frames = config['video'].get('extract_all_frames', False)

    if extract_all_frames:
        # Extract ALL frames from the video
        annotations_dict = {}
        for interval in video_record.get('annotations', []):
            for bbox_data in interval.get('bboxes', []):
                frame_num = int(bbox_data['frame'])
                box = (int(bbox_data['x1']), int(bbox_data['y1']),
                       int(bbox_data['x2']), int(bbox_data['y2']))

                if frame_num not in annotations_dict:
                    annotations_dict[frame_num] = []
                annotations_dict[frame_num].append(box)

        # Get total frame count
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        print(f"Extracting all {total_frames} frames from {video_id}")

        # Process every frame
        for frame_num in tqdm(range(total_frames), desc=f"Processing {video_id}"):
            ret, frame = cap.read()
            if not ret:
                break

            file_basename = f"{video_id}_frame_{frame_num:06d}"
            image_path = os.path.join(output_dirs['images'], f"{file_basename}.{config['files']['image_ext'][0]}")
            label_path = os.path.join(output_dirs['labels'], f"{file_basename}.txt")

            # Save the image
            cv2.imwrite(image_path, frame)

            # Get annotations for this frame (empty if none)
            boxes = annotations_dict.get(frame_num, [])

            # Write labels (empty file if no annotations)
            with open(label_path, 'w') as f:
                for box in boxes:
                    yolo_coords = convert_to_yolo(box, video_width, video_height)
                    if yolo_coords:
                        x_c, y_c, w, h = yolo_coords
                        f.write(f"{new_class_id} {x_c:.6f} {y_c:.6f} {w:.6f} {h:.6f}\n")
    else:
        # Only process annotated frames (current behavior)
        frames_dict = {}
        for interval in video_record.get('annotations', []):
            for bbox_data in interval.get('bboxes', []):
                frame_num = int(bbox_data['frame'])
                box = (int(bbox_data['x1']), int(bbox_data['y1']),
                       int(bbox_data['x2']), int(bbox_data['y2']))

                if frame_num not in frames_dict:
                    frames_dict[frame_num] = []
                frames_dict[frame_num].append(box)

        for frame_num, boxes in frames_dict.items():
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_num)
            ret, frame = cap.read()

            if not ret:
                continue

            file_basename = f"{video_id}_frame_{frame_num:06d}"
            image_path = os.path.join(output_dirs['images'], f"{file_basename}.{config['files']['image_ext'][0]}")
            label_path = os.path.join(output_dirs['labels'], f"{file_basename}.txt")

            cv2.imwrite(image_path, frame)

            with open(label_path, 'w') as f:
                for box in boxes:
                    yolo_coords = convert_to_yolo(box, video_width, video_height)
                    if yolo_coords:
                        x_c, y_c, w, h = yolo_coords
                        f.write(f"{new_class_id} {x_c:.6f} {y_c:.6f} {w:.6f} {h:.6f}\n")

    cap.release()


class NormalModeProcessor:
    """Simple, compatible processor for most systems."""

    def __init__(self, config):
        self.config = config
        # Load normal mode settings
        self.normal_config = config.get('normal_mode', {})
        self.max_io_threads = self.normal_config.get('max_io_threads', 4)

    def process_videos(self, video_records, output_dirs):
        """Sequential video processing with basic optimizations."""
        print("Running in NORMAL mode - maximum compatibility")

        for video_record in tqdm(video_records, desc="Processing videos"):
            process_video(self.config, video_record, output_dirs)

    def copy_object_images(self):
        """Standard file copying with threading."""
        if not self.config['processing']['copy_object_images']:
            return

        samples_dir = os.path.join(self.config['paths']['train_data'], 'samples')
        base_dir = os.path.join(self.config['paths']['target_dir'], self.config['paths']['target_name'])
        objects_output_dir = os.path.join(base_dir, 'objects')

        def copy_directory_task(original_class, class_info):
            video_id = f"{original_class}_0"
            source_dir = os.path.join(samples_dir, video_id, 'object_images')
            target_dir = os.path.join(objects_output_dir, class_info['new_name'])

            if os.path.exists(source_dir):
                os.makedirs(target_dir, exist_ok=True)
                for img_file in os.listdir(source_dir):
                    if any(img_file.lower().endswith(ext.lower()) for ext in self.config['files']['image_ext']):
                        src_path = os.path.join(source_dir, img_file)
                        dst_path = os.path.join(target_dir, img_file)
                        if not os.path.exists(dst_path):
                            shutil.copy2(src_path, dst_path)

        # Use basic threading for I/O operations
        threads = []
        for original_class, class_info in self.config['class_mapping'].items():
            thread = threading.Thread(target=copy_directory_task, args=(original_class, class_info))
            thread.start()
            threads.append(thread)

        for thread in threads:
            thread.join()


def turbo_video_worker(video_record, output_dirs, config, batch_size, use_numpy, vectorized_conversion, dtype):
    """Standalone worker function for turbo mode video processing to avoid pickling issues."""
    import cv2
    import numpy as np
    import os
    from tqdm import tqdm

    class_id = 0  # Will be set per class
    video_id = video_record['video_id']

    # Construct video path like normal mode
    video_dir = os.path.join(config['paths']['train_data'], 'samples', video_id)

    # Find video file
    video_path = None
    for ext in config['files']['video_ext']:
        for file in os.listdir(video_dir):
            if file.lower().endswith(ext.lower()):
                video_path = os.path.join(video_dir, file)
                break
        if video_path:
            break

    if not video_path:
        print(f"Warning: Video file not found for {video_id}")
        return

    print(f"Found video: {video_path}")

    # Find class mapping to get correct class_id
    class_info = None
    base_name = video_id.rsplit('_', 1)[0]
    for original_class, info in config['class_mapping'].items():
        if original_class == base_name:
            class_info = info
            break

    if not class_info:
        print(f"Warning: No class mapping found for {video_id}")
        return

    class_id = class_info['new_id']
    print(f"Processing {video_id} with class_id {class_id}")

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Error: Could not open video: {video_path}")
        return

    try:
        video_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        video_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = cap.get(cv2.CAP_PROP_FPS)

        frame_skip = 1
        if config['video']['fps']:
            frame_skip = max(1, int(fps / config['video']['fps']))

        # Check if we should extract all frames or just annotated frames
        extract_all_frames = config['video'].get('extract_all_frames', False)

        if extract_all_frames:
            # Extract ALL frames from the video
            annotations_dict = {}
            for interval in video_record.get('annotations', []):
                for bbox_data in interval.get('bboxes', []):
                    frame_num = int(bbox_data['frame'])
                    box = (int(bbox_data['x1']), int(bbox_data['y1']),
                           int(bbox_data['x2']), int(bbox_data['y2']))

                    if frame_num not in annotations_dict:
                        annotations_dict[frame_num] = []
                    annotations_dict[frame_num].append(box)

            # Get total frame count
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

            # Process every frame with progress bar
            batch_size = min(32, total_frames)  # Use configurable batch size
            processed_frames = 0

            with tqdm(total=total_frames, desc=f"Processing {video_id}") as pbar:
                for frame_num in range(total_frames):
                    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_num)
                    ret, frame = cap.read()
                    if not ret:
                        break

                    file_basename = f"{video_id}_frame_{frame_num:06d}"
                    image_path = os.path.join(output_dirs['images'], f"{file_basename}.{config['files']['image_ext'][0]}")
                    label_path = os.path.join(output_dirs['labels'], f"{file_basename}.txt")

                    # Save the image
                    cv2.imwrite(image_path, frame)

                    # Get annotations for this frame (empty if none)
                    boxes = annotations_dict.get(frame_num, [])

                    # Write labels (empty file if no annotations)
                    with open(label_path, 'w') as f:
                        if use_numpy and vectorized_conversion and boxes:
                            box_array = np.array(boxes, dtype=dtype)
                            x1, y1, x2, y2 = box_array[:, 0], box_array[:, 1], box_array[:, 2], box_array[:, 3]

                            x_center = (x1 + x2) / 2.0 / video_width
                            y_center = (y1 + y2) / 2.0 / video_height
                            width = (x2 - x1) / video_width
                            height = (y2 - y1) / video_height

                            for i in range(len(x_center)):
                                f.write(f"{class_id} {x_center[i]:.6f} {y_center[i]:.6f} {width[i]:.6f} {height[i]:.6f}\n")
                        else:
                            for box in boxes:
                                x1, y1, x2, y2 = float(box[0]), float(box[1]), float(box[2]), float(box[3])

                                dw = 1.0 / video_width
                                dh = 1.0 / video_height

                                x_center = (x1 + x2) / 2.0
                                y_center = (y1 + y2) / 2.0
                                width = (x2 - x1)
                                height = (y2 - y1)

                                x_c_norm = x_center * dw
                                y_c_norm = y_center * dh
                                w_norm = width * dw
                                h_norm = height * dh

                                f.write(f"{class_id} {x_c_norm:.6f} {y_c_norm:.6f} {w_norm:.6f} {h_norm:.6f}\n")

                    processed_frames += 1
                    pbar.update(1)

                    # Show batch progress info
                    if processed_frames % batch_size == 0:
                        pbar.set_postfix({"batch": f"{processed_frames//batch_size} ({processed_frames}/{total_frames})"})
        else:
            # Only process annotated frames
            frames_dict = {}
            for interval in video_record.get('annotations', []):
                for bbox_data in interval.get('bboxes', []):
                    frame_num = int(bbox_data['frame'])
                    box = (int(bbox_data['x1']), int(bbox_data['y1']),
                           int(bbox_data['x2']), int(bbox_data['y2']))

                    if frame_num not in frames_dict:
                        frames_dict[frame_num] = []
                    frames_dict[frame_num].append(box)

            # Process frames with NumPy optimization and progress bar
            total_annotated_frames = len(frames_dict)
            processed_frames = 0

            with tqdm(total=total_annotated_frames, desc=f"Processing {video_id} (annotated)") as pbar:
                for frame_num, boxes in frames_dict.items():
                    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_num)
                    ret, frame = cap.read()
                    if not ret:
                        continue

                    file_basename = f"{video_id}_frame_{frame_num:06d}"
                    image_path = os.path.join(output_dirs['images'], f"{file_basename}.{config['files']['image_ext'][0]}")
                    label_path = os.path.join(output_dirs['labels'], f"{file_basename}.txt")

                    cv2.imwrite(image_path, frame)

                    with open(label_path, 'w') as f:
                        if boxes and use_numpy and vectorized_conversion:
                            box_array = np.array(boxes, dtype=dtype)
                            x1, y1, x2, y2 = box_array[:, 0], box_array[:, 1], box_array[:, 2], box_array[:, 3]

                            x_center = (x1 + x2) / 2.0 / video_width
                            y_center = (y1 + y2) / 2.0 / video_height
                            width = (x2 - x1) / video_width
                            height = (y2 - y1) / video_height

                            for i in range(len(x_center)):
                                f.write(f"{class_id} {x_center[i]:.6f} {y_center[i]:.6f} {width[i]:.6f} {height[i]:.6f}\n")
                        else:
                            # Fallback to original method
                            for box in boxes:
                                x1, y1, x2, y2 = float(box[0]), float(box[1]), float(box[2]), float(box[3])

                                dw = 1.0 / video_width
                                dh = 1.0 / video_height

                                x_center = (x1 + x2) / 2.0
                                y_center = (y1 + y2) / 2.0
                                width = (x2 - x1)
                                height = (y2 - y1)

                                x_c_norm = x_center * dw
                                y_c_norm = y_center * dh
                                w_norm = width * dw
                                h_norm = height * dh

                                f.write(f"{class_id} {x_c_norm:.6f} {y_c_norm:.6f} {w_norm:.6f} {h_norm:.6f}\n")

                    processed_frames += 1
                    pbar.update(1)

                    # Show batch progress info
                    if processed_frames % batch_size == 0:
                        pbar.set_postfix({"batch": f"{processed_frames//batch_size} ({processed_frames}/{total_annotated_frames})"})
    finally:
        cap.release()


class TurboModeProcessor:
    """High-performance processor using advanced techniques."""

    def __init__(self, config):
        self.config = config
        if not NUMPY_AVAILABLE or not CONCURRENT_AVAILABLE:
            raise RuntimeError("Turbo mode requires NumPy and concurrent.futures")

        # Import required modules for turbo mode
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
                                      self.vectorized_conversion, self.dtype)
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

        samples_dir = os.path.join(self.config['paths']['train_data'], 'samples')
        base_dir = os.path.join(self.config['paths']['target_dir'], self.config['paths']['target_name'])
        objects_output_dir = os.path.join(base_dir, 'objects')

        def copy_directory_task(original_class, class_info):
            video_id = f"{original_class}_0"
            source_dir = os.path.join(samples_dir, video_id, 'object_images')
            target_dir = os.path.join(objects_output_dir, class_info['new_name'])

            if not os.path.exists(source_dir):
                return

            os.makedirs(target_dir, exist_ok=True)

            # Get all files to copy
            files_to_copy = []
            for img_file in os.listdir(source_dir):
                if any(img_file.lower().endswith(ext.lower()) for ext in self.config['files']['image_ext']):
                    files_to_copy.append((os.path.join(source_dir, img_file),
                                        os.path.join(target_dir, img_file)))

            # Copy files with threading
            with self.ThreadPoolExecutor(max_workers=self.max_file_workers) as executor:
                futures = []
                for src_path, dst_path in files_to_copy:
                    if not os.path.exists(dst_path):
                        future = executor.submit(shutil.copy2, src_path, dst_path)
                        futures.append(future)

                # Wait for all copies to complete
                for future in self.as_completed(futures):
                    future.result()

        # Process all directories in parallel
        with self.ThreadPoolExecutor(max_workers=min(self.max_file_workers, len(self.config['class_mapping']))) as executor:
            futures = []
            for original_class, class_info in self.config['class_mapping'].items():
                future = executor.submit(copy_directory_task, original_class, class_info)
                futures.append(future)

            for future in self.as_completed(futures):
                future.result()



def create_directories(config):
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


if __name__ == "__main__":
    main()