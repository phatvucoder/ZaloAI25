# Configuration Files

This directory contains configuration files for the ZaloAI25 project.

## data.yaml

Configuration for building YOLO dataset from raw drone footage.

### Structure

```yaml
paths:
  train_data: "dataset/raw/train"    # Path to training data
  target_dir: "dataset"              # Output directory
  target_name: "yolo_dataset"        # Dataset name

files:
  video_ext: ["mp4", "avi", "mov"]   # Supported video formats
  image_ext: ["jpg", "jpeg", "png"]  # Output image formats

video:
  fps: null                          # Target FPS (null = original)
  extract_all_frames: true           # true = all frames, false = annotated only

class_mapping:
  # Original class name: {new_id: X, new_name: "standardized_name"}
  Backpack: {new_id: 0, new_name: "backpack"}
  Jacket: {new_id: 1, new_name: "jacket"}
  Laptop: {new_id: 2, new_name: "laptop"}
  Lifering: {new_id: 3, new_name: "lifering"}
  MobilePhone: {new_id: 4, new_name: "mobile_phone"}
  Person1: {new_id: 5, new_name: "person"}
  WaterBottle: {new_id: 6, new_name: "water_bottle"}

processing:
  copy_object_images: true           # Copy object images to dataset

normal_mode:
  sequential_processing: true        # Process videos sequentially
  io_threading: true                 # Use threading for I/O
  max_io_threads: 4                  # Maximum I/O threads

turbo_mode:
  parallel_processing: true          # Process videos in parallel
  max_workers: null                  # Worker count (null = auto-detect)
  use_threading: true                # Use ThreadPool vs ProcessPool
  batch_processing: true             # Process frames in batches
  batch_size: 32                     # Frames per batch
  memory_efficient: true             # Memory optimizations
  use_numpy: true                    # Use NumPy for calculations
  vectorized_conversion: true        # Vectorized coordinate conversion
  dtype: "float32"                   # NumPy data type
  concurrent_file_operations: true   # Parallel file operations
  max_file_workers: 8                # File operation workers
  buffer_size: 8192                  # I/O buffer size
  memory_mapping: false              # Use memory mapping
  gc_frequency: 100                  # Garbage collection frequency
  prefetch_frames: true              # Prefetch video frames
  prefetch_buffer_size: 16           # Prefetch buffer size
  async_writes: true                 # Asynchronous file writes
  write_batch_size: 64               # Write batch size
```

### Usage

```bash
# Normal mode (compatible)
python -m utils.data.build_yolo --config configs/data.yaml --mode normal

# Turbo mode (maximum performance)
python -m utils.data.build_yolo --config configs/data.yaml --mode turbo
```

### Class Mapping Logic

The system extracts class names from video IDs using this pattern:
- Video ID: `"Backpack_0"` → Class: `"Backpack"`
- Video ID: `"Jacket_1"` → Class: `"Jacket"`

Then maps to standardized YOLO format:
- `"Backpack"` → `{new_id: 0, new_name: "backpack"}`
- `"Jacket"` → `{new_id: 1, new_name: "jacket"}`

### Processing Modes

#### Normal Mode
- Sequential video processing
- Basic threading for I/O operations
- Maximum compatibility across systems
- Lower memory usage

#### Turbo Mode
- Parallel video processing using ThreadPoolExecutor
- NumPy vectorized coordinate conversion
- Advanced memory optimizations
- Higher throughput for large datasets
- Requires NumPy and concurrent.futures

### Video Processing Options

- **extract_all_frames: true** - Process every frame from video start
- **extract_all_frames: false** - Only process annotated frames

### Output Structure

```
dataset/yolo_dataset/
├── images/           # Extracted frames
├── labels/           # YOLO format labels
├── splits/           # Train/val splits
└── objects/          # Object reference images (if enabled)
```