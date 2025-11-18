# CLAUDE.md

## Project Objective

ZaloAI25 is a computer vision competition project for object detection in drone videos.

### Goal
Detect objects in drone video footage and generate accurate bounding box predictions for competition submission.

### Key Challenge
The most critical issue is **data leakage** - frames from the same video should NOT be split between training and validation sets. This leads to artificially inflated performance metrics (80-90% expected vs 30-40% real performance).

**Solution**: Split by entire videos, not individual frames.

## Data Structure

### Raw Dataset Organization
```
dataset/raw/train/
├── samples/
│   ├── Backpack_0/
│   │   ├── video.mp4
│   │   └── object_images/
│   │       ├── backpack_001.jpg
│   │       └── ...
│   ├── Jacket_0/
│   │   ├── video.mp4
│   │   └── object_images/
│   └── ...
└── annotations/
    └── annotations.json
```

### Class Naming Convention
- **Video IDs**: Format `{ClassName}_{instance_number}` (e.g., `Backpack_0`, `Jacket_1`)
- **Class Extraction**: Split video ID at last underscore → `Backpack_0` → `Backpack`
- **Mapping System**: Convert original class names to standardized YOLO format

### Object Classes
**Training Classes:**
- Backpack → class_id: 0
- Jacket → class_id: 1
- Laptop → class_id: 2
- Lifering → class_id: 3
- MobilePhone → class_id: 4
- Person1 → class_id: 5
- WaterBottle → class_id: 6

**Test Classes (unseen during training):**
- BlackBox, CardboardBox, LifeJacket

### Annotation Format (JSON)
```json
[
  {
    "video_id": "Backpack_0",
    "annotations": [
      {
        "interval": [start_frame, end_frame],
        "bboxes": [
          {
            "frame": 123,
            "x1": 100, "y1": 50, "x2": 200, "y2": 150
          }
        ]
      }
    ]
  }
]
```

## Architecture Overview

### Processing Pipeline
1. **Data Processing**: Convert video annotations to YOLO training format
   - **Hybrid FPS Processing**: Different frame sampling for annotated vs empty frames
   - Bounding box coordinate conversion to YOLO format
   - Class ID assignment based on video naming convention
   - ~90% dataset size reduction while preserving all training data

2. **Dataset Generation**: Two processing modes available
   - **Normal Mode**: Sequential processing, maximum compatibility
   - **Turbo Mode**: Parallel processing with NumPy optimizations

3. **Training**: Train object detection model on properly split data
4. **Evaluation**: Validate performance with video-level splits
5. **Inference**: Generate predictions on test data
6. **Submission**: Format results for competition

### Key Implementation Features

#### Dual-Mode Architecture
- **Normal Mode**: Sequential video processing with basic threading
- **Turbo Mode**: Parallel video processing using ThreadPoolExecutor
  - NumPy vectorized coordinate conversion
  - Batch processing with configurable progress display
  - Memory-efficient operations
  - Terminal-compatible output modes

#### Configuration System
- YAML-based configuration (`configs/data.yaml`) with comprehensive inline documentation
- **Hybrid FPS Configuration**: Different frame sampling rates for annotated vs empty frames
- Class mapping system for flexible class renaming
- Multiple extraction modes: legacy, all, annotated_only, hybrid, non_annotated_only
- Processing parameters for both modes
- Configurable progress display (tqdm vs simple prints)

#### Class Extraction Logic
```python
# Extract class name from video ID
base_name = video_id.rsplit('_', 1)[0]  # "Backpack_0" → "Backpack"

# Map to standardized YOLO format
class_info = config['class_mapping'][base_name]  # → {new_id: 0, new_name: "backpack"}
```

## Usage

### Build YOLO Dataset
```bash
# Normal mode (compatible)
python -m utils.data.build_yolo --config configs/data.yaml --mode normal

# Turbo mode (maximum performance)
python -m utils.data.build_yolo --config configs/data.yaml --mode turbo
```

### Output Structure
```
dataset/yolo_dataset/
├── images/           # Extracted video frames
├── labels/           # YOLO format label files
├── splits/           # Train/validation splits
└── objects/          # Object reference images
```

### Create Dataset Subsets
```bash
# Create subset with custom sampling: keep 5 annotated frames every 30, 2 non-annotated every 25
python -m utils.data.build_subyolo --config configs/data.yaml --annotated 5 30 --non_annotated 2 25

# Small subset: 1 annotated frame every 50, no non-annotated frames
python -m utils.data.build_subyolo --config configs/data.yaml --annotated 1 50 --non_annotated 0 0

# No frames at all (test configuration)
python -m utils.data.build_subyolo --config configs/data.yaml --annotated 0 0 --non_annotated 0 0
```

### Subset Output Structure
```
dataset/yolo_subset/
├── images/           # Selected frame images with same naming convention
├── labels/           # Corresponding YOLO format label files
└── splits/           # Empty folder for train/validation splits
```

#### Subset Sampling Algorithm
The `[keep, interval]` sampling logic works as follows:
- **[5, 30]**: Each 30 frames, keep 5 frames equidistant within that interval
- **[2, 25]**: Each 25 frames, keep 2 frames equidistant within that interval
- **[0, 0]**: Don't take any frames from this category
- **[1, 50]**: Each 50 frames, keep 1 frame (the first one in interval)

**Note**: Subset creation is **non-destructive** - original dataset preserved completely.

### Create Train/Val/Test Splits
```bash
# Default OOD50 splitting (video ID suffix: _0_ → train, _1_ → val)
python utils/data/split_yolo.py

# OOD50 with reversed assignment (_1_ → train, _0_ → val)
python utils/data/split_yolo.py --method ood50 --reversed

# Random ratio splitting (ignores data leakage)
python utils/data/split_yolo.py --method random_ratio --ratio 70 30

# Random ratio with test set (70% train, 20% val, 10% test)
python utils/data/split_yolo.py --method random_ratio --ratio 70 20 10

# Custom config file
python utils/data/split_yolo.py --config my_config.yaml
```

### Split Files Output Structure
```
dataset/yolo_dataset/
├── images/           # Extracted video frames
├── labels/           # YOLO format label files
├── splits/           # Train/validation/test split files
│   ├── train.txt     # Absolute paths to training images
│   ├── val.txt       # Absolute paths to validation images
│   └── test.txt      # Absolute paths to test images (if created)
└── objects/          # Object reference images
```

#### Splitting Methods

**OOD50 Method (Default)**
- Uses video ID naming convention: `{ClassName}_{0/1}_frame_{frame_num:06d}.jpg`
- Files ending in `_0_` go to training set
- Files ending in `_1_` go to validation set
- `--reversed` flag swaps the assignments
- Preserves data leakage prevention (video-level splitting)

**Random Ratio Method**
- Randomly shuffles all images and splits by percentage
- Ignores video boundaries (potential data leakage)
- Ratio format: `[train_pct, val_pct]` or `[train_pct, val_pct, test_pct]`
- Must sum to 100 (e.g., `70 30` or `70 20 10`)

**Configuration Requirement**
Add to `configs/data.yaml` in the `paths` section:
```yaml
paths:
  split_dir: "dataset/yolo_dataset"   # Source dataset directory for splitting
```

## Important Notes

### Data Leakage Prevention
- **CRITICAL**: Use video-level splitting, not frame-level splitting
- All frames from a single video must belong to the same split (train/val)
- This prevents artificial performance inflation

### Performance Considerations
- Monitor for realistic performance metrics (30-40% expected, not 80-90%)
- Turbo mode requires NumPy and concurrent.futures
- Large video files benefit from parallel processing

#### Extraction Mode Performance Impact
- **`legacy`**: Fast (default: only annotated frames) - **Recommended for speed**
- **`annotated_only`**: Fast (only frames with annotations)
- **`all`**: Slow (every frame from video) - **Maximum coverage**
- **`hybrid`**: Medium (configurable FPS sampling) - **Storage efficient**
- **`non_annotated_only`**: Fast (only empty frames, no training data)

**Performance Ranking (Fastest → Slowest)**:
1. `annotated_only` / `legacy` (default)
2. `non_annotated_only`
3. `hybrid` (depends on FPS settings)
4. `all` (slowest but most complete)

### Configuration Guidelines
- **Hybrid FPS Mode**: Set `extraction_mode: "hybrid"` with `hybrid_fps.enabled: true`
  - `annotated_fps: null` preserves all training data at original FPS
  - `non_annotated_fps: 3` samples empty frames at 3 FPS for ~90% size reduction
  - `ensure_annotated_frames: true` guarantees no training data loss
- **Extraction Modes**:
  - `legacy` - backward compatibility mode (fast, default: annotated only)
  - `all` - process every frame from video (slow, maximum coverage)
  - `annotated_only` - process only frames with annotations (fast)
  - `hybrid` - different FPS for annotated vs empty frames (medium, storage efficient)
  - `non_annotated_only` - process only empty frames (no training data)
- **Subset Configuration**: Add to `paths` section for build_subyolo.py
  ```yaml
  paths:
    target_subset_dir: "dataset/subset"   # Subset output directory
    target_subset_name: "yolo_dataset"    # Subset dataset name
  ```
- Adjust turbo mode settings based on available CPU/memory
- Class mapping allows flexible renaming and ID assignment
- Use `print_sub_tqdm: false` for terminals that don't support `\r` (carriage return)
- Set `print_sub_tqdm: true` for detailed progress bars with batch information

### Hybrid FPS Benefits
- **Storage Efficiency**: Reduces dataset size from 20GB to ~2-3GB (~90% reduction)
- **Training Quality**: Preserves 100% of annotated frames for complete training data
- **Performance**: Faster training with redundant empty frames removed
- **Flexibility**: Configurable FPS settings for different use cases

## Troubleshooting

### Performance Issues
**Problem**: `build_yolo.py` is running very slowly
**Solution**: Check your `extraction_mode` setting:
- Use `extraction_mode: "legacy"` for fast processing (default: annotated frames only)
- Avoid `extraction_mode: "all"` unless you need every single frame
- Use `--mode turbo` for parallel processing if available

**Problem**: Memory usage too high
**Solution**:
- Use `--mode normal` for sequential processing
- Reduce turbo mode `batch_size` in config
- Close other applications while processing

### Dataset Issues
**Problem**: "Dataset generation complete!" but no files created
**Solution**:
- Check that video files exist in `dataset/raw/train/samples/*/`
- Verify video extensions match config (`video_ext: ["mp4", "avi", "mov"]`)
- Check annotations file exists at `dataset/raw/train/annotations/annotations.json`

**Problem**: build_subyolo.py can't find source dataset
**Solution**:
- Ensure you've run `build_yolo.py` first to create the source dataset
- Check `target_dir` and `target_name` paths in config
- Verify source dataset has `images/` and `labels/` folders

### Configuration Issues
**Problem**: "No class mapping found for video_id" error
**Solution**:
- Check video ID format matches class naming convention
- Verify class exists in `class_mapping` section
- Ensure video IDs follow `{ClassName}_{instance_number}` format

**Problem**: Subset creation fails with parameter errors
**Solution**:
- Ensure all parameters are non-negative integers
- Use format: `--annotated K N` where K = frames to keep, N = interval size
- Don't use `--annotated 5 0` (interval can't be 0 when keep > 0)