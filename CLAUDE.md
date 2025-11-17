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

## Important Notes

### Data Leakage Prevention
- **CRITICAL**: Use video-level splitting, not frame-level splitting
- All frames from a single video must belong to the same split (train/val)
- This prevents artificial performance inflation

### Performance Considerations
- Monitor for realistic performance metrics (30-40% expected, not 80-90%)
- Turbo mode requires NumPy and concurrent.futures
- Large video files benefit from parallel processing

### Configuration Guidelines
- **Hybrid FPS Mode**: Set `extraction_mode: "hybrid"` with `hybrid_fps.enabled: true`
  - `annotated_fps: null` preserves all training data at original FPS
  - `non_annotated_fps: 3` samples empty frames at 3 FPS for ~90% size reduction
  - `ensure_annotated_frames: true` guarantees no training data loss
- **Extraction Modes**:
  - `legacy` - backward compatibility mode
  - `all` - process every frame from video
  - `annotated_only` - process only frames with annotations
  - `hybrid` - different FPS for annotated vs empty frames
- Adjust turbo mode settings based on available CPU/memory
- Class mapping allows flexible renaming and ID assignment
- Use `print_sub_tqdm: false` for terminals that don't support `\r` (carriage return)
- Set `print_sub_tqdm: true` for detailed progress bars with batch information

### Hybrid FPS Benefits
- **Storage Efficiency**: Reduces dataset size from 20GB to ~2-3GB (~90% reduction)
- **Training Quality**: Preserves 100% of annotated frames for complete training data
- **Performance**: Faster training with redundant empty frames removed
- **Flexibility**: Configurable FPS settings for different use cases