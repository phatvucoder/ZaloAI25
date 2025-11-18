#!/usr/bin/env python3

import os
import sys
import json
import yaml
import argparse
import cv2
import time
import numpy as np
from pathlib import Path
from typing import Dict, Any, Optional, List, Tuple, Union
from datetime import datetime
from tqdm import tqdm

# Check for optional dependencies
ULTRALYTICS_AVAILABLE = False
try:
    from ultralytics import YOLO
    ULTRALYTICS_AVAILABLE = True
except ImportError:
    print("Warning: ultralytics not installed. Install with: pip install ultralytics")

TORCH_AVAILABLE = False
try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    print("Warning: PyTorch not installed. Install with: pip install torch")

def load_config(config_path: str) -> Dict[str, Any]:
    """Load inference configuration from YAML file."""
    try:
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        return config
    except Exception as e:
        print(f"Error loading configuration: {e}")
        raise


def validate_config(config: Dict[str, Any]) -> None:
    """Validate inference configuration."""
    required_sections = ['paths', 'inference']

    for section in required_sections:
        if section not in config:
            print(f"Error: Missing required configuration section: {section}")
            raise ValueError(f"Missing required configuration section: {section}")

    # Validate paths
    paths = config['paths']
    test_data_dir = paths.get('test_data_dir')
    public_test_dir = paths.get('public_test_dir')

    if not test_data_dir and not public_test_dir:
        print("Warning: No test data directories specified in configuration")

    # Validate inference settings
    inference_config = config['inference']
    required_inference = ['confidence', 'iou_threshold']

    for setting in required_inference:
        if setting not in inference_config:
            print(f"Warning: Missing inference setting: {setting}, using default")


def check_gpu_availability() -> None:
    """Check GPU availability and print status."""
    if not TORCH_AVAILABLE:
        print("Using CPU (PyTorch not available)")
        return

    import torch
    if not torch.cuda.is_available():
        print("Using CPU (CUDA not available)")
        return

    print(f"Using GPU: {torch.cuda.get_device_name(0)}")


def extract_class_from_video_name(video_folder_name: str) -> Optional[str]:
    """Extract class name from video folder name.

    Examples:
        "BlackBox_0" -> "BlackBox"
        "CardboardBox_1" -> "CardboardBox"
        "LifeJacket_0" -> "LifeJacket"
    """
    # Split by underscore and take the first part(s) until the last number
    parts = video_folder_name.split('_')
    if len(parts) < 2:
        return None

    # Find the last underscore before a number
    for i in range(len(parts) - 1, 0, -1):
        if parts[i].isdigit():
            class_name = '_'.join(parts[:i])
            return class_name

    return parts[0]  # fallback to first part


def load_model(model_path: str):
    """Load YOLO model."""
    if not ULTRALYTICS_AVAILABLE:
        print("Error: Ultralytics not available. Cannot load model.")
        raise ImportError("Install ultralytics: pip install ultralytics")

    try:
        model = YOLO(model_path)
        return model

    except Exception as e:
        print(f"Error: Failed to load model: {e}")
        raise


def find_test_videos(paths_config: Dict[str, Any], class_mapping: Dict[str, int]) -> List[Tuple[str, str, int]]:
    """Find test videos and extract class information.

    Returns:
        List of tuples: (video_folder_path, class_name, class_id)
    """
    video_info = []

    public_test_dir = paths_config.get('public_test_dir')
    if public_test_dir and os.path.exists(public_test_dir):
        # Look for samples directory first
        samples_dir = os.path.join(public_test_dir, 'samples')
        if os.path.exists(samples_dir):
            search_dir = samples_dir
        else:
            search_dir = public_test_dir

        for item in os.listdir(search_dir):
            item_path = os.path.join(search_dir, item)
            if os.path.isdir(item_path):
                video_path = os.path.join(item_path, 'drone_video.mp4')
                if os.path.exists(video_path):
                    # Extract class from folder name
                    class_name = extract_class_from_video_name(item)
                    if class_name and class_name in class_mapping:
                        class_id = class_mapping[class_name]
                        video_info.append((item_path, class_name, class_id))
                    else:
                        print(f"Warning: Class '{class_name}' not found in mapping for video {item}")

    if video_info:
        print(f"Found {len(video_info)} video folders with valid class mapping")
        # Show class distribution
        class_counts = {}
        for _, class_name, _ in video_info:
            class_counts[class_name] = class_counts.get(class_name, 0) + 1
        print(f"Class distribution: {dict(class_counts)}")

    return sorted(video_info)


def find_video_file(video_folder: str) -> Optional[str]:
    video_path = os.path.join(video_folder, 'drone_video.mp4')
    if os.path.exists(video_path):
        return video_path
    return None


def select_best_bbox_per_frame(xyxy_boxes: np.ndarray, confidences: np.ndarray) -> Tuple[np.ndarray, float]:
    """Select the best bounding box from multiple detections in a frame.

    Args:
        xyxy_boxes: Array of bounding boxes in [x1, y1, x2, y2] format
        confidences: Array of confidence scores for each box

    Returns:
        Tuple of (best_box, best_confidence)
    """
    if len(xyxy_boxes) == 0:
        return None, 0.0

    if len(xyxy_boxes) == 1:
        return xyxy_boxes[0], float(confidences[0])

    # Select the box with highest confidence score
    best_idx = np.argmax(confidences)
    return xyxy_boxes[best_idx], float(confidences[best_idx])


def process_video(model, video_path: str, inference_config: Dict[str, Any], target_class_id: int) -> Tuple[List[Dict], Dict[str, Any]]:
    """
    Process a single video and return detections and statistics.

    Args:
        model: YOLO model
        video_path: Path to video file
        inference_config: Inference configuration

    Returns:
        Tuple of (detections list, processing statistics)
    """

    # Extract inference parameters
    confidence = inference_config.get('confidence', 0.25)
    iou_threshold = inference_config.get('iou_threshold', 0.45)
    max_det = inference_config.get('max_det', 1000)
    stream = inference_config.get('stream', True)
    half = inference_config.get('half', False)

    # Statistics tracking
    stats = {
        'total_frames': 0,
        'processed_frames': 0,
        'total_detections': 0,
        'processing_time': 0,
        'fps': 0,
        'frames_with_detections': 0,
        'avg_detections_per_frame': 0
    }

    # Detections storage
    video_detections = []

    start_time = time.time()

    try:
        # Open video to get frame count for progress tracking
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            print(f"Error: Could not open video: {video_path}")
            return video_detections, stats

        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        cap.release()

        stats['total_frames'] = total_frames

        # Run inference
        results_generator = model.predict(
            video_path,
            stream=stream,
            conf=confidence,
            iou=iou_threshold,
            max_det=max_det,
            half=half,
            verbose=False
        )

        # Process each frame
        for frame_idx, results in enumerate(results_generator):
            stats['processed_frames'] += 1

            # Get bounding boxes
            boxes = results.boxes
            if boxes is not None and len(boxes) > 0:
                # Move to CPU and convert to numpy
                xyxy_boxes = boxes.xyxy.cpu().numpy()
                classes = boxes.cls.cpu().numpy()
                confidences = boxes.conf.cpu().numpy()

                # Filter for target class only
                target_mask = classes == target_class_id
                target_boxes = xyxy_boxes[target_mask]
                target_confs = confidences[target_mask]

                if len(target_boxes) > 0:
                    stats['frames_with_detections'] += 1

                    # Select ONLY the best bbox per frame (highest confidence)
                    best_box, best_conf = select_best_bbox_per_frame(target_boxes, target_confs)

                    if best_box is not None:
                        x1, y1, x2, y2 = best_box

                        detection = {
                            "frame": frame_idx,
                            "x1": int(round(float(x1))),
                            "y1": int(round(float(y1))),
                            "x2": int(round(float(x2))),
                            "y2": int(round(float(y2)))
                        }
                        video_detections.append(detection)

                    stats['total_detections'] += 1  # Only one detection per frame

        # Calculate final statistics
        end_time = time.time()
        stats['processing_time'] = end_time - start_time
        if stats['processed_frames'] > 0:
            stats['fps'] = stats['processed_frames'] / stats['processing_time']
            stats['avg_detections_per_frame'] = stats['total_detections'] / stats['processed_frames']

        
    except Exception as e:
        print(f"Error: Error processing video {video_path}: {e}")
        stats['error'] = str(e)

    return video_detections, stats


def format_predictions(video_info: List[Tuple[str, str, int]], all_detections: List[List[Dict]],
                      all_stats: List[Dict[str, Any]]) -> List[Dict]:
    """Format predictions in competition JSON format."""
    formatted_predictions = []

    for i, (video_folder, class_name, class_id) in enumerate(video_info):
        video_id = os.path.basename(video_folder)
        detections = all_detections[i] if i < len(all_detections) else []
        stats = all_stats[i] if i < len(all_stats) else {}

        # Format detections for competition
        if detections:
            # Group detections by frame (if needed for future extensions)
            detections_list = [{"bboxes": detections}]
        else:
            detections_list = []

        video_prediction = {
            "video_id": video_id,
            "detections": detections_list,
            "statistics": {
                "total_frames": stats.get('total_frames', 0),
                "processed_frames": stats.get('processed_frames', 0),
                "total_detections": stats.get('total_detections', 0),
                "frames_with_detections": stats.get('frames_with_detections', 0),
                "processing_fps": stats.get('fps', 0),
                "class_name": class_name,
                "class_id": class_id
            }
        }

        formatted_predictions.append(video_prediction)

    return formatted_predictions


def save_predictions(predictions: List[Dict], output_file: str) -> None:
    """Save predictions to JSON file."""
    try:
        # Prepare output data (without statistics for competition submission)
        competition_predictions = []
        for pred in predictions:
            comp_pred = {
                "video_id": pred["video_id"],
                "detections": pred["detections"]
            }
            competition_predictions.append(comp_pred)

        # Save competition format
        with open(output_file, 'w') as f:
            json.dump(competition_predictions, f, indent=4)

        # Also save detailed results with statistics
        detailed_file = output_file.replace('.json', '_detailed.json')
        with open(detailed_file, 'w') as f:
            json.dump(predictions, f, indent=2)

    except Exception as e:
        print(f"Error: Error saving predictions: {e}")
        raise


def print_inference_summary(predictions: List[Dict], all_stats: List[Dict[str, Any]],
                           processing_time: float) -> None:
    """Print comprehensive inference summary."""
    total_videos = len(predictions)
    successful_videos = len([stats for stats in all_stats if 'error' not in stats])

    # Frame statistics
    total_frames = sum(stats.get('processed_frames', 0) for stats in all_stats if 'error' not in stats)
    total_detections = sum(stats.get('total_detections', 0) for stats in all_stats if 'error' not in stats)
    frames_with_detections = sum(stats.get('frames_with_detections', 0) for stats in all_stats if 'error' not in stats)

    # FPS statistics
    fps_list = [stats.get('fps', 0) for stats in all_stats if 'error' not in stats and stats.get('fps', 0) > 0]
    avg_fps = sum(fps_list) / len(fps_list) if fps_list else 0
    min_fps = min(fps_list) if fps_list else 0
    max_fps = max(fps_list) if fps_list else 0

    # Detection statistics - safe division with proper checks
    detection_rate = (frames_with_detections / total_frames * 100) if total_frames > 0 else 0
    detections_per_frame = (total_detections / total_frames) if total_frames > 0 else 0
    detections_per_detected_frame = (total_detections / frames_with_detections) if frames_with_detections > 0 else 0
    overall_fps = (total_frames / processing_time) if processing_time > 0 else 0

    print(f"\n📊 INFERENCE SUMMARY")
    print("="*60)
    success_rate = (successful_videos / total_videos * 100) if total_videos > 0 else 0
    print(f"📹 Videos Processed: {successful_videos}/{total_videos} ({success_rate:.1f}%)")
    if successful_videos < total_videos:
        failed_videos = [i for i, stats in enumerate(all_stats) if 'error' in stats]
        print(f"❌ Failed Videos: {len(failed_videos)}")

    print(f"\n🎯 Frame Processing:")
    print(f"  Total Frames: {total_frames:,}")
    print(f"  Frames with Detections: {frames_with_detections:,} ({detection_rate:.1f}%)")
    print(f"  Total Detections: {total_detections:,}")
    print(f"  Detections per Frame: {detections_per_frame:.2f}")
    print(f"  Detections per Detected Frame: {detections_per_detected_frame:.2f}")

    print(f"\n⚡ Performance:")
    print(f"  Processing Time: {processing_time:.1f}s")
    print(f"  Average FPS: {avg_fps:.2f}")
    print(f"  Fastest FPS: {max_fps:.2f}")
    print(f"  Slowest FPS: {min_fps:.2f}")
    print(f"  Frames per Second: {overall_fps:.1f}")

    if total_videos > 0:
        # Per-video breakdown
        print(f"\n📋 Per-Video Breakdown:")
        for i, (pred, stats) in enumerate(zip(predictions, all_stats)):
            video_id = pred['video_id']
            if 'error' not in stats:
                frames = stats.get('processed_frames', 0)
                detections = stats.get('total_detections', 0)
                fps = stats.get('fps', 0)
                frames_with_dets = stats.get('frames_with_detections', 0)
                video_detection_rate = (frames_with_dets / frames * 100) if frames > 0 else 0
                print(f"  {video_id:<20} | {frames:>7,} frames | {detections:>6,} detections | {fps:>5.1f} fps | {video_detection_rate:5.1f}%")
            else:
                print(f"  {video_id:<20} | ❌ ERROR")

    # Additional insights
    print(f"\n💡 Insights:")
    if detection_rate > 50:
        print(f"  • High detection rate: {detection_rate:.1f}%")
    elif detection_rate > 20:
        print(f"  • Moderate detection rate: {detection_rate:.1f}%")
    else:
        print(f"  • Low detection rate: {detection_rate:.1f}%")

    if avg_fps > 30:
        print(f"  • Excellent performance: {avg_fps:.1f} FPS")
    elif avg_fps > 15:
        print(f"  • Good performance: {avg_fps:.1f} FPS")
    else:
        print(f"  • Performance: {avg_fps:.1f} FPS")

    print("="*60)


def run_inference(model_path: str, config_path: str, output_file: Optional[str] = None) -> Dict[str, Any]:
    """
    Main inference function.

    Args:
        model_path: Path to trained model file
        config_path: Path to configuration file
        output_file: Optional output file path

    Returns:
        Dictionary with inference results and statistics
    """
    print("Starting ZaloAI25 YOLO inference")
    print(f"Model: {os.path.basename(model_path)}")

    # Load and validate configuration
    config = load_config(config_path)
    validate_config(config)

    # Check GPU availability
    check_gpu_availability()

    # Load model
    model = load_model(model_path)

    # Get class mapping and inference configuration
    class_mapping = config.get('class_mapping', {})
    inference_config = config['inference']

    if not class_mapping:
        print("Error: No class_mapping found in configuration")
        raise ValueError("Missing class_mapping in configuration")

    # Find test videos with class information
    video_info = find_test_videos(config['paths'], class_mapping)

    if not video_info:
        print("Error: No test videos found for inference")
        raise FileNotFoundError("No test videos found")

    # Prepare output file
    if output_file is None:
        output_dir = config['paths'].get('inference_output', 'inference_results')
        os.makedirs(output_dir, exist_ok=True)
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        output_file = os.path.join(output_dir, f'predictions_{timestamp}.json')

    # Process videos
    all_detections = []
    all_stats = []
    failed_videos = []

    start_time = time.time()

    with tqdm(total=len(video_info), desc="Processing videos", unit="video") as pbar:
        for video_folder, class_name, class_id in video_info:
            video_id = os.path.basename(video_folder)
            video_path = find_video_file(video_folder)

            if not video_path:
                print(f"Warning: No video file found in {video_folder}")
                failed_videos.append((video_id, class_name))
                pbar.update(1)
                continue

            try:
                detections, stats = process_video(
                    model, video_path, inference_config, class_id
                )
                all_detections.append(detections)
                all_stats.append(stats)

                pbar.set_postfix({
                    'Video': video_id,
                    'Class': class_name,
                    'FPS': f"{stats.get('fps', 0):.1f}",
                    'Dets': stats.get('total_detections', 0)
                })

            except Exception as e:
                print(f"Error: Failed to process {video_id} ({class_name}): {e}")
                failed_videos.append((video_id, class_name))
                all_detections.append([])
                all_stats.append({'error': str(e)})

            pbar.update(1)

    total_processing_time = time.time() - start_time

    # Format predictions
    predictions = format_predictions(video_info, all_detections, all_stats)

    # Save predictions
    save_predictions(predictions, output_file)
    print(f"Predictions saved to: {output_file}")

    # Print summary
    print_inference_summary(predictions, all_stats, total_processing_time)

    # Create inference summary
    inference_summary = {
        'model_path': model_path,
        'config_path': config_path,
        'output_file': output_file,
        'inference_timestamp': datetime.now().isoformat(),
        'total_processing_time': total_processing_time,
        'videos_processed': len(video_info) - len(failed_videos),
        'failed_videos': failed_videos,
        'total_frames_processed': sum(stats.get('processed_frames', 0) for stats in all_stats),
        'total_detections': sum(stats.get('total_detections', 0) for stats in all_stats),
        'average_fps': sum(stats.get('fps', 0) for stats in all_stats if 'error' not in stats) / len([s for s in all_stats if 'error' not in stats]) if all_stats and any('error' not in s for s in all_stats) else 0,
        'inference_config': inference_config,
        'predictions': predictions
    }

    # Save inference summary
    summary_file = output_file.replace('.json', '_summary.json')
    with open(summary_file, 'w') as f:
        json.dump(inference_summary, f, indent=2)

    return inference_summary


def main():
    """Main entry point for inference script."""
    parser = argparse.ArgumentParser(
        description="ZaloAI25 YOLO Inference Script",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python -m utils.infer.infer --model runs/detect/exp/weights/best.pt --config configs/infer.yaml
  python -m utils.infer.infer --model models/best_model.pt --config configs/infer.yaml --output predictions.json
  python -m utils.infer.infer --model runs/detect/exp/weights/last.pt --config configs/infer.yaml --confidence 0.3
        """
    )

    parser.add_argument(
        "--model", "-m",
        type=str,
        required=True,
        help="Path to trained model file (.pt)"
    )

    parser.add_argument(
        "--config", "-c",
        type=str,
        default="configs/infer.yaml",
        help="Path to inference configuration file (default: configs/infer.yaml)"
    )

    parser.add_argument(
        "--output", "-o",
        type=str,
        help="Output JSON file path"
    )

    
    parser.add_argument(
        "--confidence",
        type=float,
        default=None,
        help="Confidence threshold (overrides config)"
    )

    parser.add_argument(
        "--iou-threshold",
        type=float,
        default=None,
        help="IoU threshold for NMS (overrides config)"
    )

    args = parser.parse_args()

    try:
        # Load and update config
        config = load_config(args.config)

        # Override inference parameters if specified
        if args.confidence is not None:
            config['inference']['confidence'] = args.confidence

        if args.iou_threshold is not None:
            config['inference']['iou_threshold'] = args.iou_threshold

        # Save updated config temporarily
        temp_config_path = args.config
        if args.confidence is not None or args.iou_threshold is not None:
            temp_config_path = args.config.replace('.yaml', '_temp.yaml')
            with open(temp_config_path, 'w') as f:
                yaml.dump(config, f, default_flow_style=False)

        try:
            # Run inference
            results = run_inference(
                model_path=args.model,
                config_path=temp_config_path,
                output_file=args.output
            )

            print(f"\n✅ Inference complete!")

        finally:
            # Clean up temporary config
            if temp_config_path != args.config and os.path.exists(temp_config_path):
                os.remove(temp_config_path)

    except Exception as e:
        print(f"Error: Inference failed: {e}")
        print(f"\nERROR: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()