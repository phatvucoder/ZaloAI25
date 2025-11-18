#!/usr/bin/env python3
"""
Relabel.py - Universal Label Mapping Script for ZaloAI25 Project

This script maps all object classes to a single "target_object" label (class_id: 0)
across the entire project, including:
- Configuration files (YAML)
- YOLO annotation files (.txt)
- Future YOLO datasets

Usage:
    python relabel.py --dry-run     # Preview changes without applying
    python relabel.py --execute     # Apply all changes
    python relabel.py --backup      # Create backups before modifying (default with --execute)
"""

import argparse
import glob
import os
import shutil
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import yaml


class Relabeler:
    """Main class for handling universal label mapping to target_object."""

    def __init__(self, project_root: str = None):
        """Initialize relabeler with project root directory."""
        if project_root is None:
            # If script is in scripts/ directory, use parent directory as project root
            script_dir = Path(__file__).parent
            if script_dir.name == "scripts":
                self.project_root = script_dir.parent
            else:
                self.project_root = script_dir
        else:
            self.project_root = Path(project_root)

        self.stats = {
            'config_files_updated': 0,
            'yolo_files_processed': 0,
            'annotations_relabelled': 0,
            'files_backed_up': 0,
            'errors': 0
        }

        # Target configuration
        self.target_class_id = 0
        self.target_class_name = "target_object"

    def update_configurations(self, execute: bool = True, backup: bool = True) -> bool:
        """
        Update configuration files to map all classes to target_object.

        Args:
            execute: Whether to apply changes or just preview
            backup: Whether to create backup files

        Returns:
            bool: Success status
        """
        print("🔧 Updating configuration files...")

        config_files = [
            self.project_root / "configs" / "data.yaml",
            self.project_root / "configs" / "infer.yaml"
        ]

        success = True

        for config_file in config_files:
            if not config_file.exists():
                print(f"⚠️  Config file not found: {config_file}")
                continue

            try:
                with open(config_file, 'r', encoding='utf-8') as f:
                    config = yaml.safe_load(f)

                # Create backup if needed
                if backup and execute:
                    backup_file = config_file.with_suffix('.yaml.backup')
                    shutil.copy2(config_file, backup_file)
                    self.stats['files_backed_up'] += 1
                    print(f"📋 Backup created: {backup_file}")

                # Update class mapping based on config type
                original_config = config.copy()

                if config_file.name == "data.yaml":
                    config = self._update_data_config(config)
                elif config_file.name == "infer.yaml":
                    config = self._update_infer_config(config)

                # Show changes
                if not execute:
                    self._show_config_changes(config_file, original_config, config)
                else:
                    with open(config_file, 'w', encoding='utf-8') as f:
                        yaml.dump(config, f, default_flow_style=False, sort_keys=False)

                    print(f"✅ Updated: {config_file}")
                    self.stats['config_files_updated'] += 1

            except Exception as e:
                print(f"❌ Error updating {config_file}: {e}")
                success = False
                self.stats['errors'] += 1

        return success

    def _update_data_config(self, config: Dict) -> Dict:
        """Update data.yaml configuration to map all classes to target_object."""
        if 'class_mapping' in config:
            for class_name in config['class_mapping']:
                config['class_mapping'][class_name] = {
                    'new_id': self.target_class_id,
                    'new_name': self.target_class_name
                }

        return config

    def _update_infer_config(self, config: Dict) -> Dict:
        """Update infer.yaml configuration to map all classes to target_object."""
        if 'class_mapping' in config:
            for class_name in config['class_mapping']:
                config['class_mapping'][class_name] = self.target_class_id

        return config

    def _show_config_changes(self, config_file: Path, original: Dict, updated: Dict):
        """Show configuration changes in dry-run mode."""
        print(f"\n📄 Changes for {config_file.name}:")

        if 'class_mapping' in original and 'class_mapping' in updated:
            orig_mapping = original['class_mapping']
            updated_mapping = updated['class_mapping']

            for class_name in orig_mapping:
                orig_value = orig_mapping[class_name]
                updated_value = updated_mapping[class_name]

                if orig_value != updated_value:
                    print(f"  {class_name}: {orig_value} → {updated_value}")

    def find_yolo_files(self) -> List[Path]:
        """
        Find all YOLO annotation files using glob.glob recursive search.

        Returns:
            List of paths to YOLO .txt files
        """
        print("🔍 Searching for YOLO annotation files...")

        # Search locations
        search_paths = [
            self.project_root / "dataset" / "yolo_dataset",
            self.project_root / "dataset" / "yolo_subset",
            self.project_root / "dataset"  # Search recursively in entire dataset folder
        ]

        yolo_files = []

        for search_path in search_paths:
            if not search_path.exists():
                continue

            # Use glob.glob recursively to find all .txt files
            pattern = str(search_path / "**" / "*.txt")
            found_files = glob.glob(pattern, recursive=True)

            # Filter to only YOLO annotation files (contain numerical data)
            for file_path in found_files:
                path_obj = Path(file_path)
                if self._is_yolo_annotation_file(path_obj):
                    yolo_files.append(path_obj)

        print(f"📁 Found {len(yolo_files)} YOLO annotation files")
        return yolo_files

    def _is_yolo_annotation_file(self, file_path: Path) -> bool:
        """
        Check if a file is a valid YOLO annotation file.

        Args:
            file_path: Path to the file to check

        Returns:
            bool: True if file appears to be a YOLO annotation file
        """
        try:
            if not file_path.is_file() or file_path.stat().st_size == 0:
                return False

            with open(file_path, 'r', encoding='utf-8') as f:
                first_line = f.readline().strip()

            # YOLO format: class_id x_center y_center width height
            # Should have 5 numerical values
            if not first_line:
                return False

            parts = first_line.split()
            if len(parts) != 5:
                return False

            # Check if all parts are numbers
            try:
                [float(x) for x in parts]
                return True
            except ValueError:
                return False

        except Exception:
            return False

    def relabel_yolo_files(self, execute: bool = True, backup: bool = True) -> bool:
        """
        Relabel all YOLO annotation files to use class_id 0.

        Args:
            execute: Whether to apply changes or just preview
            backup: Whether to create backup files

        Returns:
            bool: Success status
        """
        yolo_files = self.find_yolo_files()

        if not yolo_files:
            print("ℹ️  No YOLO annotation files found. Run build_yolo.py first if needed.")
            return True

        print(f"🔄 Processing {len(yolo_files)} YOLO files...")

        success = True

        for yolo_file in yolo_files:
            try:
                # Read original file
                with open(yolo_file, 'r', encoding='utf-8') as f:
                    lines = f.readlines()

                if not lines:
                    continue

                # Create backup if needed
                if backup and execute:
                    backup_file = yolo_file.with_suffix('.txt.backup')
                    shutil.copy2(yolo_file, backup_file)
                    self.stats['files_backed_up'] += 1

                # Process lines
                new_lines = []
                annotations_changed = 0

                for line in lines:
                    line = line.strip()
                    if not line:
                        new_lines.append('\n')
                        continue

                    parts = line.split()
                    if len(parts) == 5:
                        # Replace class_id with 0
                        if parts[0] != '0':
                            annotations_changed += 1
                        parts[0] = '0'
                        new_lines.append(' '.join(parts) + '\n')
                    else:
                        # Keep invalid lines as-is
                        new_lines.append(line + '\n')

                # Show changes in dry-run mode
                if not execute and annotations_changed > 0:
                    rel_path = yolo_file.relative_to(self.project_root)
                    print(f"  📝 {rel_path}: {annotations_changed} annotations → class_id 0")
                    self.stats['annotations_relabelled'] += annotations_changed

                # Apply changes
                if execute:
                    with open(yolo_file, 'w', encoding='utf-8') as f:
                        f.writelines(new_lines)

                    if annotations_changed > 0:
                        rel_path = yolo_file.relative_to(self.project_root)
                        print(f"  ✅ {rel_path}: {annotations_changed} annotations relabelled")
                        self.stats['annotations_relabelled'] += annotations_changed

                    self.stats['yolo_files_processed'] += 1

            except Exception as e:
                print(f"❌ Error processing {yolo_file}: {e}")
                success = False
                self.stats['errors'] += 1

        return success

    def print_statistics(self):
        """Print processing statistics."""
        print("\n" + "="*50)
        print("📊 PROCESSING STATISTICS")
        print("="*50)
        print(f"🔧 Configuration files updated: {self.stats['config_files_updated']}")
        print(f"📁 YOLO files processed: {self.stats['yolo_files_processed']}")
        print(f"🏷️  Annotations relabelled: {self.stats['annotations_relabelled']}")
        print(f"📋 Backup files created: {self.stats['files_backed_up']}")
        print(f"❌ Errors encountered: {self.stats['errors']}")
        print("="*50)

    def run(self, execute: bool = False, backup: bool = True):
        """
        Run the complete relabeling process.

        Args:
            execute: Whether to apply changes (False = dry-run)
            backup: Whether to create backups
        """
        mode = "EXECUTE" if execute else "DRY RUN"
        print(f"🚀 Starting universal relabeling - {mode} MODE")
        print(f"📂 Project root: {self.project_root}")
        print(f"🎯 Target: class_id {self.target_class_id} ('{self.target_class_name}')")
        print()

        # Step 1: Update configurations
        config_success = self.update_configurations(execute=execute, backup=backup)

        print()

        # Step 2: Relabel YOLO files
        yolo_success = self.relabel_yolo_files(execute=execute, backup=backup)

        print()

        # Print statistics
        self.print_statistics()

        if not execute:
            print("\n💡 This was a DRY RUN. No files were modified.")
            print("   Use --execute to apply the changes.")

        if config_success and yolo_success and self.stats['errors'] == 0:
            print("\n✅ All operations completed successfully!")
            return 1 if execute else 0
        else:
            print("\n⚠️  Some operations encountered errors. Please review the output above.")
            return 2


def main():
    """Main entry point for the relabel.py script."""
    parser = argparse.ArgumentParser(
        description="Map all object classes to 'target_object' (class_id: 0)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    python relabel.py --dry-run                 # Preview changes
    python relabel.py --execute                 # Apply changes with backup
    python relabel.py --execute --no-backup     # Apply changes without backup
    python relabel.py --project-root /path/to/project  # Custom project path
        """
    )

    parser.add_argument(
        '--execute',
        action='store_true',
        help='Execute the relabeling (default: dry-run mode)'
    )

    parser.add_argument(
        '--dry-run',
        action='store_true',
        help='Preview changes without applying them (default mode)'
    )

    parser.add_argument(
        '--no-backup',
        action='store_true',
        help='Skip creating backup files (use with caution)'
    )

    parser.add_argument(
        '--project-root',
        type=str,
        default=None,
        help='Path to project root directory (default: script directory)'
    )

    args = parser.parse_args()

    # Initialize relabeler
    relabeler = Relabeler(project_root=args.project_root)

    # Run the process
    execute_mode = args.execute and not args.dry_run
    backup_enabled = not args.no_backup and execute_mode
    exit_code = relabeler.run(execute=execute_mode, backup=backup_enabled)

    sys.exit(exit_code)


if __name__ == "__main__":
    main()