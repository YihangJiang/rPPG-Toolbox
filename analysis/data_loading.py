"""
Data loading functions for test results and cached video files.

This module provides functions to:
1. Load test results from pickle and CSV files
2. Find and cache preprocessed video files
"""

import pickle
import re
import pandas as pd
import numpy as np
import os


def load_test_data(test_results_dir):
    """Load test results from pickle and CSV files.
    
    The pickle file contains:
    - 'predictions': dict[video_id][chunk_index] -> prediction array (model outputs)
    - 'labels': dict[video_id][chunk_index] -> label array (ground truth PPG/BVP signals)
    - 'label_type': str (e.g., 'DiffNormalized')
    - 'fs': int (sampling frequency in Hz)
    
    The pickle file is saved during model testing at:
    {test_results_dir}/saved_test_outputs/{model_file_name}_outputs.pickle
    
    Args:
        test_results_dir: Directory containing test results (should contain 'saved_test_outputs' subdirectory)
        
    Returns:
        tuple: (predictions_dict, labels_dict, metrics_df, metadata)
    """
    saved_outputs_dir = os.path.join(test_results_dir, 'saved_test_outputs')
    
    if not os.path.exists(saved_outputs_dir):
        raise FileNotFoundError(
            f"Directory not found: {saved_outputs_dir}\n"
            f"The pickle file should be in: {saved_outputs_dir}\n"
            f"Expected structure: {test_results_dir}/saved_test_outputs/*_outputs.pickle\n"
            f"This file is created automatically during model testing."
        )
    
    # Find pickle file
    pickle_files = [f for f in os.listdir(saved_outputs_dir) if f.endswith('.pickle')]
    if not pickle_files:
        available_files = os.listdir(saved_outputs_dir)
        raise FileNotFoundError(
            f"No pickle file found in {saved_outputs_dir}\n"
            f"Available files: {available_files}\n"
            f"Expected file pattern: *_outputs.pickle\n"
            f"The pickle file is created during testing when config.TEST.OUTPUT_SAVE_DIR is set."
        )
    pickle_filename = pickle_files[0]
    pickle_path = os.path.join(saved_outputs_dir, pickle_filename)

    # Best-effort parse run info from filename.
    # Common pattern used by this repo: <train>_<train2>_<test>_<model>_outputs.pickle
    # Example: UBFC_UBFC_PURE_deepphys_outputs.pickle
    run_info = {
        "pickle_filename": pickle_filename,
        "model_name": None,
        "train_dataset": None,
        "test_dataset": None,
    }
    m = re.match(r"^(?P<stem>.+?)_outputs\.pickle$", pickle_filename)
    if m:
        stem = m.group("stem")
        parts = stem.split("_")
        if len(parts) >= 2:
            run_info["model_name"] = parts[-1]
        if len(parts) >= 3:
            run_info["test_dataset"] = parts[-2]
        if len(parts) >= 4:
            # Everything before <test>_<model> is considered training descriptor.
            run_info["train_dataset"] = "_".join(parts[:-2])
    
    # Load pickle
    with open(pickle_path, 'rb') as f:
        data = pickle.load(f)
    
    predictions = data['predictions']
    labels = data['labels']
    fs = data.get('fs', 30)
    label_type = data.get('label_type', 'DiffNormalized')
    
    # Load metrics CSV (may not contain all chunks, only those with metrics calculated)
    metrics_files = [f for f in os.listdir(saved_outputs_dir) if 'per_chunk' in f and f.endswith('.csv')]
    if not metrics_files:
        print(f"Warning: No per_chunk metrics CSV found in {saved_outputs_dir}")
        metrics_df = None
    else:
        metrics_path = os.path.join(saved_outputs_dir, metrics_files[0])
        metrics_df = pd.read_csv(metrics_path)
    
    return predictions, labels, metrics_df, {'fs': fs, 'label_type': label_type, "run_info": run_info}


def find_video_cache_file(base_cached_path, video_id, chunk_index, file_list_cache=None, exp_data_name=None):
    """Find the cached video file for a given video_id and chunk_index.
    
    Args:
        base_cached_path: Base path to cached preprocessed data (e.g., DATASET_PRE)
        video_id: Video/subject ID
        chunk_index: Chunk index within the video
        file_list_cache: Optional dict mapping (video_id, chunk_index) to file paths
        exp_data_name: Optional EXP_DATA_NAME to search in subdirectory
        
    Returns:
        str or None: Path to the cached video file, or None if not found
    """
    # First check cache if provided
    if file_list_cache is not None:
        key = (str(video_id), int(chunk_index))
        if key in file_list_cache:
            file_path = file_list_cache[key]
            if os.path.exists(file_path):
                return file_path
    
    # Try different naming patterns
    patterns = [
        f"{video_id}_input{chunk_index}.npy",
        f"subject{video_id}_input{chunk_index}.npy",
        f"{int(video_id):04d}_input{chunk_index}.npy",
    ]
    
    # If exp_data_name is provided, try in that subdirectory first
    if exp_data_name:
        exp_dir = os.path.join(base_cached_path, exp_data_name)
        if os.path.exists(exp_dir):
            for pattern in patterns:
                file_path = os.path.join(exp_dir, pattern)
                if os.path.exists(file_path):
                    return file_path
    
    # Try in base cached_path
    for pattern in patterns:
        file_path = os.path.join(base_cached_path, pattern)
        if os.path.exists(file_path):
            return file_path
    
    # Search in subdirectories (EXP_DATA_NAME folders)
    if os.path.exists(base_cached_path):
        for item in os.listdir(base_cached_path):
            # Skip DataFileLists directory
            if item == 'DataFileLists':
                continue
            item_path = os.path.join(base_cached_path, item)
            if os.path.isdir(item_path):
                for pattern in patterns:
                    file_path = os.path.join(item_path, pattern)
                    if os.path.exists(file_path):
                        return file_path
    
    # If exact match not found, search in file list
    # DataFileLists are in DATASET_PRE/DataFileLists (not in subdirectories)
    file_list_dir = os.path.join(base_cached_path, 'DataFileLists')
    
    if os.path.exists(file_list_dir):
        for file_list in os.listdir(file_list_dir):
            if file_list.endswith('.csv'):
                file_list_path = os.path.join(file_list_dir, file_list)
                try:
                    df = pd.read_csv(file_list_path)
                    for input_file in df['input_files']:
                        filename = os.path.basename(input_file)
                        # Check if this file matches our video_id and chunk_index
                        if f"{video_id}" in filename and f"input{chunk_index}" in filename:
                            # Check if file exists
                            if os.path.exists(input_file):
                                return input_file
                            # Try relative to base_cached_path
                            rel_path = os.path.join(base_cached_path, os.path.basename(input_file))
                            if os.path.exists(rel_path):
                                return rel_path
                            # Try in subdirectories (EXP_DATA_NAME)
                            if os.path.exists(base_cached_path):
                                for item in os.listdir(base_cached_path):
                                    if item == 'DataFileLists':
                                        continue
                                    item_path = os.path.join(base_cached_path, item)
                                    if os.path.isdir(item_path):
                                        test_path = os.path.join(item_path, os.path.basename(input_file))
                                        if os.path.exists(test_path):
                                            return test_path
                except Exception as e:
                    continue
    
    return None


def build_file_list_cache(base_cached_path, exp_data_name=None):
    """Build a cache mapping (video_id, chunk_index) to file paths.
    
    Args:
        base_cached_path: Base path to cached preprocessed data (e.g., DATASET_PRE)
                         DataFileLists should be at: base_cached_path/DataFileLists
        exp_data_name: Optional EXP_DATA_NAME to filter CSV files. CSV files are named
                      as {EXP_DATA_NAME}_{BEGIN}_{END}{FOLD_STR}.csv
                      If provided, only CSV files starting with exp_data_name will be processed.
        
    Returns:
        dict: Mapping (video_id, chunk_index) -> file_path
    """
    cache = {}
    
    print(f"  Searching for cached files. Base path: {base_cached_path}")
    if exp_data_name:
        print(f"  Filtering for EXP_DATA_NAME: {exp_data_name}")
    
    if not os.path.exists(base_cached_path):
        print(f"  ERROR: Base cached path does not exist: {base_cached_path}")
        return cache
    
    # DataFileLists are stored directly in DATASET_PRE/DataFileLists (not in EXP_DATA_NAME subdirectory)
    file_list_dir = os.path.join(base_cached_path, 'DataFileLists')
    print(f"  Checking for DataFileLists at: {file_list_dir}")
    
    if not os.path.exists(file_list_dir):
        print(f"  ERROR: DataFileLists directory not found at: {file_list_dir}")
        print(f"  Expected structure: {base_cached_path}/DataFileLists/*.csv")
        return cache
    
    print(f"  Using DataFileLists directory: {file_list_dir}")
    all_csv_files = [f for f in os.listdir(file_list_dir) if f.endswith('.csv')]
    print(f"  Found {len(all_csv_files)} total CSV file(s) in DataFileLists")
    
    # Filter CSV files: include only if filename matches exp_data_name + _ + number (e.g. ..._0.0_1.0.csv)
    if exp_data_name:
        pattern = re.compile(r'^' + re.escape(exp_data_name) + r'_\d+\.?\d*')
        csv_files = [f for f in all_csv_files if pattern.match(f)]
        print(f"  Filtered to {len(csv_files)} CSV file(s) matching {exp_data_name}_<number>...")
        if len(csv_files) == 0:
            print(f"  WARNING: No CSV files found matching EXP_DATA_NAME '{exp_data_name}' (pattern: exp_data_name_<number>)")
            print(f"  Available CSV files: {all_csv_files[:5]}..." if len(all_csv_files) > 5 else f"  Available CSV files: {all_csv_files}")
    else:
        csv_files = all_csv_files
        print(f"  Processing all {len(csv_files)} CSV file(s)")
    
    if not csv_files:
        print(f"  ERROR: No matching CSV files found in {file_list_dir}")
        return cache
    
    files_found = 0
    files_missing = 0
    files_parsed = 0
    
    for file_list in csv_files:
        file_list_path = os.path.join(file_list_dir, file_list)
        print(f"  Processing CSV: {file_list}")
        try:
            df = pd.read_csv(file_list_path)
            print(f"    CSV has {len(df)} rows, columns: {list(df.columns)}")
            
            if 'input_files' not in df.columns:
                print(f"    ERROR: 'input_files' column not found! Available columns: {list(df.columns)}")
                continue
            
            for idx, input_file in enumerate(df['input_files']):
                if pd.isna(input_file):
                    continue
                    
                if os.path.exists(input_file):
                    files_found += 1
                    filename = os.path.basename(input_file)
                    # Parse video_id and chunk_index from filename
                    # Format: {video_id}_input{chunk_index}.npy
                    if '_input' in filename:
                        parts = filename.replace('.npy', '').split('_input')
                        if len(parts) == 2:
                            video_id = parts[0]
                            try:
                                chunk_index = int(parts[1])
                                cache[(str(video_id), chunk_index)] = input_file
                                files_parsed += 1
                            except ValueError:
                                continue
                else:
                    files_missing += 1
                    if files_missing <= 3:  # Show first 3 missing files
                        print(f"    WARNING: File does not exist: {input_file}")
            
        except Exception as e:
            print(f"    ERROR processing CSV {file_list}: {e}")
            import traceback
            traceback.print_exc()
            continue
    
    print(f"  Summary: {files_found} files exist, {files_missing} files missing, {files_parsed} files parsed into cache")
    
    return cache


def normalize_video_id(video_id):
    """Normalize video_id to match pickle format.
    
    Handles cases where CSV has '1001.0' but pickle has '1001'
    
    Args:
        video_id: Video ID (can be string, int, or float)
        
    Returns:
        str: Normalized video ID string
    """
    # Convert to string and remove trailing '.0' if present
    vid_str = str(video_id)
    # Remove trailing .0 if it's a float representation
    if vid_str.endswith('.0'):
        vid_str = vid_str[:-2]
    return vid_str


def build_metrics_lookup(metrics_df):
    """Build a lookup dictionary mapping (video_id, chunk_index) to metrics.
    
    Args:
        metrics_df: DataFrame with columns including 'video_id', 'chunk_index', 
                   'SNR', 'MACC', 'gt_hr', 'pred_hr'
        
    Returns:
        dict: Mapping (video_id, chunk_index) -> {'SNR': ..., 'MACC': ..., 'gt_hr': ..., 'pred_hr': ...}
    """
    metrics_lookup = {}
    
    if metrics_df is None:
        print(f"\nWARNING: metrics_df is None - no metrics CSV found!")
        return metrics_lookup
    
    print(f"\nBuilding metrics lookup from CSV...")
    print(f"  CSV columns: {list(metrics_df.columns)}")
    print(f"  CSV shape: {metrics_df.shape}")
    
    # Check for required columns
    required_cols = ['video_id', 'chunk_index']
    missing_cols = [col for col in required_cols if col not in metrics_df.columns]
    if missing_cols:
        print(f"  WARNING: Missing required columns: {missing_cols}")
        print(f"  Available columns: {list(metrics_df.columns)}")
    
    for idx, row in metrics_df.iterrows():
        try:
            # Normalize video_id to handle float->string conversion (e.g., '1001.0' -> '1001')
            video_id_normalized = normalize_video_id(row['video_id'])
            key = (video_id_normalized, int(row['chunk_index']))
            metrics_lookup[key] = {
                'SNR': row.get('SNR', np.nan),
                'MACC': row.get('MACC', np.nan),
                'gt_hr': row.get('gt_hr', np.nan),
                'pred_hr': row.get('pred_hr', np.nan),
            }
        except Exception as e:
            print(f"  Error processing row {idx}: {e}")
            print(f"    Row data: {row.to_dict()}")
    
    print(f"  Built lookup with {len(metrics_lookup)} entries")
    if len(metrics_lookup) > 0:
        sample_keys = list(metrics_lookup.keys())[:5]
        print(f"  Sample keys (after normalization): {sample_keys}")
    
    return metrics_lookup


def get_chunk_metrics(video_id, chunk_index, metrics_lookup):
    """Get metrics for a specific chunk from the lookup dictionary.
    
    Args:
        video_id: Video/subject ID
        chunk_index: Chunk index within the video
        metrics_lookup: Dictionary mapping (video_id, chunk_index) to metrics
        
    Returns:
        dict: Metrics dictionary with 'SNR', 'MACC', 'gt_hr', 'pred_hr'
              Returns NaN values if not found in lookup
    """
    key = (str(video_id), int(chunk_index))
    if key in metrics_lookup:
        return metrics_lookup[key]
    else:
        # If metrics not found, use NaN
        return {
            'SNR': np.nan,
            'MACC': np.nan,
            'gt_hr': np.nan,
            'pred_hr': np.nan,
        }


def calculate_hr_metrics(gt_hr, pred_hr):
    """Calculate heart rate difference metrics.
    
    Args:
        gt_hr: Ground truth heart rate
        pred_hr: Predicted heart rate
        
    Returns:
        tuple: (hr_diff, hr_abs_diff)
            - hr_diff: pred_hr - gt_hr (signed difference, positive = overestimation)
            - hr_abs_diff: |pred_hr - gt_hr| (absolute error)
    """
    if not (np.isnan(gt_hr) or np.isnan(pred_hr)):
        hr_diff = pred_hr - gt_hr
        hr_abs_diff = np.abs(hr_diff)
    else:
        hr_diff = np.nan
        hr_abs_diff = np.nan
    
    return hr_diff, hr_abs_diff


def create_chunk_info(video_id, chunk_index, metrics):
    """Create chunk info dictionary with metrics.
    
    Args:
        video_id: Video/subject ID
        chunk_index: Chunk index within the video
        metrics: Dictionary with 'SNR', 'MACC', 'gt_hr', 'pred_hr'
        
    Returns:
        dict: Chunk info dictionary with all metrics and HR differences
    """
    hr_diff, hr_abs_diff = calculate_hr_metrics(metrics['gt_hr'], metrics['pred_hr'])
    
    return {
        'video_id': str(video_id),
        'chunk_index': int(chunk_index),
        'SNR': metrics['SNR'],
        'MACC': metrics['MACC'],
        'gt_hr': metrics['gt_hr'],
        'pred_hr': metrics['pred_hr'],
        'hr_diff': hr_diff,  # pred_hr - gt_hr (signed difference)
        'hr_abs_diff': hr_abs_diff,  # |pred_hr - gt_hr| (absolute error)
    }


def print_metrics_matching_summary(metrics_found_count, metrics_not_found_count, 
                                   chunk_count, sample_not_found_keys, metrics_lookup):
    """Print summary of metrics matching between pickle file and CSV.
    
    Args:
        metrics_found_count: Number of chunks with metrics found
        metrics_not_found_count: Number of chunks without metrics
        chunk_count: Total number of chunks processed
        sample_not_found_keys: List of sample keys not found (for debugging)
        metrics_lookup: Metrics lookup dictionary
    """
    print(f"\n{'='*60}")
    print(f"Metrics Matching Summary:")
    print(f"  Metrics found: {metrics_found_count}/{chunk_count} chunks ({100*metrics_found_count/max(chunk_count,1):.1f}%)")
    print(f"  Metrics NOT found: {metrics_not_found_count}/{chunk_count} chunks ({100*metrics_not_found_count/max(chunk_count,1):.1f}%)")
    if sample_not_found_keys:
        print(f"  Sample keys NOT found in metrics CSV: {sample_not_found_keys}")
        if len(metrics_lookup) > 0:
            sample_found_keys = list(metrics_lookup.keys())[:3]
            print(f"  Sample keys FOUND in metrics CSV: {sample_found_keys}")
            print(f"  NOTE: Check if video_id/chunk_index formats match between pickle and CSV!")
    print(f"{'='*60}")

