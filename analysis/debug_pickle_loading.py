"""
Debug script to check what's in the pickle file and why features aren't being extracted.
"""

import pickle
import pandas as pd
import numpy as np
import os

# Configuration
test_results_dir = os.path.expanduser("~/Desktop/rPPG-Toolbox/scripts/test_runs/exp/PURE_ClipLength240_DataTypeDiffNormalized_Standardized_DataAugNone_LabelTypeDiffNormalized_Crop_faceTrue_BackendHC_Large_boxTrue_Large_size1.5_Dyamic_DetFalse_det_len30_Median_face_boxFalse_SizeW72_SizeH72")
test_results_dir = os.path.abspath(test_results_dir)

saved_outputs_dir = os.path.join(test_results_dir, 'saved_test_outputs')

print(f"Looking in: {saved_outputs_dir}")
print(f"Directory exists: {os.path.exists(saved_outputs_dir)}")

if os.path.exists(saved_outputs_dir):
    print(f"\nFiles in directory:")
    for f in os.listdir(saved_outputs_dir):
        print(f"  - {f}")
    
    # Find pickle file
    pickle_files = [f for f in os.listdir(saved_outputs_dir) if f.endswith('.pickle')]
    print(f"\nPickle files found: {pickle_files}")
    
    if pickle_files:
        pickle_path = os.path.join(saved_outputs_dir, pickle_files[0])
        print(f"\nLoading: {pickle_path}")
        
        # Load pickle
        with open(pickle_path, 'rb') as f:
            data = pickle.load(f)
        
        print(f"\nPickle contents (keys): {data.keys()}")
        
        predictions = data['predictions']
        labels = data['labels']
        
        print(f"\nPredictions type: {type(predictions)}")
        print(f"Labels type: {type(labels)}")
        
        print(f"\nNumber of videos in predictions: {len(predictions)}")
        print(f"Number of videos in labels: {len(labels)}")
        
        # Show first few video IDs
        pred_videos = list(predictions.keys())[:5]
        label_videos = list(labels.keys())[:5]
        
        print(f"\nFirst 5 video IDs in predictions: {pred_videos}")
        print(f"First 5 video IDs in labels: {label_videos}")
        
        # Check first video in detail
        if len(label_videos) > 0:
            first_video = label_videos[0]
            print(f"\n--- Examining video: {first_video} ---")
            print(f"Type: {type(first_video)}")
            print(f"Number of chunks: {len(labels[first_video])}")
            
            # Get first chunk
            chunk_indices = list(labels[first_video].keys())
            print(f"Chunk indices: {chunk_indices[:10]}...")
            
            if len(chunk_indices) > 0:
                first_chunk = chunk_indices[0]
                chunk_data = labels[first_video][first_chunk]
                
                print(f"\n--- Examining chunk {first_chunk} ---")
                print(f"Type: {type(chunk_data)}")
                print(f"Shape: {chunk_data.shape if hasattr(chunk_data, 'shape') else 'N/A'}")
                print(f"Dtype: {chunk_data.dtype if hasattr(chunk_data, 'dtype') else 'N/A'}")
                
                # Convert to numpy if needed
                if hasattr(chunk_data, 'numpy'):
                    chunk_data = chunk_data.numpy()
                else:
                    chunk_data = np.array(chunk_data)
                
                print(f"After conversion - Shape: {chunk_data.shape}")
                print(f"After conversion - Dtype: {chunk_data.dtype}")
                print(f"Sample values: {chunk_data.flatten()[:10]}")
        
        # Load metrics CSV
        print("\n" + "="*80)
        metrics_files = [f for f in os.listdir(saved_outputs_dir) if 'per_chunk' in f and f.endswith('.csv')]
        print(f"Metrics CSV files: {metrics_files}")
        
        if metrics_files:
            metrics_path = os.path.join(saved_outputs_dir, metrics_files[0])
            metrics_df = pd.read_csv(metrics_path)
            
            print(f"\nMetrics DataFrame shape: {metrics_df.shape}")
            print(f"Columns: {list(metrics_df.columns)}")
            print(f"\nFirst few rows:")
            print(metrics_df.head())
            
            print(f"\nVideo ID types in metrics:")
            print(f"  Type: {type(metrics_df['video_id'].iloc[0])}")
            print(f"  Sample values: {metrics_df['video_id'].head().tolist()}")
            
            # Check for matching video IDs
            metrics_videos = set(metrics_df['video_id'].astype(str).unique())
            label_videos_set = set(str(v) for v in labels.keys())
            
            print(f"\nVideo ID matching:")
            print(f"  Videos in metrics: {len(metrics_videos)}")
            print(f"  Videos in labels: {len(label_videos_set)}")
            print(f"  Overlap: {len(metrics_videos & label_videos_set)}")
            print(f"  In metrics but not in labels: {metrics_videos - label_videos_set}")
            print(f"  In labels but not in metrics: {label_videos_set - metrics_videos}")
            
else:
    print(f"\nError: Directory does not exist!")
    print(f"Please check the path.")

