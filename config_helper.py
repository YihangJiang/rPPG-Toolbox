"""
Configuration Helper for Hierarchical Config System

This module provides functions to automatically load dataset templates
and merge them with experiment configurations, eliminating redundancy.
"""

import os
import yaml
from yacs.config import CfgNode as CN


def load_dataset_config(dataset_name):
    """
    Load dataset configuration template.
    
    Args:
        dataset_name: Name of the dataset (e.g., 'UBFC-rPPG', 'PURE')
        
    Returns:
        Dictionary with dataset configuration or None if not found
    """
    # Map dataset names to config file names
    dataset_file_map = {
        'UBFC-rPPG': 'ubfc_rppg.yaml',
        'UBFC-rPPG-IN': 'ubfc_rppg_in.yaml',  # Infraorbital variant
        'UBFC-PHYS': 'ubfc_phys.yaml',
        'UBFC-PHYS-IN': 'ubfc_phys_in.yaml',  # Infraorbital variant
        'PURE': 'pure.yaml',
        'PURE-IN': 'pure_in.yaml',  # Infraorbital variant (if exists)
        'SCAMPS': 'scamps.yaml',
        'MMPD': 'mmpd.yaml',
        'BP4DPlus': 'bp4dplus.yaml',
        'BP4DPlusBigSmall': 'bp4dplus_bigsmall.yaml',
        'iBVP': 'ibvp.yaml',
    }
    
    if dataset_name not in dataset_file_map:
        return None
    
    dataset_file = dataset_file_map[dataset_name]
    
    # Try to find the config file
    script_dir = os.path.dirname(os.path.abspath(__file__))
    dataset_config_path = os.path.join(script_dir, 'configs', 'base', 'datasets', dataset_file)
    
    if not os.path.exists(dataset_config_path):
        print(f"Warning: Dataset config not found: {dataset_config_path}")
        return None
    
    # Load the dataset template
    with open(dataset_config_path, 'r') as f:
        dataset_yaml = yaml.load(f, Loader=yaml.FullLoader)
    
    if 'DATA_TEMPLATE' in dataset_yaml:
        return dataset_yaml['DATA_TEMPLATE']
    
    return None


def merge_dataset_config(config, section_name, overrides=None):
    """
    Merge dataset template into a config section (TRAIN.DATA, VALID.DATA, TEST.DATA).
    
    Args:
        config: Configuration object
        section_name: Section to update ('TRAIN', 'VALID', 'TEST')
        overrides: Dictionary of values to override from experiment config
        
    Returns:
        None (modifies config in-place)
    """
    config.defrost()
    
    # Get the DATA section
    if section_name == 'TRAIN':
        data_section = config.TRAIN.DATA
    elif section_name == 'VALID':
        data_section = config.VALID.DATA
    elif section_name == 'TEST':
        data_section = config.TEST.DATA
    else:
        return
    
    # Get dataset name from either the config or overrides
    dataset_name = data_section.DATASET if hasattr(data_section, 'DATASET') else None
    if overrides and 'DATASET' in overrides:
        dataset_name = overrides['DATASET']
    
    if not dataset_name:
        config.freeze()
        return
    
    # Load dataset template
    dataset_config = load_dataset_config(dataset_name)
    
    if dataset_config:
        # Merge dataset template first using CN constructor
        temp_cn = CN(dataset_config)
        data_section.merge_from_other_cfg(temp_cn)
        
        # Then apply overrides from experiment config
        if overrides:
            override_cn = CN(overrides)
            data_section.merge_from_other_cfg(override_cn)
        
        # print(f'=> Auto-loaded dataset template for {dataset_name} into {section_name}.DATA')
    
    config.freeze()


def apply_dataset_templates(config):
    """
    Automatically apply dataset templates to all data sections.
    This should be called after loading the experiment config.
    
    Args:
        config: Configuration object with TRAIN.DATA.DATASET, etc. specified
    """
    sections = ['TRAIN', 'VALID', 'TEST']
    
    for section in sections:
        if hasattr(config, section):
            section_obj = getattr(config, section)
            if hasattr(section_obj, 'DATA') and hasattr(section_obj.DATA, 'DATASET'):
                dataset_name = section_obj.DATA.DATASET
                if dataset_name:
                    # Store experiment-specific overrides before merging
                    config.defrost()
                    data_config = getattr(config, section).DATA
                    
                    # Collect experiment-specific settings to preserve
                    overrides = {}
                    if hasattr(data_config, 'BEGIN'):
                        overrides['BEGIN'] = data_config.BEGIN
                    if hasattr(data_config, 'END'):
                        overrides['END'] = data_config.END
                    
                    # Preserve PREPROCESS settings if they exist in experiment config
                    # But exclude default values that weren't explicitly set
                    preprocess_override = None
                    if hasattr(data_config, 'PREPROCESS'):
                        # Create a deep copy of PREPROCESS config
                        preprocess_override = CN(data_config.PREPROCESS)
                        
                        # Remove RESIZE if it matches base defaults (128x128)
                        # This prevents default values from overriding dataset template values
                        # if hasattr(preprocess_override, 'RESIZE'):
                        #     resize_h = preprocess_override.RESIZE.H if hasattr(preprocess_override.RESIZE, 'H') else None
                        #     resize_w = preprocess_override.RESIZE.W if hasattr(preprocess_override.RESIZE, 'W') else None
                        #     # If RESIZE matches base defaults, remove it so dataset template value is used
                        #     if resize_h == 128 and resize_w == 128:
                        #         del preprocess_override.RESIZE
                    
                    # Load and merge dataset template
                    dataset_config = load_dataset_config(dataset_name)
                    if dataset_config:
                        # Create temp config from dict using CN's constructor
                        temp_cn = CN(dataset_config)
                        data_config.merge_from_other_cfg(temp_cn)
                        
                        # Re-apply overrides (BEGIN, END)
                        for key, value in overrides.items():
                            setattr(data_config, key, value)
                        
                        # Re-apply PREPROCESS settings, merging to preserve structure
                        if preprocess_override is not None:
                            # Merge PREPROCESS settings, allowing experiment config to override template
                            # (RESIZE with default 128x128 has been removed, so dataset template value will be used)
                            data_config.PREPROCESS.merge_from_other_cfg(preprocess_override)
                                            
                    config.freeze()

