# %%
import pickle

# Path to your pickle file
file_path = "/hpc/group/dunnlab/rppg_data/rPPG-Toolbox/runs/exp/UBFC-rPPG_SizeW72_SizeH72_ClipLength180_DataTypeDiffNormalized_Standardized_DataAugNone_LabelTypeDiffNormalized_Crop_faceTrue_BackendHC_Large_boxTrue_Large_size1.5_Dyamic_DetFalse_det_len30_Median_face_boxFalse/saved_test_outputs/PURE_TSCAN_UBFC-rPPG_outputs.pickle"

# Load the pickle file
with open(file_path, "rb") as f:
    data = pickle.load(f)

# Print the structure of the loaded data
print(type(data))  # Check what type of data is stored
print(data.keys() if isinstance(data, dict) else data)  # Print keys if it's a dictionary
data.get("labels", [])
# %%
