# %%
import os
import cv2
import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from tqdm import tqdm
import matplotlib.pyplot as plt
import seaborn as sns
from video_utils import * 

# ================= Feature extractors ==================
def luminance_blur_torch(frames_np):
    frames = torch.from_numpy(frames_np).permute(0,3,1,2).float().to("cuda") / 255.
    gray = (0.2989*frames[:,0]+0.5870*frames[:,1]+0.1140*frames[:,2])
    lum_mean = gray.mean().item()
    lum_std  = gray.std().item()

    # simple Laplacian blur metric
    lap_kernel = torch.tensor([[0,1,0],[1,-4,1],[0,1,0]],dtype=torch.float32,device="cuda").view(1,1,3,3)
    lap = torch.nn.functional.conv2d(gray.unsqueeze(1), lap_kernel, padding=1)
    blur = (lap**2).mean().sqrt().item()
    return lum_mean, lum_std, blur

def optical_flow_features(frames):
    grays = [cv2.cvtColor(f, cv2.COLOR_RGB2GRAY) for f in frames]
    mags = []
    for i in range(1, len(grays)):
        flow = cv2.calcOpticalFlowFarneback(
            grays[i-1], grays[i], None, 0.5, 3, 15, 3, 5, 1.2, 0)
        mag, _ = cv2.cartToPolar(flow[..., 0], flow[..., 1])
        mags.append(np.mean(mag))
    return np.mean(mags), np.std(mags)


def luminance_features(frames):
    grays = [cv2.cvtColor(f, cv2.COLOR_RGB2GRAY) for f in frames]
    lum = np.array([np.mean(g) for g in grays])
    return np.mean(lum), np.std(lum)


def blur_feature(frames):
    return np.mean([
        cv2.Laplacian(cv2.cvtColor(f, cv2.COLOR_RGB2GRAY), cv2.CV_64F).var()
        for f in frames
    ])

def luminance_blur_torch(frames_np):
    frames = torch.from_numpy(frames_np).permute(0,3,1,2).float().to("cuda") / 255.
    gray = (0.2989*frames[:,0]+0.5870*frames[:,1]+0.1140*frames[:,2])
    lum_mean = gray.mean().item()
    lum_std  = gray.std().item()

    # simple Laplacian blur metric
    lap_kernel = torch.tensor([[0,1,0],[1,-4,1],[0,1,0]],dtype=torch.float32,device="cuda").view(1,1,3,3)
    lap = torch.nn.functional.conv2d(gray.unsqueeze(1), lap_kernel, padding=1)
    blur = (lap**2).mean().sqrt().item()
    return lum_mean, lum_std, blur

# ================= Main analysis ==================
# %%
src_root="/work/yj167/DATASET_1"
dst_root="/work/yj167/DATASET_1IN"
per_vid_path = "/hpc/group/dunnlab/rppg_data/rPPG-Toolbox/infraorbital_dataset_preprocessing/per_video_metrics.csv"

df_metrics = pd.read_csv(per_vid_path)
list_phys1, list_phys2 = get_ubfc_paths(src_root, dst_root)

print(f"Found {len(list_phys1)} total video files from UBFC datasets")
# %%
# map from video ID (string key used in per_video_metrics) to video path
video_map = {}
for path in list_phys1:
    name = os.path.splitext(os.path.basename(path))[0]  # e.g., vid_s1_T1
    # drop "vid_" if present to match metrics naming
    if name.startswith("vid_"):
        name = name[4:]
    video_map[name] = path

# %%

results = []

for _, row in tqdm(df_metrics.iterrows(), total=len(df_metrics)):
    vid_id = str(row["video_id"])
    # handle subject/task naming differences
    if vid_id in video_map:
        vid_path = video_map[vid_id]
    elif "vid_" + vid_id in video_map:
        vid_path = video_map["vid_" + vid_id]
    else:
        # try partial match
        matches = [p for k, p in video_map.items() if vid_id in k]
        if matches:
            vid_path = matches[0]
        else:
            print(f"⚠️ Video not found for ID {vid_id}")
            continue

    cap = cv2.VideoCapture(vid_path)
    frames = []
    success, frame = cap.read()
    while success and len(frames) < 100:
        frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        success, frame = cap.read()
    cap.release()
    if len(frames) < 2:
        continue

    # m_mean, m_std = optical_flow_features(frames)
    l_mean, l_std = luminance_features(frames)
    b_mean = blur_feature(frames)

    results.append({
        "video_id": vid_id,
        "video_path": vid_path,
        # "motion_mean": m_mean,
        # "motion_std": m_std,
        "luminance_mean": l_mean,
        "luminance_std": l_std,
        "blur": b_mean,
        "hr_error": abs(row["avg_gt_hr_fft"] - row["avg_pred_hr_fft"]),
        "snr": row["avg_SNR"],
        "macc": row["avg_MACC"],
    })

df_feat = pd.DataFrame(results)
df_feat.to_csv("video_features_with_errors.csv", index=False)
print(f"[INFO] Saved features to video_features_with_errors.csv ({len(df_feat)} videos)")
# %%
# --- load your feature CSV ---
df = df_feat

# --- select numeric columns only ---
numeric_df = df.select_dtypes(include=['number'])

# --- compute correlation matrix ---
corr = numeric_df.corr(method='spearman')  # or 'pearson'

# --- extract correlations with hr_error ---
corr_hr = corr[['hr_error']].sort_values(by='hr_error', ascending=False)

# --- plot as a heatmap ---
plt.figure(figsize=(6, 8))
sns.heatmap(corr_hr, annot=True, cmap='coolwarm', center=0,
            linewidths=0.5, cbar=False, fmt=".2f")
plt.title("Correlation of HR Error with Other Features")
plt.ylabel("Features")
plt.xlabel("Spearman Correlation")
plt.tight_layout()
plt.show()

