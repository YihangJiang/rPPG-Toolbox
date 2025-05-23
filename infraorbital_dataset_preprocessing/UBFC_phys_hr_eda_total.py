# %%
import os
import pandas as pd
import matplotlib.pyplot as plt

# Directory containing the CSV files
data_dir = '/hpc/group/dunnlab/rppg_data/data/meta_d_1'

# Read and merge CSV files
csv_files = [f for f in os.listdir(data_dir) if f.endswith('.csv')]

# Initialize an empty DataFrame
merged_df = pd.DataFrame()

for file in csv_files:
    file_path = os.path.join(data_dir, file)
    temp_df = pd.read_csv(file_path)
    merged_df = pd.concat([merged_df, temp_df], ignore_index=True)

# Plot histogram of heart rate
plt.figure(figsize=(10, 6))
plt.hist(merged_df['heart_rate'], bins=15, alpha=0.7, color='blue', edgecolor='black')
plt.xlabel('Heart Rate ')
plt.ylabel('Frequency')
plt.title('Distribution of Heart Rate Across Subjects')
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.tight_layout()
plt.show()

# Calculate the number of observations per subject
observations_per_subject = merged_df.groupby('subject_number').size()
# Calculate observations per subject across heart rate ranges with a gap of 20 bpm
hr_bins = range(0, int(merged_df['heart_rate'].max()) + 20, 20)
merged_df['hr_range'] = pd.cut(merged_df['heart_rate'], bins=hr_bins)

# Calculate observations per subject across heart rate ranges
observations_per_subject_hr_range = merged_df.groupby(['subject_number', 'hr_range']).size()
print("Observations per subject across heart rate ranges:")
print(observations_per_subject_hr_range)

print("Observations per subject:")
print(observations_per_subject)


# %%
