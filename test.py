# %%
import pandas as pd
import matplotlib.pyplot as plt

# Load CSV file (replace with your actual filename)
df = pd.read_csv("face_stats.csv", index_col=[0])

# Drop duplicate rows
df_unique = df.drop_duplicates()

# Plot histogram of the last column
plt.figure(figsize=(8, 5))
plt.hist(df_unique[df_unique.columns[-1]], bins=10, edgecolor='black')
plt.title("Histogram of frames with multiple faces")
plt.xlabel("number of frames")
plt.ylabel("Count of subjects")
plt.grid(True)
plt.tight_layout()
plt.show()
# %%
