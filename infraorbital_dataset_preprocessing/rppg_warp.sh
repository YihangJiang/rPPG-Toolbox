#!/usr/bin/bash

#SBATCH --job-name=rppg_warp
#SBATCH --output=res_rppg.txt
#SBATCH --error=res_rppg.err
#SBATCH --time=800:00
#SBATCH --mem=100G


#SBATCH -p gpu-common --gres=gpu:1
#SBATCH --exclusive
start_time=$(date "+%Y-%m-%d %H:%M:%S")
echo "Script started at: $start_time"

python rppg_warp.py >> kk.txt

end_time=$(date "+%Y-%m-%d %H:%M:%S")
echo "Script ended at: $end_time"
