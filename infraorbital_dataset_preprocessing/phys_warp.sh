#!/usr/bin/bash

#SBATCH --job-name=phys_warp
#SBATCH --output=res_phys.txt
#SBATCH --error=res_phys.err
#SBATCH --time=800:00
#SBATCH --mem=100G

#SBATCH -p gpu-common --gres=gpu:1
#SBATCH --exclusive
start_time=$(date "+%Y-%m-%d %H:%M:%S")
echo "Script started at: $start_time"

python phys_warp.py >> pp.txt

end_time=$(date "+%Y-%m-%d %H:%M:%S")
echo "Script ended at: $end_time"
