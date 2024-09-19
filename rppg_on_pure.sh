#!/usr/bin/bash

#SBATCH --job-name=supervised_rppg
#SBATCH --output=res.txt
#SBATCH --error=res.err
#SBATCH --ntasks=1
#SBATCH --time=800:00


# SBATCH -p gpu-common --gres=gpu:1
#SBATCH --exclusive
start_time=$(date "+%Y-%m-%d %H:%M:%S")
echo "Script started at: $start_time"

python main.py --config_file ./configs/infer_configs/PURE_UBFC-rPPG_TSCAN_BASIC.yaml

end_time=$(date "+%Y-%m-%d %H:%M:%S")
echo "Script ended at: $end_time"
