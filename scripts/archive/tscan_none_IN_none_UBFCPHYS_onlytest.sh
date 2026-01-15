#!/usr/bin/bash

#SBATCH --job-name=tscan_none_IN_none_UBFCPHYS_onlytest
#SBATCH --output=tscan_none_IN_none_UBFCPHYS_onlytest.txt
#SBATCH --error=tscan_none_IN_none_UBFCPHYS_onlytest.err
#SBATCH --time=800:00
#SBATCH --mem=100G

#SBATCH -p gpu-common --gres=gpu:1
#SBATCH --exclusive
start_time=$(date "+%Y-%m-%d %H:%M:%S")
echo "Script started at: $start_time"

python tscan_none_IN_none_UBFCPHYS_onlytest.py >> tscan_none_IN_none_UBFCPHYS_onlytest_output.txt

end_time=$(date "+%Y-%m-%d %H:%M:%S")
echo "Script ended at: $end_time"

