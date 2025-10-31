#!/usr/bin/bash

#SBATCH --job-name=base_IN_test
#SBATCH --output=base_IN_test.txt
#SBATCH --error=base_IN_test.err
#SBATCH --time=800:00
#SBATCH --mem=100G

#SBATCH -p gpu-common --gres=gpu:1
#SBATCH --exclusive
start_time=$(date "+%Y-%m-%d %H:%M:%S")
echo "Script started at: $start_time"

python base_IN_test.py >> base_IN_test_output.txt

end_time=$(date "+%Y-%m-%d %H:%M:%S")
echo "Script ended at: $end_time"

