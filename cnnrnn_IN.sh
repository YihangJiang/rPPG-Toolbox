#!/usr/bin/bash

#SBATCH --job-name=cnnrnn_IN
#SBATCH --output=cnnrnn_IN.txt
#SBATCH --error=cnnrnn_IN.err
#SBATCH --time=800:00
#SBATCH --mem=100G

#SBATCH -p gpu-common --gres=gpu:1
#SBATCH --exclusive
start_time=$(date "+%Y-%m-%d %H:%M:%S")
echo "Script started at: $start_time"

python training_testing.py >> training_testing_output.txt

end_time=$(date "+%Y-%m-%d %H:%M:%S")
echo "Script ended at: $end_time"
