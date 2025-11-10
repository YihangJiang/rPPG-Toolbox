#!/usr/bin/bash

#SBATCH --job-name=tscan_face_testing
#SBATCH --output=tscan_face_testing.txt
#SBATCH --error=tscan_face_testing.err
#SBATCH --time=800:00
#SBATCH --mem=100G

#SBATCH -p gpu-common --gres=gpu:1
#SBATCH --exclusive
start_time=$(date "+%Y-%m-%d %H:%M:%S")
echo "Script started at: $start_time"

python tscan_face_testing.py >> tscan_face_testing_output.txt

end_time=$(date "+%Y-%m-%d %H:%M:%S")
echo "Script ended at: $end_time"


