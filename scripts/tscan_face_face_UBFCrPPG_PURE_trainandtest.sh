#!/usr/bin/bash

#SBATCH --job-name=tscan_face_face_UBFCrPPG_PURE_trainandtest
#SBATCH --output=tscan_face_face_UBFCrPPG_PURE_trainandtest.txt
#SBATCH --error=tscan_face_face_UBFCrPPG_PURE_trainandtest.err
#SBATCH --time=800:00
#SBATCH --mem=100G

#SBATCH -p gpu-common --gres=gpu:1
#SBATCH --exclusive

start_time=$(date "+%Y-%m-%d %H:%M:%S")
echo "Script started at: $start_time"

python tscan_face_face_UBFCrPPG_PURE_trainandtest.py >> tscan_face_face_UBFCrPPG_PURE_trainandtest_output.txt

end_time=$(date "+%Y-%m-%d %H:%M:%S")
echo "Script ended at: $end_time"

