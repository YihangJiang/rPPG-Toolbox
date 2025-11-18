#!/bin/bash
#SBATCH --job-name=tt      # 作业名称
#SBATCH --output=UBFC_rPPG_UNSUPERVISED.log     # 标准输出文件（%j 会被作业ID替换）
#SBATCH --error=UBFC_rPPG_UNSUPERVISED.err      # 错误输出文件
#SBATCH --time=08:00:00                   # 运行时间上限 (HH:MM:SS)
#SBATCH --partition=common                # 分区名称
#SBATCH --ntasks=1                        # 任务数
#SBATCH --cpus-per-task=4                 # 每个任务使用的CPU核心数
#SBATCH --mem=256G                          # 分配的内存


python main.py --config_file ./configs/train_configs/UBFC-rPPG_UBFC-rPPG_UBFC-PHYS_EFFICIENTPHYS.yaml >> tscan_training_output.txt
