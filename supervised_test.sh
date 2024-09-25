#!/bin/bash
#SBATCH --job-name=UBFC_rPPG_UNSUPERVISED         # 作业名称
#SBATCH --output=UBFC_rPPG_UNSUPERVISED.log     # 标准输出文件（%j 会被作业ID替换）
#SBATCH --error=UBFC_rPPG_UNSUPERVISED.err      # 错误输出文件
#SBATCH --time=02:00:00                   # 运行时间上限 (HH:MM:SS)
#SBATCH --partition=common                # 分区名称
#SBATCH --ntasks=1                        # 任务数
#SBATCH --cpus-per-task=4                 # 每个任务使用的CPU核心数



python main.py --config_file ./configs/infer_configs/PURE_UBFC-rPPG_TSCAN_BASIC.yaml
