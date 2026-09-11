#!/bin/bash

# Submit only pending q16 noiseless REG replacements on the free node03 GPUs.
# These execute on GPU but keep aerCPU run IDs.
sbatch slurm/slurm.sbatch_train_conv_gpu_q16_reg_001_rtx4500
sbatch slurm/slurm.sbatch_train_conv_gpu_q16_reg_002_rtx4500
sbatch slurm/slurm.sbatch_train_conv_gpu_q16_reg_003_rtx6000
sbatch slurm/slurm.sbatch_train_conv_gpu_q16_reg_004_h100
sbatch slurm/slurm.sbatch_train_conv_gpu_q16_reg_005_rtx4500
sbatch slurm/slurm.sbatch_train_conv_gpu_q16_reg_006_rtx4500
sbatch slurm/slurm.sbatch_train_conv_gpu_q16_reg_007_rtx6000
