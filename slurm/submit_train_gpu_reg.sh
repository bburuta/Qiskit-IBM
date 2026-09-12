#!/bin/bash

# Submit only pending q16 noiseless REG replacements.
# GPU jobs execute on node03 GPUs but keep aerCPU run IDs.
# The former H100 job is replaced by its matching CPU node15 run.
# Broader helper: submit_train.sh now contains only the node03/H100 resubmission set.
sbatch slurm/slurm.sbatch_train_conv_gpu_q16_reg_001_rtx4500
sbatch slurm/slurm.sbatch_train_conv_gpu_q16_reg_002_rtx4500
sbatch slurm/slurm.sbatch_train_conv_gpu_q16_reg_003_rtx6000
sbatch slurm/slurm.sbatch_train_conv_remaining_cpu_081_node15
sbatch slurm/slurm.sbatch_train_conv_gpu_q16_reg_005_rtx4500
sbatch slurm/slurm.sbatch_train_conv_gpu_q16_reg_006_rtx4500
sbatch slurm/slurm.sbatch_train_conv_gpu_q16_reg_007_rtx6000
