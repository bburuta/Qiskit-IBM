#!/bin/bash

# Resubmit the three selected q16 noiseless REG runs on GPUs.
# The batteries keep run.device=CPU so they resume the original convergence run ids.
sbatch slurm/slurm.sbatch_train_conv_remaining_cpu_081_rtx6000
sbatch slurm/slurm.sbatch_train_conv_remaining_cpu_069_rtx6000
sbatch slurm/slurm.sbatch_train_conv_remaining_cpu_068_rtx4500
