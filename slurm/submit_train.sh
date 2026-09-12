#!/bin/bash

# Resubmit only the jobs moved away from node03 CPU and the H100 GPU.
# Cancel first:
# scancel 1582445 1582444 1582438 1582442 1582443 1582434
sbatch slurm/slurm.sbatch_train_conv_remaining_cpu_045_node14
sbatch slurm/slurm.sbatch_train_conv_remaining_cpu_065_node15
sbatch slurm/slurm.sbatch_train_conv_remaining_cpu_066_node14
sbatch slurm/slurm.sbatch_train_conv_remaining_cpu_092_node15
sbatch slurm/slurm.sbatch_train_conv_remaining_cpu_093_node14
sbatch slurm/slurm.sbatch_train_conv_remaining_cpu_081_node15
