#!/bin/bash

# Resubmit only the three pending q16 noiseless REG GPU jobs on node14 CPUs.
# Cancel first:
# scancel 1582435 1582436 1582437
sbatch slurm/slurm.sbatch_train_conv_remaining_cpu_107_node14
sbatch slurm/slurm.sbatch_train_conv_remaining_cpu_108_node14
sbatch slurm/slurm.sbatch_train_conv_remaining_cpu_109_node14
