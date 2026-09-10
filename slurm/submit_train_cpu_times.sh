#!/bin/bash

# Generated for unfinished CPU timing runs from train_times_cpu.yaml.
# Uses node03 CPUs with 4 CPUs per task, matching the GPU timing CPU allocation.
# q16 noisy CPU timing may OOM here; keep 16GB to match the GPU timing memory request.
sbatch slurm/slurm.sbatch_train_times_cpu_remaining_001_node03
sbatch slurm/slurm.sbatch_train_times_cpu_remaining_002_node03
sbatch slurm/slurm.sbatch_train_times_cpu_remaining_003_node03
sbatch slurm/slurm.sbatch_train_times_cpu_remaining_004_node03
