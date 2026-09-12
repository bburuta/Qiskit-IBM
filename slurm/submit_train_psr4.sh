#!/bin/bash

# Prioritized q4 noisy PSR convergence runs.
# CPU-only jobs on node14; no GPU resources are requested.
# Each run uses one seed and resumes its existing checkpoint if present.
sbatch slurm/slurm.sbatch_psr_noisy_q4_base_seed0_node14
sbatch slurm/slurm.sbatch_psr_noisy_q4_base_seed1_node14
sbatch slurm/slurm.sbatch_psr_noisy_q4_base_seed2_node14
sbatch slurm/slurm.sbatch_psr_noisy_q4_ang_seed0_node14
sbatch slurm/slurm.sbatch_psr_noisy_q4_ang_seed1_node14
sbatch slurm/slurm.sbatch_psr_noisy_q4_ang_seed2_node14
sbatch slurm/slurm.sbatch_psr_noisy_q4_amp_seed0_node14
sbatch slurm/slurm.sbatch_psr_noisy_q4_amp_seed1_node14
sbatch slurm/slurm.sbatch_psr_noisy_q4_amp_seed2_node14
