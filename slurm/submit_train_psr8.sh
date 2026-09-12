#!/bin/bash

# Optional q8 noisy PSR convergence runs.
# CPU-only jobs on node15; no GPU resources are requested.
# Run after q4 PSR if there is enough remaining time.
sbatch slurm/slurm.sbatch_psr_noisy_q8_base_seed0_node15
sbatch slurm/slurm.sbatch_psr_noisy_q8_base_seed1_node15
sbatch slurm/slurm.sbatch_psr_noisy_q8_base_seed2_node15
sbatch slurm/slurm.sbatch_psr_noisy_q8_ang_seed0_node15
sbatch slurm/slurm.sbatch_psr_noisy_q8_ang_seed1_node15
sbatch slurm/slurm.sbatch_psr_noisy_q8_ang_seed2_node15
sbatch slurm/slurm.sbatch_psr_noisy_q8_amp_seed0_node15
sbatch slurm/slurm.sbatch_psr_noisy_q8_amp_seed1_node15
sbatch slurm/slurm.sbatch_psr_noisy_q8_amp_seed2_node15
