#!/bin/bash

# Resubmit the selected q4 noisy PSR runs on GPUs.
# The batteries keep backend.simulator.device_name=CPU so they resume the original aerCPU run ids.
sbatch slurm/slurm.sbatch_psr_noisy_q4_base_seed1_rtx6000
sbatch slurm/slurm.sbatch_psr_noisy_q4_ang_seed0_rtx6000
sbatch slurm/slurm.sbatch_psr_noisy_q4_amp_seed0_rtx4500
