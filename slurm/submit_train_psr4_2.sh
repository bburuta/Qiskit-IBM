#!/bin/bash

# Move the three pending q4 amplitude noisy PSR jobs from node14 to node15.
# Cancel the old pending node14 jobs first if they are still queued:
# scancel 1582903 1582904 1582905
sbatch slurm/slurm.sbatch_psr_noisy_q4_amp_seed0_node15
sbatch slurm/slurm.sbatch_psr_noisy_q4_amp_seed1_node15
sbatch slurm/slurm.sbatch_psr_noisy_q4_amp_seed2_node15
