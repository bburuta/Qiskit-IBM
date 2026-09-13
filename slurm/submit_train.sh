#!/bin/bash

# Resubmit the two selected CPU-labelled convergence runs on GPUs.
# The batteries keep backend.simulator.device_name=CPU so they resume the original run ids.
sbatch slurm/slurm.sbatch_train_conv_remaining_cpu_092_rtx6000
sbatch slurm/slurm.sbatch_train_conv_remaining_cpu_065_rtx4500
