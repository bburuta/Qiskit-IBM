#!/bin/bash

echo "submit_train_extra.sh is deprecated for the current thesis plan." >&2
echo "Use slurm/submit_train.sh for fake_real and q16 noisy SPSA runs." >&2
echo "Use slurm/submit_train_psr.sh only for the CPU noisy PSR viability test." >&2
exit 1
