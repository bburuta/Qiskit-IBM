#!/bin/bash

printf "%-12s %-30s %-12s %-8s %s\n" "JOB_ID" "NAME" "NODE" "CPUS" "GPU"

while IFS='|' read -r job name node cpus; do
    gpu=$(scontrol show job -dd -o "$job" |
        grep -oE 'GRES=[^ ]+' |
        tail -1)
    printf "%-12s %-30s %-12s %-8s %s\n" \
        "$job" "$name" "$node" "$cpus" "${gpu:-no GPU}"
done < <(squeue -u "$USER" -t RUNNING -h -o "%i|%j|%N|%C")
