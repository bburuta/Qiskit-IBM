#!/bin/bash

set -euo pipefail

partition=QPU
me=$(id -un)

declare -A cpu_alloc cpu_idle cpu_unusable cpu_total
declare -A gpu_alloc gpu_total node_state
declare -A my_cpus my_gpus known_node

tres_count() {
    local tres=$1 key=$2 item count=0
    local -a items

    IFS=',' read -ra items <<< "$tres"
    for item in "${items[@]}"; do
        if [[ ${item%%=*} == "$key" ]]; then
            printf '%s\n' "${item#*=}"
            return
        fi
        if [[ ${item%%=*} == "$key":* ]]; then
            count=$((count + ${item#*=}))
        fi
    done
    printf '%s\n' "$count"
}

# Read the CPU totals for every node in the QPU partition.
while IFS='|' read -r node cpus; do
    node=${node//[[:space:]]/}
    cpus=${cpus//[[:space:]]/}
    [[ -n $node ]] || continue

    IFS='/' read -r allocated idle unusable total <<< "$cpus"
    known_node[$node]=1
    cpu_alloc[$node]=$allocated
    cpu_idle[$node]=$idle
    cpu_unusable[$node]=$unusable
    cpu_total[$node]=$total
done < <(sinfo --noheader --Node --partition="$partition" --format='%N|%C')

# Read node state and GPU totals/allocations.
for node in "${!known_node[@]}"; do
    info=$(scontrol show node -o "$node")
    state=unknown
    cfg_tres=
    alloc_tres=

    for field in $info; do
        case $field in
            State=*) state=${field#State=} ;;
            CfgTRES=*) cfg_tres=${field#CfgTRES=} ;;
            AllocTRES=*) alloc_tres=${field#AllocTRES=} ;;
        esac
    done

    node_state[$node]=$state
    gpu_total[$node]=$(tres_count "$cfg_tres" gres/gpu)
    gpu_alloc[$node]=$(tres_count "$alloc_tres" gres/gpu)
done

# Count this user's running allocations. QPU jobs in this repository use one
# node; the division also handles ordinary equal allocations across many nodes.
while read -r user job_cpus job_nodes alloc_tres nodelist; do
    [[ $user == "$me" ]] || continue
    [[ $job_nodes =~ ^[1-9][0-9]*$ ]] || continue

    job_gpus=$(tres_count "$alloc_tres" gres/gpu)
    mapfile -t nodes < <(scontrol show hostnames "$nodelist")

    for i in "${!nodes[@]}"; do
        node=${nodes[$i]}
        [[ -n ${known_node[$node]:-} ]] || continue

        cpus_here=$((job_cpus / job_nodes))
        gpus_here=$((job_gpus / job_nodes))
        ((i < job_cpus % job_nodes)) && ((cpus_here += 1))
        ((i < job_gpus % job_nodes)) && ((gpus_here += 1))

        my_cpus[$node]=$((${my_cpus[$node]:-0} + cpus_here))
        my_gpus[$node]=$((${my_gpus[$node]:-0} + gpus_here))
    done
done < <(squeue --noheader --states=RUNNING \
    --Format='UserName:50,NumCPUs:10,NumNodes:10,tres-alloc:200,NodeList:100')

printf '%-10s %-14s | %5s %5s %6s %8s %5s | %5s %5s %6s %8s %5s\n' \
    NODE STATE FREE MINE OTHER UNUSABLE TOTAL FREE MINE OTHER UNUSABLE TOTAL
printf '%-10s %-14s | %5s %5s %6s %8s %5s | %5s %5s %6s %8s %5s\n' \
    '' '' CPU CPU CPU CPU CPU GPU GPU GPU GPU GPU

while IFS= read -r node; do
    [[ -n $node ]] || continue

    mine_cpu=${my_cpus[$node]:-0}
    mine_gpu=${my_gpus[$node]:-0}
    ((mine_cpu > cpu_alloc[$node])) && mine_cpu=${cpu_alloc[$node]}
    ((mine_gpu > gpu_alloc[$node])) && mine_gpu=${gpu_alloc[$node]}

    other_cpu=$((cpu_alloc[$node] - mine_cpu))
    other_gpu=$((gpu_alloc[$node] - mine_gpu))
    free_cpu=${cpu_idle[$node]}
    bad_cpu=${cpu_unusable[$node]}
    free_gpu=$((gpu_total[$node] - gpu_alloc[$node]))
    bad_gpu=0

    # Idle resources on an unavailable node are not usable by new jobs.
    if [[ ${node_state[$node]} =~ (DOWN|DRAIN|FAIL|FUTURE|INVAL|MAINT|NO_RESPOND|PLANNED|POWER|REBOOT|RESERVED|UNKNOWN) ]]; then
        bad_cpu=$((bad_cpu + free_cpu))
        bad_gpu=$free_gpu
        free_cpu=0
        free_gpu=0
    fi

    printf '%-10s %-14s | %5d %5d %6d %8d %5d | %5d %5d %6d %8d %5d\n' \
        "$node" "${node_state[$node]}" \
        "$free_cpu" "$mine_cpu" "$other_cpu" "$bad_cpu" "${cpu_total[$node]}" \
        "$free_gpu" "$mine_gpu" "$other_gpu" "$bad_gpu" "${gpu_total[$node]}"
done < <(printf '%s\n' "${!known_node[@]}" | sort -V)
