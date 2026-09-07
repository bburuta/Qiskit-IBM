#!/bin/bash

if [ "$#" -lt 1 ]; then
    echo "Usage: $0 JOB_ID|all [INTERVAL_SECONDS]"
    exit 1
fi

target="$1"
if [ "$target" = "all" ]; then
    interval="${2:-10}"
else
    interval="${2:-2}"
fi

case "$interval" in
    ''|*[!0-9]*)
        echo "INTERVAL_SECONDS must be a positive integer."
        exit 1
        ;;
esac

if [ "$interval" -lt 1 ]; then
    echo "INTERVAL_SECONDS must be at least 1."
    exit 1
fi

if [ "$target" = "all" ]; then
    printf "Monitoring all your active jobs every %s seconds.\n" "$interval"
    printf "Press Ctrl+C to stop monitoring.\n"
    printf "GPU utilization appears in TRES only when GPU accounting is enabled on the cluster.\n"

    while true; do
        if [ -t 1 ]; then
            printf '\033[2J\033[H'
        fi

        printf "===== %s =====\n" "$(date '+%F %T')"
        squeue -u "$USER" \
            -o "%i | %j | %T | %N | CPUs=%C | MEM=%m | GRES=%b"

        mapfile -t active_jobs < <(squeue -u "$USER" -t RUNNING -h -o "%i")
        if [ "${#active_jobs[@]}" -eq 0 ]; then
            printf "\nNo running jobs.\n"
            sleep "$interval"
            continue
        fi

        job_steps=""
        for active_job in "${active_jobs[@]}"; do
            if [ -n "$job_steps" ]; then
                job_steps="${job_steps},"
            fi
            job_steps="${job_steps}${active_job}.batch"
        done

        printf "\nUsage by running job:\n"
        sstat -j "$job_steps" \
            --format=JobID%18,AveCPU,MaxRSS,AveRSS,MaxDiskRead,MaxDiskWrite,TRESUsageInAve%100

        printf "\nLow CPU/RSS/TRES values identify candidates for detailed inspection.\n"
        printf "Run '%s JOB_ID' in another terminal for live process and nvidia-smi data.\n" "$0"
        sleep "$interval"
    done
fi

job_id="$target"

job_info=$(squeue -h -j "$job_id" -o "%T|%N|%j")
if [ -z "$job_info" ]; then
    echo "Job $job_id was not found in the active queue."
    exit 1
fi

IFS='|' read -r job_state job_node job_name <<< "$job_info"
if [ "$job_state" != "RUNNING" ]; then
    echo "Job $job_id is $job_state, not RUNNING."
    exit 1
fi

printf "Job: %s  Name: %s  Node: %s\n\n" "$job_id" "$job_name" "$job_node"
scontrol show job -dd -o "$job_id"
printf "\nSlurm accounting snapshot:\n"
sstat -j "${job_id}.batch" \
    --format=JobID,AveCPU,MaxRSS,AveRSS,MaxDiskRead,MaxDiskWrite

printf "\nInterpretation:\n"
printf "  GPU low + qgan/Python CPU near its allocation: CPU bottleneck.\n"
printf "  GPU consistently high: GPU compute bottleneck.\n"
printf "  Memory or swap nearly full: memory pressure.\n"
printf "  High node load with low GPU use: CPU contention or thread overhead.\n"
printf "\nSampling every %s seconds. Press Ctrl+C to stop monitoring only.\n" "$interval"

srun --jobid="$job_id" --overlap --nodes=1 --ntasks=1 \
    bash -s -- "$job_id" "$interval" <<'MONITOR'
job_id="$1"
interval="$2"

while true; do
    printf "\n===== %s =====\n" "$(date '+%F %T')"

    printf "Node load:\n"
    uptime

    printf "\nNode memory:\n"
    free -h | awk 'NR == 1 || /^Mem:/ || /^Swap:/'

    printf "\nJob processes:\n"
    pids=$(scontrol listpids "$job_id" 2>/dev/null |
        awk 'NR > 1 && $1 ~ /^[0-9]+$/ {printf "%s%s", separator, $1; separator=","}')
    if [ -n "$pids" ]; then
        ps -p "$pids" -o pid,psr,nlwp,pcpu,pmem,rss,comm,args --sort=-pcpu
    else
        echo "No job processes reported by scontrol."
    fi

    printf "\nGPU usage:\n"
    if nvidia-smi -L >/dev/null 2>&1; then
        nvidia-smi \
            --query-gpu=index,name,utilization.gpu,utilization.memory,memory.used,memory.total,power.draw \
            --format=csv
    else
        echo "No GPU is visible in this job allocation."
    fi

    sleep "$interval"
done
MONITOR
