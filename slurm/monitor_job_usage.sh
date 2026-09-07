#!/bin/bash

if [ "$#" -ne 2 ]; then
    echo "Usage: $0 JOB_ID|all SECONDS"
    echo "Examples:"
    echo "  $0 1566420 60"
    echo "  $0 all 60"
    exit 1
fi

target="$1"
duration="$2"

case "$duration" in
    ''|*[!0-9]*)
        echo "SECONDS must be a positive integer."
        exit 1
        ;;
esac

if [ "$duration" -lt 1 ]; then
    echo "SECONDS must be at least 1."
    exit 1
fi

job_ids=()
job_names=()
job_nodes=()
job_cpus=()
job_gres=()

if [ "$target" = "all" ]; then
    job_query=$(squeue -u "$USER" -t RUNNING -h -o "%i|%j|%N|%C|%b")
else
    job_query=$(squeue -u "$USER" -t RUNNING -h -j "$target" -o "%i|%j|%N|%C|%b")
fi

while IFS='|' read -r job_id job_name job_node cpus gres; do
    if [ -z "$job_id" ]; then
        continue
    fi
    job_ids+=("$job_id")
    job_names+=("$job_name")
    job_nodes+=("$job_node")
    job_cpus+=("$cpus")
    job_gres+=("$gres")
done <<< "$job_query"

if [ "${#job_ids[@]}" -eq 0 ]; then
    echo "No matching running jobs."
    exit 1
fi

monitor_dir=$(mktemp -d)
monitor_pids=()

stop_samplers() {
    local monitor_pid
    for monitor_pid in "${monitor_pids[@]}"; do
        kill "$monitor_pid" 2>/dev/null
    done
    for monitor_pid in "${monitor_pids[@]}"; do
        wait "$monitor_pid" 2>/dev/null
    done
    monitor_pids=()
}

cleanup() {
    stop_samplers
    rm -rf "$monitor_dir"
}

trap cleanup EXIT
trap 'exit 130' INT TERM

start_sampler() {
    local job_id="$1"
    local cpus="$2"
    local gres="$3"
    local sample_file="$monitor_dir/$job_id.samples"
    local error_file="$monitor_dir/$job_id.errors"

    srun --jobid="$job_id" --overlap --nodes=1 --ntasks=1 \
        bash -s -- "$job_id" "$cpus" "$gres" "$duration" \
        >"$sample_file" 2>"$error_file" <<'SAMPLER' &
job_id="$1"
allocated_cpus="$2"
gres="$3"
duration="$4"
end_time=$((SECONDS + duration + 5))

while [ "$SECONDS" -lt "$end_time" ]; do
    pids=$(scontrol listpids "$job_id" 2>/dev/null |
        awk -v monitor_step="${SLURM_STEP_ID:-}" '
        NR > 1 && $1 ~ /^[0-9]+$/ && $3 != monitor_step {
            printf "%s%s", separator, $1
            separator=","
        }')

    cpu_raw=0
    rss_kib=0
    if [ -n "$pids" ]; then
        read -r cpu_raw rss_kib <<< "$(
            ps -p "$pids" -o pcpu=,rss= 2>/dev/null |
                awk '{cpu += $1; rss += $2} END {printf "%.1f %.0f", cpu, rss}'
        )"
    fi

    cpu_usage=$(awk -v used="$cpu_raw" -v total="$allocated_cpus" \
        'BEGIN {if (total > 0) printf "%.1f", used / total; else print "0.0"}')

    gpu_id="-"
    gpu_usage="-"
    gpu_memory_used="-"
    gpu_memory_total="-"
    allocated_gpu="${SLURM_STEP_GPUS:-${SLURM_JOB_GPUS:-}}"
    allocated_gpu="${allocated_gpu%%,*}"
    if [[ "$gres" == *gpu* ]] && [ -n "$allocated_gpu" ]; then
        gpu_id="$allocated_gpu"
        gpu_data=$(
            nvidia-smi \
                --id="$allocated_gpu" \
                --query-gpu=utilization.gpu,memory.used,memory.total \
                --format=csv,noheader,nounits 2>/dev/null |
                awk -F',' 'NR == 1 {
                    gsub(/[[:space:]]/, "", $1)
                    gsub(/[[:space:]]/, "", $2)
                    gsub(/[[:space:]]/, "", $3)
                    print $1, $2, $3
                }'
        )
        if [ -n "$gpu_data" ]; then
            read -r gpu_usage gpu_memory_used gpu_memory_total <<< "$gpu_data"
        fi
    fi

    printf "%s|%s|%s|%s|%s|%s|%s\n" \
        "$cpu_usage" "$cpu_raw" "$rss_kib" \
        "$gpu_id" "$gpu_usage" "$gpu_memory_used" "$gpu_memory_total"
    sleep 1
done
SAMPLER

    monitor_pids+=("$!")
}

for index in "${!job_ids[@]}"; do
    start_sampler \
        "${job_ids[$index]}" \
        "${job_cpus[$index]}" \
        "${job_gres[$index]}"
done

summarize_samples() {
    local sample_file="$1"

    if [ ! -s "$sample_file" ]; then
        echo "waiting|-|-|-|-|-|-|-|-|-|0"
        return
    fi

    awk -F'|' '
        NF == 7 && $1 != "" && $3 != "" {
            samples++
            cpu_now = $1
            cpu_sum += $1
            if ($1 > cpu_max) cpu_max = $1
            rss_now = $3 / 1024
            if (rss_now > rss_max) rss_max = rss_now

            if ($4 != "" && $4 != "-" && $5 != "" && $5 != "-") {
                gpu_samples++
                gpu_id = $4
                gpu_now = $5
                gpu_sum += $5
                if ($5 > gpu_max) gpu_max = $5
                gpu_mem_total = $7
                if ($6 > gpu_mem_max) gpu_mem_max = $6
            }
        }
        END {
            if (samples == 0) {
                print "waiting|-|-|-|-|-|-|-|-|-|0"
                exit
            }

            printf "%.1f|%.1f|%.1f|%.0f|%.0f|", \
                cpu_now, cpu_sum / samples, cpu_max, rss_now, rss_max

            if (gpu_samples > 0) {
                printf "%s|%.1f|%.1f|%.1f|%.0f/%.0f|%d\n", \
                    gpu_id, gpu_now, gpu_sum / gpu_samples, gpu_max, \
                    gpu_mem_max, gpu_mem_total, samples
            } else {
                printf "-|-|-|-|-|%d\n", samples
            }
        }
    ' "$sample_file"
}

print_live_results() {
    local elapsed="$1"
    local remaining=$((duration - elapsed))
    local index summary
    local cpu_now cpu_average cpu_max rss_now rss_max
    local gpu_id gpu_now gpu_average gpu_max gpu_memory samples

    if [ "$remaining" -lt 0 ]; then
        remaining=0
    fi

    printf '\033[2J\033[H'
    printf "Usage inspection: %ss elapsed, %ss remaining — press q to finish early\n\n" \
        "$elapsed" "$remaining"
    printf "%-10s %-32s %-8s %-7s %-10s %-10s %-10s %-10s %-14s\n" \
        "JOB" "NAME" "CPUS" "GPU ID" "CPU NOW" "CPU AVG" "RAM MiB" "GPU NOW" "VRAM MiB"

    for index in "${!job_ids[@]}"; do
        summary=$(summarize_samples "$monitor_dir/${job_ids[$index]}.samples")
        IFS='|' read -r cpu_now cpu_average cpu_max rss_now rss_max \
            gpu_id gpu_now gpu_average gpu_max gpu_memory samples <<< "$summary"

        if [ "$cpu_now" != "waiting" ]; then
            cpu_now="${cpu_now}%"
            cpu_average="${cpu_average}%"
        fi
        if [ "$gpu_now" != "-" ]; then
            gpu_now="${gpu_now}%"
        fi

        printf "%-10s %-32s %-8s %-7s %-10s %-10s %-10s %-10s %-14s\n" \
            "${job_ids[$index]}" "${job_names[$index]}" "${job_cpus[$index]}" \
            "$gpu_id" "$cpu_now" "$cpu_average" "$rss_now" "$gpu_now" "$gpu_memory"
    done
}

interpret_result() {
    local cpu_average="$1"
    local gpu_average="$2"
    local gres="$3"
    local cpus="$4"

    if [ "$cpu_average" = "-" ]; then
        echo "no data"
        return
    fi

    if [[ "$gres" == *gpu* ]] && [ "$gpu_average" = "-" ]; then
        echo "GPU unavailable"
        return
    fi

    awk -v cpu="$cpu_average" -v gpu="$gpu_average" -v cpus="$cpus" '
        BEGIN {
            if (gpu != "-") {
                if (gpu >= 70) print "GPU busy"
                else if (gpu < 30 && cpu * cpus / 100 >= 0.8) print "CPU bottleneck"
                else if (gpu < 30) print "low usage/waiting"
                else print "mixed usage"
            } else {
                if (cpu >= 80) print "CPU busy"
                else if (cpu < 30) print "CPU underused"
                else print "CPU moderate"
            }
        }
    '
}

print_final_results() {
    local reason="$1"
    local index summary error_file
    local cpu_now cpu_average cpu_max rss_now rss_max
    local gpu_id gpu_now gpu_average gpu_max gpu_memory samples result

    printf '\033[2J\033[H'
    printf "Usage inspection finished: %s\n\n" "$reason"
    printf "%-10s %-32s %-8s %-7s %-10s %-10s %-12s %-10s %-10s %-16s %-8s %-18s\n" \
        "JOB" "NAME" "CPUS" "GPU ID" "CPU AVG" "CPU MAX" "RAM MAX MiB" \
        "GPU AVG" "GPU MAX" "VRAM MAX/TOTAL" "SAMPLES" "RESULT"

    for index in "${!job_ids[@]}"; do
        summary=$(summarize_samples "$monitor_dir/${job_ids[$index]}.samples")
        IFS='|' read -r cpu_now cpu_average cpu_max rss_now rss_max \
            gpu_id gpu_now gpu_average gpu_max gpu_memory samples <<< "$summary"

        if [ "$cpu_now" = "waiting" ]; then
            cpu_average="-"
            cpu_max="-"
            rss_max="-"
            result="no data"
        else
            result=$(interpret_result \
                "$cpu_average" "$gpu_average" "${job_gres[$index]}" "${job_cpus[$index]}")
            cpu_average="${cpu_average}%"
            cpu_max="${cpu_max}%"
        fi
        if [ "$gpu_average" != "-" ]; then
            gpu_average="${gpu_average}%"
            gpu_max="${gpu_max}%"
        fi

        printf "%-10s %-32s %-8s %-7s %-10s %-10s %-12s %-10s %-10s %-16s %-8s %-18s\n" \
            "${job_ids[$index]}" "${job_names[$index]}" "${job_cpus[$index]}" \
            "$gpu_id" "$cpu_average" "$cpu_max" "$rss_max" \
            "$gpu_average" "$gpu_max" "$gpu_memory" "$samples" "$result"

        error_file="$monitor_dir/${job_ids[$index]}.errors"
        if [ ! -s "$monitor_dir/${job_ids[$index]}.samples" ] && [ -s "$error_file" ]; then
            printf "  Monitor error: "
            tail -n 1 "$error_file"
        fi
    done

    printf "\nInterpretation:\n"
    printf "  GPU busy: GPU AVG >= 70%%. CPU bottleneck: GPU AVG < 30%% with about one CPU core busy.\n"
    printf "  CPU-only jobs near 100%% use most allocated CPUs; below 30%% means underused.\n"
    printf "  RAM/VRAM near the allocation limit: memory pressure.\n"
    printf "  Results are simple heuristics; short or bursty workloads may appear as mixed/low usage.\n"
}

start_time=$SECONDS
finish_reason="requested time completed"

while true; do
    elapsed=$((SECONDS - start_time))
    if [ "$elapsed" -ge "$duration" ]; then
        break
    fi

    print_live_results "$elapsed"

    key=""
    if [ -t 0 ]; then
        read -r -s -n 1 -t 1 key
    else
        sleep 1
    fi

    case "$key" in
        q|Q)
            finish_reason="stopped early with q"
            break
            ;;
    esac
done

stop_samplers
print_final_results "$finish_reason"
