#!/bin/bash
# Show QPU partition hardware, or capture details on an experiment's node.
set -uo pipefail
export LC_ALL=C

usage() {
    cat <<'EOF'
Usage: ./slurm/show_device_info.sh [QPU|--probe-qpu|--local|JOB_ID|all]

No argument (or QPU): report Slurm hardware metadata for every QPU node.
--probe-qpu: also run a short hardware probe on each available QPU node.
--local: report the current host, including inside a batch job.
JOB_ID: inspect every node of this user's matching running job.
all: inspect every node of all this user's running jobs.

Examples:
  ./slurm/show_device_info.sh | tee qpu_devices.txt
  ./slurm/show_device_info.sh --probe-qpu | tee qpu_hardware.txt
  ./slurm/show_device_info.sh --local | tee device_info.txt
  ./slurm/show_device_info.sh 1566450_3
  ./slurm/show_device_info.sh all | tee experiment_devices.txt

Job inspection starts a small overlapping srun step in an existing allocation.
QPU probes request one CPU, 128 MiB RAM and one GPU on GPU nodes, for up to
one minute per node. Busy/unavailable nodes are skipped after five seconds.
EOF
}

if [ "$#" -gt 1 ]; then
    usage >&2
    exit 1
fi
target="${1:-QPU}"
case "$target" in
    -h|--help) usage; exit 0 ;;
    --local) ;;
    QPU|--probe-qpu) ;;
    all) ;;
    *)
        if [[ ! "$target" =~ ^[0-9]+(_[0-9]+)?$ ]]; then
            usage >&2
            exit 1
        fi
        ;;
esac

if [ "$target" = "QPU" ] || [ "$target" = "--probe-qpu" ]; then
    required=(sinfo scontrol)
    if [ "$target" = "--probe-qpu" ]; then
        required+=(srun)
    fi
    for command in "${required[@]}"; do
        if ! command -v "$command" >/dev/null 2>&1; then
            printf 'Required command unavailable: %s\n' "$command" >&2
            exit 1
        fi
    done
    node_query=$(sinfo --noheader --Node --partition=QPU --format='%N') || exit 1
    nodes=$(printf '%s\n' "$node_query" | awk 'NF {print $1}' | sort -Vu)
    if [ -z "$nodes" ]; then
        printf 'No nodes found in partition QPU.\n' >&2
        exit 1
    fi
    printf 'Experimental setup: QPU partition devices\n'
    printf 'Captured (UTC): %s\n' "$(date -u '+%Y-%m-%dT%H:%M:%SZ')"
    printf 'Source: Slurm configured node resources (memory in MiB).\n'
    status=0
    while IFS= read -r node; do
        printf '\n========== QPU node %s ==========\n' "$node"
        info=$(scontrol show node -o "$node") || { status=1; continue; }
        # Slurm does not normally record the physical CPU model or GPU VRAM.
        # Features/GRES may contain administrator-supplied model labels.
        printf '%s\n' "$info" | awk '{
            for (i = 1; i <= NF; i++) {
                if ($i ~ /^(NodeName|Arch|CPUAlloc|CPUTot|CPUEfctv|CPULoad|Sockets|Boards|CoresPerSocket|ThreadsPerCore|RealMemory|AllocMem|FreeMem|Gres|GresUsed|AvailableFeatures|ActiveFeatures|State|OS|Version|CfgTRES|AllocTRES)=/) {
                    print $i
                }
            }
        }'
        if [ "$target" = "--probe-qpu" ]; then
            probe=(--partition=QPU --nodelist="$node" --nodes=1 --ntasks=1
                --cpus-per-task=1 --mem=128M --time=00:01:00 --immediate=5
                --job-name=device_info)
            if [[ "$info" =~ Gres=[^[:space:]]*gpu: ]]; then
                probe+=(--gres=gpu:1)
            fi
            printf '\nLive hardware probe:\n'
            if ! srun "${probe[@]}" bash -s -- --local < "${BASH_SOURCE[0]}"; then
                printf 'Live probe unavailable on %s; Slurm metadata is shown above.\n' "$node" >&2
                status=1
            fi
        fi
    done <<< "$nodes"
    printf '\nCPU model, caches, GPU VRAM, driver and CUDA require a live probe.\n'
    printf 'Use --probe-qpu, or inspect an existing JOB_ID on a busy node.\n'
    exit "$status"
fi

if [ "$target" != "--local" ]; then
    for command in squeue scontrol srun; do
        if ! command -v "$command" >/dev/null 2>&1; then
            printf 'Required command unavailable: %s\n' "$command" >&2
            exit 1
        fi
    done
    query=(--array --user="${USER:-$(id -un)}" --states=RUNNING --noheader --format='%i|%N')
    if [ "$target" != "all" ]; then
        query+=(--jobs="$target")
    fi
    jobs=$(squeue "${query[@]}") || exit 1
    if [ -z "$jobs" ]; then
        printf 'No matching running jobs.\n' >&2
        exit 1
    fi
    status=0
    while IFS='|' read -r job nodes; do
        hosts=$(scontrol show hostnames "$nodes") || { status=1; continue; }
        while IFS= read -r host; do
            printf '\n========== Job %s / node %s ==========\n' "$job" "$host"
            if ! srun --jobid="$job" --overlap --exact --nodes=1 --ntasks=1 \
                --cpus-per-task=1 --nodelist="$host" \
                bash -s -- --local < "${BASH_SOURCE[0]}"; then
                printf 'Could not inspect job %s on %s.\n' "$job" "$host" >&2
                status=1
            fi
        done <<< "$hosts"
    done <<< "$jobs"
    exit "$status"
fi

printf 'Experimental setup device report\n'
printf 'Captured (UTC): %s\n' "$(date -u '+%Y-%m-%dT%H:%M:%SZ')"
printf 'Host: %s\n' "$(hostname)"
printf 'Kernel: %s\n' "$(uname -srmo)"
if [ -r /etc/os-release ]; then
    # Read the distribution label without executing the file.
    awk -F= '/^PRETTY_NAME=/ {sub(/^PRETTY_NAME=/, ""); gsub(/^"|"$/, ""); print "OS: " $0}' /etc/os-release
fi

printf '\nCPU hardware (whole host)\n'
if command -v lscpu >/dev/null 2>&1; then
    lscpu | awk -F: '
        $1 ~ /^(Architecture|CPU op-mode\(s\)|Byte Order|CPU\(s\)|On-line CPU\(s\) list|Vendor ID|Model name|CPU family|Model|Stepping|Thread\(s\) per core|Core\(s\) per socket|Socket\(s\)|CPU MHz|CPU max MHz|CPU min MHz|L[123].*cache|NUMA node.*)$/ {print}
    '
else
    printf 'lscpu unavailable; CPU model from /proc/cpuinfo:\n'
    if [ -r /proc/cpuinfo ]; then
        awk -F: '/^(model name|Hardware|Processor)[[:space:]]*:/ {print; exit}' /proc/cpuinfo
    fi
fi
if command -v nproc >/dev/null 2>&1; then
    printf 'Logical CPUs available to this reporting process: %s\n' "$(nproc)"
fi
if [ -r /proc/self/status ]; then
    awk '/^Cpus_allowed_list:/ {print "Reporting process CPU affinity: " $2}' /proc/self/status
fi
if [ -r /proc/meminfo ]; then
    awk '/^MemTotal:/ {printf "Host RAM: %.2f GiB (%s KiB)\n", $2 / 1048576, $2}' /proc/meminfo
fi

printf '\nSlurm allocation and runtime visibility\n'
for variable in SLURM_JOB_ID SLURM_JOB_PARTITION SLURM_JOB_NODELIST \
    SLURM_JOB_NUM_NODES SLURM_JOB_CPUS_PER_NODE SLURM_CPUS_PER_TASK \
    SLURM_MEM_PER_NODE SLURM_MEM_PER_CPU SLURM_JOB_GPUS SLURM_STEP_GPUS \
    CUDA_VISIBLE_DEVICES NVIDIA_VISIBLE_DEVICES OMP_NUM_THREADS \
    MKL_NUM_THREADS OPENBLAS_NUM_THREADS; do
    printf '%s=%s\n' "$variable" "${!variable-<unset>}"
done
printf 'Slurm memory variables use MiB. Host capacity differs from job allocation.\n'
printf 'An inspection step uses one CPU; its affinity/CPUs-per-task describe that step.\n'
if [ -n "${SLURM_JOB_ID:-}" ] && command -v scontrol >/dev/null 2>&1; then
    printf '\nJob allocation metadata (including requested/allocated TRES):\n'
    scontrol show job -o "$SLURM_JOB_ID" || printf 'Job metadata unavailable.\n'
fi

printf '\nGPU hardware enumerated by nvidia-smi\n'
printf 'Enumeration may include unassigned GPUs; use visibility and Slurm GPU IDs above to identify assigned devices.\n'
if command -v nvidia-smi >/dev/null 2>&1; then
    if nvidia-smi --query-gpu=index,uuid,name,pci.bus_id,memory.total,driver_version \
        --format=csv; then
        printf '\nGPU compute capability (when supported by the installed driver):\n'
        nvidia-smi --query-gpu=index,compute_cap --format=csv 2>/dev/null || \
            printf 'Compute capability query unavailable.\n'
        printf '\nDriver status (CUDA Version here is the maximum supported by the driver):\n'
        nvidia-smi || printf 'Driver status unavailable.\n'
    else
        printf 'No accessible NVIDIA GPU, or the NVIDIA driver query failed.\n'
    fi
else
    printf 'nvidia-smi unavailable; NVIDIA model, VRAM and driver cannot be queried.\n'
    if command -v lspci >/dev/null 2>&1; then
        printf 'PCI display/3D devices (may include unassigned devices):\n'
        lspci | awk 'tolower($0) ~ /vga compatible controller|3d controller|display controller/ {print}'
    fi
fi

printf '\nCUDA toolkit compiler on PATH\n'
if command -v nvcc >/dev/null 2>&1; then
    nvcc --version || printf 'CUDA toolkit compiler query failed.\n'
else
    printf 'nvcc unavailable; the installed toolkit version cannot be established from PATH.\n'
fi
