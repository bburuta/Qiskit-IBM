# SLURM Helpers

Run these helpers from the repository root on a SLURM login node. They inspect
the current user's jobs and do not require the Python package. They assume the
standard `squeue`, `scontrol`, `sinfo`, `srun`, and `scancel` commands are
available; GPU sampling additionally requires `nvidia-smi` on the allocated
node.

## Queue and allocation summaries

Show the normal queue view followed by the node, allocated CPU count, and GPU
GRES for each running job:

```bash
./slurm/list_job_resources.sh
```

Show per-node CPU and GPU availability for the `QPU` partition:

```bash
./slurm/list_qpu_resources.sh
```

The second table separates free resources, resources allocated to the current
user (`MINE`), resources allocated to other users (`OTHER`), and resources on
unavailable nodes (`UNUSABLE`). User allocation counts include running jobs,
not pending jobs. The script is tailored to the repository's one-node jobs; for
multi-node jobs it distributes an equal allocation across the expanded node
list.

## Experimental setup hardware details

Show the configured devices of **every node in the QPU partition** from a
login node (the default mode):

```bash
./slurm/show_device_info.sh | tee qpu_devices.txt
```

This reports node architecture, CPU count and topology, RAM, GPU GRES/type
labels, features, state, and configured/allocated resources from Slurm. It
includes busy and unavailable nodes and does not require a running job.
Slurm usually does not store physical CPU model, GPU VRAM, driver or CUDA
versions. To collect those details, also probe each QPU node:

```bash
./slurm/show_device_info.sh --probe-qpu | tee qpu_hardware.txt
```

Probes request one CPU, 128 MiB RAM, and one GPU on GPU nodes, with a one-minute
time limit per node. Each request waits at most five seconds for resources;
unavailable probes retain the metadata report and cause a nonzero exit status.
GPU probes can only access devices exposed to their allocation. Use an existing
job ID to inspect a busy node when a fresh allocation cannot start.

To capture CPU model, architecture, sockets/cores/threads, caches, host RAM,
OS/kernel, NVIDIA GPU model and VRAM, driver, and CUDA toolkit compiler version
on the current host:

```bash
./slurm/show_device_info.sh --local | tee device_info.txt
```

Run this on the compute node used by the experiment, or add it to the batch
script before the training command so the report appears in the job log.
From a login node, inspect an existing running job (including an array task),
or all your running jobs:

```bash
./slurm/show_device_info.sh 1566450_3
./slurm/show_device_info.sh all | tee experiment_devices.txt
```

Each job node is inspected with a one-CPU overlapping `srun` step. The report
includes job allocation metadata, GPU visibility, and thread environment
variables. The reporting step's CPU affinity and CPUs-per-task describe that
step; use the job metadata for the experiment's allocation. Host RAM and CPU
topology describe physical capacity. GPU enumeration may include unassigned
devices; correlate it with the Slurm GPU IDs and visibility variables. The
CUDA version shown by `nvidia-smi` is the driver's maximum supported version;
`nvcc --version` reports the toolkit compiler available on `PATH`. Run inside
the experiment's container/environment to capture its OS and toolkit.

Missing GPU tools or unsupported compute-capability queries are reported
without preventing CPU information from being printed. A cluster may disable
overlapping steps. Reports describe hardware at capture time; they do not
recover the devices or software versions used by completed jobs.

## Live logs

Print the last ten lines of standard output for every running job:

```bash
./slurm/tail_running_logs.sh
```

Change the line count, include standard error, or print complete logs:

```bash
./slurm/tail_running_logs.sh 50
./slurm/tail_running_logs.sh --errors 50
./slurm/tail_running_logs.sh --errors all
```

Log paths are taken from each job's SLURM metadata, so this works with absolute
paths and paths relative to the job working directory.

## CPU, memory, and GPU sampling

Sample one running job for 60 seconds:

```bash
./slurm/monitor_job_usage.sh 1566420 60
```

Array task IDs are accepted, and `all` samples all of the current user's
running jobs concurrently:

```bash
./slurm/monitor_job_usage.sh 1566450_3 60
./slurm/monitor_job_usage.sh all 60
```

The monitor starts an overlapping `srun` step inside each allocation and
samples approximately once per second. During an interactive run, press `q` to
finish early and print the summary. The per-job table reports:

- CPU use as a percentage of the CPUs allocated to that job;
- resident memory in MiB;
- GPU utilization and peak VRAM use when a GPU is allocated;
- simple labels such as `GPU busy`, `CPU bottleneck`, or `CPU underused`.

The combined table groups monitored jobs by node. `CPU USED/ALLOC/NODE`
compares estimated cores in use with reserved cores and total node capacity;
`GPU EQUIV` expresses combined utilization as a number of fully occupied GPUs.

These labels are heuristics based on a short sampling window. Initialization,
I/O, queueing, or bursty work can look underutilized, and a cluster may disable
overlapping job steps. GPU inspection targets the first GPU assigned to each
job, matching the repository's usual one-GPU-per-job batch files.

## Cancelling jobs

```bash
./slurm/cancel_all_jobs.sh
```

This sends `SIGINT` to all jobs owned by the current user, including their job
steps. Check `squeue -u "$USER"` first; the command is intentionally broad.
