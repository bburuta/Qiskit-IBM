#!/bin/bash

# No CPU timing jobs are submitted by this file right now.
# The reduced train_times_cpu.yaml plan omits q16 noisy CPU timing because those
# density-matrix jobs raised Out of Memory errors under the 16GB memory limit.
# Uses node03 CPUs with 4 CPUs per task, matching the GPU timing CPU allocation.
