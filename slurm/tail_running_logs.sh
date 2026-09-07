#!/bin/bash

show_errors=false

case "${1:-}" in
    "")
        ;;
    errors|--errors|-e)
        show_errors=true
        ;;
    *)
        echo "Usage: $0 [errors|--errors|-e]"
        exit 1
        ;;
esac

if [ "$#" -gt 1 ]; then
    echo "Usage: $0 [errors|--errors|-e]"
    exit 1
fi

mapfile -t jobs < <(squeue -u "$USER" -t RUNNING -h -o "%i|%j")

if [ "${#jobs[@]}" -eq 0 ]; then
    echo "No running jobs."
    exit 0
fi

print_log_tail() {
    local label="$1"
    local filename="$2"

    printf "%s: %s\n" "$label" "$filename"
    if [ -f "$filename" ]; then
        tail -n 10 -- "$filename"
    else
        echo "Log file not found yet."
    fi
}

for job in "${jobs[@]}"; do
    IFS='|' read -r job_id job_name <<< "$job"
    job_details=$(scontrol show job -o "$job_id")
    work_dir=$(printf "%s\n" "$job_details" | grep -oE 'WorkDir=[^ ]+' | cut -d= -f2-)
    stdout_file=$(printf "%s\n" "$job_details" | grep -oE 'StdOut=[^ ]+' | cut -d= -f2-)
    stderr_file=$(printf "%s\n" "$job_details" | grep -oE 'StdErr=[^ ]+' | cut -d= -f2-)

    case "$stdout_file" in
        /*) ;;
        *) stdout_file="$work_dir/$stdout_file" ;;
    esac
    case "$stderr_file" in
        /*) ;;
        *) stderr_file="$work_dir/$stderr_file" ;;
    esac

    printf "\n====== Job %s: %s ======\n" "$job_id" "$job_name"
    print_log_tail "Output" "$stdout_file"

    if [ "$show_errors" = true ]; then
        print_log_tail "Errors" "$stderr_file"
    fi
done
