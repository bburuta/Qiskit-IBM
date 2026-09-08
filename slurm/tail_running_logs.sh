#!/bin/bash

show_errors=false
line_count=10
print_all=false

usage() {
    echo "Usage: $0 [errors|--errors|-e] [LINES|all]"
}

for arg in "$@"; do
    case "$arg" in
        errors|--errors|-e)
            show_errors=true
            ;;
        all|--all)
            print_all=true
            ;;
        ''|*[!0-9]*)
            usage
            exit 1
            ;;
        *)
            if [ "$arg" -eq 0 ]; then
                echo "LINES must be greater than 0."
                usage
                exit 1
            fi
            line_count="$arg"
            ;;
    esac
done

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
        if [ "$print_all" = true ]; then
            cat -- "$filename"
        else
            tail -n "$line_count" -- "$filename"
        fi
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
