#!/bin/bash
# ==============================================================================
# NanoGPT Speedrun Job Runner
#
# This script is executed on a provisioned pod to run a single training job.
# It runs in the foreground and streams all output directly.
# ==============================================================================

set -e
set -o pipefail

# --- Configuration ---
RESULT_JSON_PATH="/root/job_result.json"
FAILURE_FLAG_PATH="/root/job_failed.flag"
NANOGPT_DIR="/root/modded-nanogpt"

# --- Pre-flight Checks ---
if [[ $# -ne 1 ]]; then
    echo "[ERROR] Usage: $0 <path_to_custom_train_script.py>" >&2
    exit 1
fi

CUSTOM_SCRIPT_PATH="$1"
if [[ ! -f "$CUSTOM_SCRIPT_PATH" ]];
then
    echo "[ERROR] Custom script not found at: $CUSTOM_SCRIPT_PATH" >&2
    touch "$FAILURE_FLAG_PATH"
    exit 1
fi

# --- Cleanup old job files ---
echo "[INFO] Cleaning up any previous job artifacts..."
rm -f "$RESULT_JSON_PATH" "$FAILURE_FLAG_PATH"

# --- Execute Training Job ---
echo "[INFO] Starting training job. CWD is $NANOGPT_DIR."
cd "$NANOGPT_DIR"

# Capture output to a temporary log file while also streaming it
TEMP_LOG_PATH="/tmp/torchrun_output.log"
torchrun --standalone --nproc_per_node=8 "$CUSTOM_SCRIPT_PATH" 2>&1 | tee "$TEMP_LOG_PATH"
TORCHRUN_EXIT_CODE=${PIPESTATUS[0]} # Get exit code of torchrun, not tee

# --- Process Results ---
if [[ $TORCHRUN_EXIT_CODE -eq 0 ]]; then
    echo "[SUCCESS] Training script completed successfully."

    # Parse the final line of the log to extract results
    FINAL_STATS_LINE=$(grep 'val_loss:' "$TEMP_LOG_PATH" | tail -n 1)
    echo "[INFO] Parsing final stats line: $FINAL_STATS_LINE"

    if [[ -n "$FINAL_STATS_LINE" ]]; then
        TRAINING_TIME_MS=$(echo "$FINAL_STATS_LINE" | grep -o 'train_time:[0-9]*ms' | cut -d':' -f2 | tr -d 'ms')
        VAL_LOSS=$(echo "$FINAL_STATS_LINE" | grep -o 'val_loss:[0-9.]*' | cut -d':' -f2)

        if [[ -n "$TRAINING_TIME_MS" && -n "$VAL_LOSS" ]]; then
            echo "[INFO] Successfully parsed results: Time=${TRAINING_TIME_MS}ms, Loss=${VAL_LOSS}"
            # Create the JSON result file using jq
            jq -n \
              --arg time_ms "$TRAINING_TIME_MS" \
              --arg val_loss "$VAL_LOSS" \
              '{training_time_ms: $time_ms, val_loss: $val_loss}' > "$RESULT_JSON_PATH"
            echo "[INFO] Result JSON created at $RESULT_JSON_PATH"
        else
            echo "[ERROR] Failed to parse training time or validation loss from the log."
            touch "$FAILURE_FLAG_PATH"
            exit 1
        fi
    else
        echo "[ERROR] Could not find the final statistics line in the training log."
        touch "$FAILURE_FLAG_PATH"
        exit 1
    fi
else
    echo "[ERROR] Training script failed with exit code $TORCHRUN_EXIT_CODE."
    touch "$FAILURE_FLAG_PATH"
    exit 1
fi