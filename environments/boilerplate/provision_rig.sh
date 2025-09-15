#!/bin/bash

# ==============================================================================
# Prime Intellect NanoGPT Speedrun Rig Provisioner
#
# This script automates the creation and configuration of a multi-GPU pod.
# - Accepts a --pod-name flag for unique pod identification.
# - Intelligently defaults the --socket type based on the selected --gpu-type.
# - Automatically terminates the pod if it enters an ERROR state.
# - Accepts an --offset flag to prevent race conditions when provisioning in parallel.
# ==============================================================================

set -e
set -o pipefail

SCRIPT_DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )

# --- Configuration ---
DISK_SIZE=1600 # in GB
IMAGE="ubuntu_22_cuda_12"
SETUP_SCRIPT_PATH="$SCRIPT_DIR/setup_pod.sh"
SETUP_TIMEOUT_MINS=30 # 30 minutes

# --- Argument Parsing ---
ON_DEMAND=false
GPU_TYPE_ARG=""
SOCKET_TYPE_ARG=""
POD_NAME=""
OFFSET=0

while [[ $# -gt 0 ]]; do
  key="$1"
  case $key in
    --on-demand)
      ON_DEMAND=true
      shift
      ;;
    --socket)
      SOCKET_TYPE_ARG="$2"
      shift 2
      ;;
    --pod-name)
      POD_NAME="$2"
      shift 2
      ;;
    --offset)
      OFFSET="$2"
      shift 2
      ;;
    *)
      GPU_TYPE_ARG="$1"
      shift
      ;;
  esac
done

# --- Helper Functions ---
log_info() { echo -e "\033[34m[INFO]\033[0m $1" >&2; }
log_success() { echo -e "\033[32m[SUCCESS]\033[0m $1" >&2; }
log_warn() { echo -e "\033[33m[WARN]\033[0m $1" >&2; }
log_error() { echo -e "\033[31m[ERROR]\033[0m $1" >&2; exit 1; }

# --- Validate required arguments ---
if [[ -z "$POD_NAME" ]]; then
    log_error "The --pod-name flag is required."
fi

GPU_TYPE=${GPU_TYPE_ARG:-"RTX6000Ada_48GB"}
GPU_COUNT=8

# --- Smart default for socket type based on GPU type ---
SOCKET_TYPE_DEFAULT="PCIe" # Safe fallback
case "$GPU_TYPE" in
  *H100*)
    SOCKET_TYPE_DEFAULT="SXM5"
    ;;
  *A100*)
    SOCKET_TYPE_DEFAULT="SXM4"
    ;;
  *RTX6000Ada*)
    SOCKET_TYPE_DEFAULT="PCIe"
    ;;
esac
SOCKET_TYPE=${SOCKET_TYPE_ARG:-$SOCKET_TYPE_DEFAULT}

# Memoize SSH details to avoid repeated CLI calls
declare -A SSH_DETAILS_CACHE

get_ssh_details() {
    local pod_id="$1"
    if [[ -n "${SSH_DETAILS_CACHE[$pod_id]}" ]]; then
        echo "${SSH_DETAILS_CACHE[$pod_id]}"
        return 0
    fi

    local ssh_key_path
    ssh_key_path=$(prime config view | grep "SSH Key Path" | awk -F'│' '{print $3}' | xargs)
    if [[ ! -f "$ssh_key_path" ]]; then
        log_error "SSH key not found at configured path: $ssh_key_path. Please run 'prime config set-ssh-key-path'."
    fi

    log_info "Fetching SSH details for pod $pod_id..."
    for attempt in {1..5}; do
        local status_output
        status_output=$(prime pods status "$pod_id" --output json 2>/dev/null || true)
        
        local connection_details
        connection_details=$(echo "$status_output" | sed '/^{/,$!d; /^}/q' | jq -r '.ssh // "null"')

        if [[ -n "$connection_details" && "$connection_details" != "null" ]]; then
            local user_host=$(echo "$connection_details" | awk '{print $1}')
            local port=$(echo "$connection_details" | awk '{print $3}')
            
            local details="$user_host $port $ssh_key_path"
            SSH_DETAILS_CACHE[$pod_id]="$details"
            echo "$details"
            return 0
        fi
        log_warn "Could not retrieve SSH details from API (attempt $attempt/5). Retrying in 5 seconds..."
        sleep 5
    done
    
    log_error "Failed to retrieve SSH details for pod $pod_id after multiple attempts."
    return 1
}

run_ssh_cmd() {
    local pod_id="$1"
    local remote_command="$2"
    local details
    details=$(get_ssh_details "$pod_id") || return 1
    read -r user_host port ssh_key_path <<< "$details"
    
    ssh -i "$ssh_key_path" -p "$port" -o "StrictHostKeyChecking=no" -o "UserKnownHostsFile=/dev/null" -o "ConnectTimeout=15" -o "BatchMode=yes" "$user_host" "$remote_command"
}

run_scp_upload_cmd() {
    local pod_id="$1"
    local local_path="$2"
    local remote_path="$3"
    local details
    details=$(get_ssh_details "$pod_id") || return 1
    read -r user_host port ssh_key_path <<< "$details"

    scp -i "$ssh_key_path" -P "$port" -o "StrictHostKeyChecking=no" -o "UserKnownHostsFile=/dev/null" -o "ConnectTimeout=15" "$local_path" "${user_host}:${remote_path}"
}

run_scp_download_cmd() {
    local pod_id="$1"
    local remote_path="$2"
    local local_path="$3"
    local details
    details=$(get_ssh_details "$pod_id") || return 1
    read -r user_host port ssh_key_path <<< "$details"

    scp -i "$ssh_key_path" -P "$port" -o "StrictHostKeyChecking=no" -o "UserKnownHostsFile=/dev/null" -o "ConnectTimeout=15" "${user_host}:${remote_path}" "$local_path"
}

wait_for_ssh() {
    local pod_id="$1"
    log_info "Waiting for SSH service on pod $pod_id to become available..."
    local details
    details=$(get_ssh_details "$pod_id") || log_error "Could not get SSH details to begin connection test."
    read -r user_host port ssh_key_path <<< "$details"

    local ssh_debug_log="/tmp/prime_ssh_debug_${pod_id}.log"
    
    for attempt in {1..30}; do # Try for up to 5 minutes
        log_info "Attempting SSH connection to $user_host:$port (attempt $attempt/30)..."
        
        if ssh -v -i "$ssh_key_path" -p "$port" -o "StrictHostKeyChecking=no" -o "UserKnownHostsFile=/dev/null" -o "ConnectTimeout=10" -o "BatchMode=yes" "$user_host" "echo 'SSH connection successful'" &> "$ssh_debug_log"; then
            log_success "SSH is ready."
            rm -f "$ssh_debug_log"
            return 0
        else
            log_warn "SSH connection test failed. See details below."
            echo "--- SSH Verbose Log (last 5 lines) ---" >&2
            tail -n 5 "$ssh_debug_log" >&2
            echo "------------------------------------" >&2
            log_info "Retrying in 10 seconds..."
            sleep 10
        fi
    done

    log_error "Failed to establish SSH connection to the pod after multiple attempts."
}

main() {
    # --- Phase 1: Pre-flight Checks ---
    log_info "Starting pre-flight checks..."
    command -v prime >/dev/null 2>&1 || log_error "prime-cli is not installed or not in your PATH."
    command -v jq >/dev/null 2>&1 || log_error "jq is not installed. Please install it."
    [[ -f "$SETUP_SCRIPT_PATH" ]] || log_error "Setup script '$SETUP_SCRIPT_PATH' not found."
    if ! prime config view > /dev/null 2>&1; then
        log_error "You are not logged into the Prime CLI. Please run 'prime login' first."
    fi
    log_success "Pre-flight checks passed."

    # --- Phase 2: Find or Create Pod ---
    log_info "Checking for existing pod named '$POD_NAME'..."
    POD_LIST_OUTPUT=$(prime pods list --output json || echo "{}")
    EXISTING_POD_JSON=$(echo "$POD_LIST_OUTPUT" | sed '/^{/,$!d; /^}/q' | jq -r --arg POD_NAME "$POD_NAME" '.pods[] | select(.name == $POD_NAME)')

    if [[ -n "$EXISTING_POD_JSON" ]]; then
        POD_ID=$(echo "$EXISTING_POD_JSON" | jq -r '.id')
        log_info "Found existing pod '$POD_NAME' with ID: $POD_ID. Checking its status..."
        wait_for_ssh "$POD_ID"
        if run_ssh_cmd "$POD_ID" "test -f /root/setup_complete.flag"; then
            log_success "Rig is already provisioned and configured! You are ready to go."
            echo -e "\nPod ID: \033[32m$POD_ID\033[0m"; echo -e "To connect: \033[32mprime pods ssh $POD_ID\033[0m\n"
            exit 0
        else
            log_warn "Pod exists but is not configured. Proceeding to configuration phase."
        fi
    else
        # --- Phase 3: Provision a New Pod ---
        log_info "No existing pod found. Provisioning a new one..."
        log_info "Searching for available ${GPU_COUNT}x ${GPU_TYPE} instances with ${SOCKET_TYPE} socket..."
        AVAILABILITY_JSON=$(prime availability list --gpu-type "$GPU_TYPE" --gpu-count "$GPU_COUNT" --output json | sed '/^{/,$!d; /^}/q')
        if [[ $(echo "$AVAILABILITY_JSON" | jq '.gpu_resources | length') -eq 0 ]]; then
            log_error "No available instances found for ${GPU_COUNT}x ${GPU_TYPE}."
        fi
        
        CREATE_ARGS=(--name "$POD_NAME" --disk-size "$DISK_SIZE" --image "$IMAGE" --yes)
        if [[ "$ON_DEMAND" == "true" ]]; then
            log_info "Attempting to provision an ON-DEMAND instance..."
            CHOSEN_CONFIG=$(echo "$AVAILABILITY_JSON" | jq -r '.gpu_resources | map(select(.is_spot != true and .socket == "'"$SOCKET_TYPE"'")) | sort_by(.price_value) | .['"$OFFSET"']')
            if [[ -z "$CHOSEN_CONFIG" || "$CHOSEN_CONFIG" == "null" ]]; then log_error "No on-demand ${SOCKET_TYPE} instances found at offset ${OFFSET}."; fi
            CHOSEN_ID=$(echo "$CHOSEN_CONFIG" | jq -r '.id')
            log_info "Found on-demand option at offset ${OFFSET}: Short ID '$CHOSEN_ID'."
            prime pods create --id "$CHOSEN_ID" "${CREATE_ARGS[@]}"
        else
            log_info "Searching for the best available SPOT instance..."
            SORTED_INSTANCES=$(echo "$AVAILABILITY_JSON" | jq -c 'def stock_priority: if . == "High" or . == "Available" then 3 elif . == "Medium" then 2 else 1 end; .gpu_resources | map(select(.is_spot == true and .socket == "'"$SOCKET_TYPE"'")) | sort_by([(.stock_status | stock_priority | -. ), .price_value]) | .[]' | tail -n +$((OFFSET + 1)))
            if [[ -z "$SORTED_INSTANCES" ]]; then log_error "No spot ${SOCKET_TYPE} instances found at offset ${OFFSET}. Try --on-demand or a different --socket type."; fi
            
            POD_CREATED=false
            while IFS= read -r instance_json; do
                CHOSEN_ID=$(echo "$instance_json" | jq -r '.id')
                log_info "Attempting to provision spot instance '$CHOSEN_ID' (from offset ${OFFSET})..."
                if CREATE_OUTPUT=$(prime pods create --id "$CHOSEN_ID" "${CREATE_ARGS[@]}" 2>&1); then
                    log_success "Successfully initiated pod creation for '$CHOSEN_ID'."
                    POD_CREATED=true
                    break
                else
                    log_warn "Provisioning failed for '$CHOSEN_ID'. Reason: $(echo "$CREATE_OUTPUT" | tail -n 1)"; sleep 3
                fi
            done <<< "$SORTED_INSTANCES"
            [[ "$POD_CREATED" == "true" ]] || log_error "All available spot instances from offset ${OFFSET} failed to provision. Try --on-demand."
        fi

        log_info "Waiting for pod to appear in the list..."; for i in {1..20}; do POD_ID=$(prime pods list --output json | sed '/^{/,$!d; /^}/q' | jq -r --arg POD_NAME "$POD_NAME" '.pods[] | select(.name == $POD_NAME) | .id'); if [[ -n "$POD_ID" ]]; then log_success "Pod created with ID: $POD_ID"; break; fi; sleep 10; done; if [[ -z "$POD_ID" ]]; then log_error "Failed to retrieve Pod ID after creation."; fi
        
        log_info "Waiting for pod to become ACTIVE..."
        POD_STATUS=""
        for i in {1..60}; do
            STATUS_JSON=$(prime pods status "$POD_ID" --output json 2>/dev/null || true)
            
            if [[ -n "$STATUS_JSON" ]]; then
                POD_STATUS=$(echo "$STATUS_JSON" | sed '/^{/,$!d; /^}/q' | jq -r '.status // "UNKNOWN"')
            else
                POD_STATUS="UNKNOWN"
            fi

            if [[ "$POD_STATUS" == "ACTIVE" ]]; then
                log_success "Pod is now ACTIVE."
                break
            elif [[ "$POD_STATUS" == "ERROR" ]]; then
                log_warn "Pod entered ERROR state. Terminating pod to prevent charges."
                prime pods terminate "$POD_ID" --yes
                log_error "Pod $POD_ID failed to provision and has been terminated."
            fi
            
            log_info "Current status: $POD_STATUS. Waiting... (attempt $i/60)"
            sleep 15
            POD_STATUS=""
        done

        if [[ "$POD_STATUS" != "ACTIVE" ]]; then
            log_warn "Pod did not become active in time. Terminating pod to prevent charges."
            prime pods terminate "$POD_ID" --yes
            log_error "Pod $POD_ID has been terminated due to timeout."
        fi
    fi
    
    # --- Phase 4: Configure the Pod ---
    wait_for_ssh "$POD_ID"
    log_info "Uploading setup script to the pod..."
    run_scp_upload_cmd "$POD_ID" "$SETUP_SCRIPT_PATH" "/tmp/setup_pod.sh"
    
    log_info "Executing setup script on the pod (in background). This will take 15-25 minutes..."
    run_ssh_cmd "$POD_ID" "chmod +x /tmp/setup_pod.sh && nohup /tmp/setup_pod.sh > /dev/null 2>&1 &"
    log_info "A detailed log is being saved to /root/setup.log on the pod."

    # --- Phase 5: Wait for Completion & Verify ---
    log_info "Waiting for setup to complete (timeout: ${SETUP_TIMEOUT_MINS} minutes)..."
    SECONDS=0
    TIMEOUT_SECONDS=$((SETUP_TIMEOUT_MINS * 60))
    while true; do
        if run_ssh_cmd "$POD_ID" "test -f /root/setup_failed.flag"; then
            log_warn "Setup script failed on the pod. Retrieving log..."
            
            LOCAL_LOG_FILE="failed_setup_${POD_ID}.log"
            if run_scp_download_cmd "$POD_ID" "/root/setup.log" "$LOCAL_LOG_FILE"; then
                log_warn "Log file saved to ./$LOCAL_LOG_FILE. Displaying contents:"
                echo -e "\n--- START OF FAILED POD LOG ---" >&2
                cat "$LOCAL_LOG_FILE" >&2
                echo -e "--- END OF FAILED POD LOG ---\n" >&2
            else
                log_warn "Could not retrieve setup.log from the pod."
            fi

            log_warn "Terminating failed pod to prevent charges."
            prime pods terminate "$POD_ID" --yes
            log_error "Pod $POD_ID has been terminated. Please review the log output above."
        fi

        if run_ssh_cmd "$POD_ID" "test -f /root/setup_complete.flag"; then
            log_success "Rig successfully provisioned and configured!"
            echo -e "\n=============================================="
            echo -e "  ✅ NanoGPT Speedrun Rig is Ready"
            echo -e "=============================================="
            echo -e "Pod ID:     \033[32m$POD_ID\033[0m"
            echo -e "GPU Type:   \033[32m$GPU_TYPE x$GPU_COUNT ($SOCKET_TYPE)\033[0m"
            echo -e "To connect: \033[32mprime pods ssh $POD_ID\033[0m"
            echo -e "==============================================\n"
            echo "FINAL_POD_ID:$POD_ID"
            exit 0
        fi

        if (( SECONDS > TIMEOUT_SECONDS )); then
            log_warn "Setup script timed out after ${SETUP_TIMEOUT_MINS} minutes. Terminating pod."
            prime pods terminate "$POD_ID" --yes
            log_error "Pod $POD_ID has been terminated. Check /root/setup.log on a future attempt for clues."
        fi

        sleep 30
        log_info "Setup in progress... ($(($SECONDS / 60))m elapsed). Checking again in 30s."
    done
}

# --- Script Entrypoint ---
main