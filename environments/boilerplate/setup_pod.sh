#!/bin/bash
# ==============================================================================
# NanoGPT Speedrun Rig - Pod Setup Script
#
# This script is uploaded to and executed on the newly provisioned pod.
# It handles all software installation and data download.
# Logs are saved to /root/setup.log on the pod.
# ==============================================================================

set -e
set -o pipefail

# On any error, create a failure flag and exit.
trap 'echo "[$(date)] --- Setup FAILED! ---" >&2; touch /root/setup_failed.flag; exit 1' ERR

# Redirect all output to a log file and the console
exec > >(tee /root/setup.log) 2>&1

echo "[$(date)] --- Starting NanoGPT Rig Setup ---"

echo "[$(date)] 1. Updating system packages and installing prerequisites..."
apt-get update
# Install base dependencies, including apt-utils to prevent debconf warnings.
DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends \
    build-essential \
    git \
    curl \
    python3-pip \
    software-properties-common \
    apt-utils \
    gnupg \
    ca-certificates \
    jq

# --- ROBUST PPA ADDITION ---
# Manually add the deadsnakes PPA to avoid dependency on the host's potentially
# broken python3-apt configuration, which causes the 'add-apt-repository' script to fail.
echo "[$(date)] Manually adding deadsnakes PPA for Python 3.12..."
# Get the PPA's GPG key
curl -sL "https://keyserver.ubuntu.com/pks/lookup?op=get&search=0xF23C5A6CF475977595C89F51BA6932366A755776" | gpg --dearmor > /etc/apt/trusted.gpg.d/deadsnakes.gpg
# Add the PPA to the sources list
echo "deb http://ppa.launchpadcontent.net/deadsnakes/ppa/ubuntu jammy main" > /etc/apt/sources.list.d/deadsnakes.list
echo "[$(date)] PPA added successfully. Updating package lists again..."
apt-get update

# Now install python 3.12 from the PPA
echo "[$(date)] Installing Python 3.12 and development headers..."
DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends \
    python3.12 \
    python3.12-dev \
    python3.12-venv

# Bootstrap pip using ensurepip and upgrade it.
echo "[$(date)] Bootstrapping pip for Python 3.12..."
python3.12 -m ensurepip --upgrade
python3.12 -m pip install --upgrade pip

# Upgrade setuptools to ensure distutils is available for Python 3.12
echo "[$(date)] Upgrading setuptools to be compatible with Python 3.12..."
python3.12 -m pip install --upgrade setuptools

echo "[$(date)] 2. Cloning modded-nanogpt repository..."
git clone https://github.com/KellerJordan/modded-nanogpt.git /root/modded-nanogpt
cd /root/modded-nanogpt

echo "[$(date)] 3. Installing Python dependencies..."
# Install requirements *except* for torch, then install torch separately
grep -v '^torch' requirements.txt | python3.12 -m pip install -r /dev/stdin
python3.12 -m pip install --pre torch --index-url https://download.pytorch.org/whl/nightly/cu126 --upgrade

echo "[$(date)] 4. Downloading the FineWeb dataset (first 800M tokens)..."
python3.12 data/cached_fineweb10B.py 8

echo "[$(date)] 5. Creating 'ready' flag."
touch /root/setup_complete.flag

echo "[$(date)] --- Setup Complete! You can now connect to the pod. ---"