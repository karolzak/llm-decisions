#!/usr/bin/env bash
# Azure ML compute instance creation script (runs once, as root, at provisioning).
# Installs uv for `azureuser`; the project itself is set up by bootstrap-env.sh.
set -euo pipefail

sudo -u azureuser -i bash -c 'curl -LsSf https://astral.sh/uv/install.sh | sh'
apt-get update -y && apt-get install -y --no-install-recommends nvtop htop tmux || true

# Keep Hugging Face downloads on the large local disk instead of the small OS disk.
mkdir -p /mnt/hf-cache && chown azureuser:azureuser /mnt/hf-cache
echo 'export HF_HOME=/mnt/hf-cache' >> /home/azureuser/.bashrc
