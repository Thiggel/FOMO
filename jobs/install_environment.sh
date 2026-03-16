#!/bin/sh
#SBATCH --job-name=install-environment
#SBATCH --output=job_logs/install_environment_%A.out
#SBATCH --partition=a100
#SBATCH --gres=gpu:a100:2 -C a100_80
#SBATCH --time=24:00:00
#SBATCH --nodes=1

set -eu

SCRIPT_DIR="$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)"
"$SCRIPT_DIR/setup_uv_environment.sh"
