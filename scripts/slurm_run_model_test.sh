#!/usr/bin/env bash
#SBATCH --job-name=adk-model-test
#SBATCH --output=slurm-%x-%j.out
#SBATCH --error=slurm-%x-%j.err
#SBATCH --time=2:00:00
#SBATCH --partition=gpu_h100
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=16

set -euo pipefail

PROJECT_DIR="${PROJECT_DIR:-$PWD}"
MODEL="upet"

export UV_CACHE_DIR="${UV_CACHE_DIR:-${SCRATCH:-$HOME}/uv-cache}"
export HF_HOME="${HF_HOME:-${SCRATCH:-$HOME}/huggingface}"
export TORCH_HOME="${TORCH_HOME:-${SCRATCH:-$HOME}/torch}"
export MPLCONFIGDIR="${MPLCONFIGDIR:-${SCRATCH:-$HOME}/matplotlib}"
export UV_PROJECT_ENVIRONMENT="${UV_PROJECT_ENVIRONMENT:-.venv-${MODEL}}"

cd "$PROJECT_DIR"

echo "Project directory: $PROJECT_DIR"
echo "Model: $MODEL"
echo "UV cache: $UV_CACHE_DIR"
echo "Virtual environment: $UV_PROJECT_ENVIRONMENT"

uv python install 3.11
uv sync --extra "$MODEL"
uv run --extra "$MODEL" python ~/AtomDefectKit/scripts/run_tests_${MODEL}_bcc_elements.py