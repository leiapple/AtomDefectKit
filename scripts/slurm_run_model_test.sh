#!/usr/bin/env bash
#SBATCH --job-name=adk-model-test
#SBATCH --output=slurm-%x-%j.out
#SBATCH --error=slurm-%x-%j.err
#SBATCH --time=24:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G

set -euo pipefail

# Usage examples:
#   sbatch scripts/slurm_run_model_test.sh
#   sbatch --export=ALL,MODEL=upet,RUN_MODE=bcc_elements scripts/slurm_run_model_test.sh
#   sbatch --export=ALL,MODEL=mace,RUN_MODE=single,ELEMENT=V,INITIAL_A0=2.997 scripts/slurm_run_model_test.sh
#
# Optional site-specific setup:
#   module load gcc
#   module load cuda

PROJECT_DIR="${PROJECT_DIR:-$PWD}"
MODEL="${MODEL:-upet}"
RUN_MODE="${RUN_MODE:-bcc_elements}"
ELEMENT="${ELEMENT:-Nb}"
INITIAL_A0="${INITIAL_A0:-3.3}"
WORKING_DIR="${WORKING_DIR:-}"

export UV_CACHE_DIR="${UV_CACHE_DIR:-${SCRATCH:-$HOME}/uv-cache}"
export HF_HOME="${HF_HOME:-${SCRATCH:-$HOME}/huggingface}"
export TORCH_HOME="${TORCH_HOME:-${SCRATCH:-$HOME}/torch}"
export MPLCONFIGDIR="${MPLCONFIGDIR:-${SCRATCH:-$HOME}/matplotlib}"
export UV_PROJECT_ENVIRONMENT="${UV_PROJECT_ENVIRONMENT:-.venv-${MODEL}}"

cd "$PROJECT_DIR"

echo "Project directory: $PROJECT_DIR"
echo "Model: $MODEL"
echo "Run mode: $RUN_MODE"
echo "Element: $ELEMENT"
echo "Initial a0: $INITIAL_A0"
echo "UV cache: $UV_CACHE_DIR"
echo "Virtual environment: $UV_PROJECT_ENVIRONMENT"

uv python install 3.11
uv sync --extra "$MODEL"

case "$RUN_MODE" in
    bcc_elements)
        if [[ "$MODEL" != "upet" ]]; then
            echo "RUN_MODE=bcc_elements is currently implemented by run_tests_upet_bcc_elements.py."
            echo "Set MODEL=upet or use RUN_MODE=single."
            exit 2
        fi
        uv run --extra "$MODEL" python scripts/run_tests_upet_bcc_elements.py
        ;;
    single)
        script="scripts/run_tests_${MODEL}.py"
        if [[ ! -f "$script" ]]; then
            echo "No test script found for MODEL=$MODEL at $script"
            exit 2
        fi

        if [[ "$MODEL" == "upet" || "$MODEL" == "mace" ]]; then
            if [[ -n "$WORKING_DIR" ]]; then
                uv run --extra "$MODEL" python "$script" \
                    --element "$ELEMENT" \
                    --initial-a0 "$INITIAL_A0" \
                    --working-dir "$WORKING_DIR"
            else
                uv run --extra "$MODEL" python "$script" \
                    --element "$ELEMENT" \
                    --initial-a0 "$INITIAL_A0"
            fi
        else
            uv run --extra "$MODEL" python "$script"
        fi
        ;;
    *)
        echo "Unknown RUN_MODE=$RUN_MODE. Use bcc_elements or single."
        exit 2
        ;;
esac
