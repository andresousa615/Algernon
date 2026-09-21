#!/bin/bash
#SBATCH --job-name=algernon_nsys
#SBATCH --account=<ACCOUNT>
#SBATCH --partition=<PARTITION>
#SBATCH --nodes=2
#SBATCH --gpus-per-node=2
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=64
#SBATCH --mem=300G
#SBATCH --time=00:40:00
#SBATCH --output=logs/nsys_%j.log
#SBATCH --error=logs/nsys_%j.err
# ==============================================================================
#  Nsight Systems profiling of a short training run (10 epochs), one nsys
#  process per rank through profiling/nsys_launcher.py:
#
#    srun -> torchrun -> [rank0: nsys_launcher.py -> nsys -t cuda,nvtx -> train.py]
#                        [rank1: nsys_launcher.py -> nsys -t nvtx      -> train.py]
#
#  Output: nsys_profiles/job_<id>/rank<r>.nsys-rep. Open rank0 and rank1 in the
#  Nsight Systems GUI and use View > Synchronize Timelines.
# ==============================================================================
module purge
module load Python/3.10.8
module load Miniconda3/23.5.2-0
module load CUDA/12.4.0

CONDA_ENV="${CONDA_ENV:-/path/to/conda/env_algernon}"   # <-- edit
eval "$(conda shell.bash hook)"
source activate "$CONDA_ENV"

PROJ_DIR="$(cd "$(dirname "$(realpath "$0")")/.." && pwd)"
cd "$PROJ_DIR"
export PYTHONPATH="$PROJ_DIR:${PYTHONPATH:-}"
mkdir -p logs

nodes=( $( scontrol show hostnames $SLURM_JOB_NODELIST ) )
head_node=${nodes[0]}
export NCCL_SOCKET_IFNAME=^lo,docker0
export LOGLEVEL=INFO
RDZV_PORT=$(( 29500 + (SLURM_JOB_ID % 1000) ))

export OMP_NUM_THREADS=2
export MKL_NUM_THREADS=2
export OPENBLAS_NUM_THREADS=2
export VECLIB_MAXIMUM_THREADS=2
export NUMEXPR_NUM_THREADS=2
export ITK_GLOBAL_DEFAULT_NUMBER_OF_THREADS=2
export TORCHINDUCTOR_CACHE_DIR="$PROJ_DIR/torch_compile_cache"
export TORCHINDUCTOR_FX_GRAPH_CACHE=1
export TORCHINDUCTOR_AUTOGRAD_CACHE=1

# The torch.compile cache must be warm: two ranks compiling into a cold cache
# on the same node corrupt the Triton autotune JSON files. Run the normal
# pipeline once (EPOCHS=1 is enough) before profiling.
if [ ! -d "$TORCHINDUCTOR_CACHE_DIR" ] || [ -z "$(ls -A "$TORCHINDUCTOR_CACHE_DIR" 2>/dev/null)" ]; then
    echo "ERROR: torch.compile cache is empty at $TORCHINDUCTOR_CACHE_DIR"
    echo "Run slurm/pipeline_train_eval.sh once (EPOCHS=1) to warm it up first."
    exit 1
fi

# ------------------------------------------------------------------------------
# CONFIGURATION
# ------------------------------------------------------------------------------
CONFIG_YAML="configs/train_128.yaml"
export NSYS_WARMUP_STEPS=250    # steps skipped before capture (~5 epochs at ~50 steps/rank)
export NSYS_PROFILE_STEPS=200   # steps captured (~4 epochs)

PROFDIR="${PROJ_DIR}/nsys_profiles/job_${SLURM_JOB_ID}"
mkdir -p "$PROFDIR"
export NSYS_PROFDIR="$PROFDIR"

echo "========================================================"
echo " NSIGHT SYSTEMS PROFILING — job $SLURM_JOB_ID — output $PROFDIR"
echo "========================================================"

srun bash -c '
    GPU_COUNT=$(nvidia-smi --list-gpus 2>/dev/null | wc -l)
    if [ "$GPU_COUNT" -lt 2 ]; then echo "ERROR on $(hostname): expected 2 GPUs, found $GPU_COUNT."; exit 1; fi
    echo "OK $(hostname): $GPU_COUNT GPUs."
' || { echo "ABORTED: missing GPUs."; exit 1; }

# Short, job-specific temp dir: avoids the two ranks clashing in /tmp while
# keeping paths under the 107-char Unix-socket limit of Python multiprocessing.
NSYS_TMPDIR="/tmp/nsys_${SLURM_JOB_ID}"
mkdir -p "$NSYS_TMPDIR"
export TMPDIR="$NSYS_TMPDIR"

srun torchrun \
    --nnodes=2 --nproc_per_node=2 \
    --rdzv_id=$SLURM_JOB_ID --rdzv_backend=c10d --rdzv_endpoint=$head_node:$RDZV_PORT \
    profiling/nsys_launcher.py \
    train.py \
        --config "$CONFIG_YAML" \
        --epochs 10 \
        --profile \
        --nsys-mode

# ------------------------------------------------------------------------------
# Post-processing: convert any .qdstrm nsys failed to convert automatically
# (QdstrmImporter; `nsys export --type=nsys-rep` does not exist in 2023.4.x).
# ------------------------------------------------------------------------------
NSYS_BIN=$(which nsys 2>/dev/null || echo "")
QDSTRM_IMPORTER="$(find "$(dirname "$NSYS_BIN")/.." -name "QdstrmImporter" 2>/dev/null | grep "nsight-systems" | head -1)"

for qdstrm in "$PROFDIR"/*.qdstrm; do
    [ -f "$qdstrm" ] || continue
    nsys_rep="${qdstrm%.qdstrm}.nsys-rep"
    if [ -f "$nsys_rep" ]; then
        rm -f "$qdstrm"
    elif [[ -x "$QDSTRM_IMPORTER" ]]; then
        echo "Converting $(basename "$qdstrm")"
        "$QDSTRM_IMPORTER" -i "$qdstrm" -o "$nsys_rep" 2>/dev/null || true
        if [[ -s "$nsys_rep" ]]; then
            rm -f "$qdstrm"
        else
            echo "WARNING: conversion failed — open $(basename "$qdstrm") directly in the GUI."
            rm -f "$nsys_rep" 2>/dev/null || true
        fi
    fi
done

echo "========================================================"
echo " PROFILING DONE — files:"
ls -lh "$PROFDIR"/rank*.nsys-rep 2>/dev/null
echo "========================================================"
