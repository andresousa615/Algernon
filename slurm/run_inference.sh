#!/bin/bash
#SBATCH --job-name=algernon_inference
#SBATCH --account=<ACCOUNT>
#SBATCH --partition=<PARTITION>
#SBATCH --nodes=1
#SBATCH --gpus-per-node=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=128G
#SBATCH --time=00:30:00
#SBATCH --output=logs/inference_%j.log
#SBATCH --error=logs/inference_%j.err
# ==============================================================================
#  Inference + anonymisation with existing weights (phase 3 on its own).
#  The quickest way to try a region selection without retraining.
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
export OMP_NUM_THREADS=4
export TORCHINDUCTOR_CACHE_DIR="$PROJ_DIR/torch_compile_cache"
export TORCHINDUCTOR_FX_GRAPH_CACHE=1

# ------------------------------------------------------------------------------
# CONFIGURATION
# ------------------------------------------------------------------------------
RUN_DIR="${RUN_DIR:-$PROJ_DIR/results/<RUN_DIRECTORY>}"
WEIGHTS="${WEIGHTS:-$RUN_DIR/best_mednext_0.0005_mednext_128.pt}"
CONFIG_YAML="configs/train_128.yaml"
TEST_CSVS="data/test.csv"          # image_path required, mask_path optional
OUT_NAME="inference"               # sub-folder of RUN_DIR for the outputs

N_EXAMS=""                         # hard limit on exams per set (dry runs); empty = all
PRESERVE_MARGIN="0"                # extra voxels protected around preserved regions
ANONYMISE_REGIONS=""               # e.g. "nose,mouth"
PRESERVE_REGIONS=""                # e.g. "eyes"  (set only one of the two)
# ------------------------------------------------------------------------------

REGIONS_FLAG=""
if [ -n "$ANONYMISE_REGIONS" ] && [ -n "$PRESERVE_REGIONS" ]; then
    echo "[ERROR] Set ANONYMISE_REGIONS or PRESERVE_REGIONS, not both." >&2; exit 1
elif [ -n "$ANONYMISE_REGIONS" ]; then
    REGIONS_FLAG="--anonymise_regions $ANONYMISE_REGIONS"
elif [ -n "$PRESERVE_REGIONS" ]; then
    REGIONS_FLAG="--preserve_regions $PRESERVE_REGIONS"
fi

N_EXAMS_FLAG=""
if [ -n "$N_EXAMS" ] && [ "$N_EXAMS" -gt 0 ] 2>/dev/null; then
    N_EXAMS_FLAG="--n_exams $N_EXAMS"
fi

RDZV_PORT=$(( 29500 + (SLURM_JOB_ID % 1000) ))
start_time=$(date +%s)

srun torchrun \
    --nnodes=1 --nproc_per_node=1 \
    --rdzv_id=$SLURM_JOB_ID --rdzv_backend=c10d --rdzv_endpoint=$head_node:$RDZV_PORT \
    inference.py \
        --config "$CONFIG_YAML" \
        --weights "$WEIGHTS" \
        --test_csv $TEST_CSVS \
        --out_dir "$RUN_DIR/$OUT_NAME" \
        --metrics_csv "$RUN_DIR/${OUT_NAME}_metrics.csv" \
        --metrics_txt "$RUN_DIR/${OUT_NAME}_metrics.txt" \
        --preserve_margin "$PRESERVE_MARGIN" \
        $N_EXAMS_FLAG $REGIONS_FLAG

echo "Inference finished in $(( $(date +%s) - start_time )) s"
