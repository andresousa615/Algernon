#!/bin/bash
#SBATCH --job-name=algernon_pipeline
#SBATCH --account=<ACCOUNT>
#SBATCH --partition=<PARTITION>
#SBATCH --nodes=2
#SBATCH --gpus-per-node=2
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=64
#SBATCH --mem=200G
#SBATCH --time=06:00:00
#SBATCH --output=logs/pipeline_%j.log
#SBATCH --error=logs/pipeline_%j.err
# ==============================================================================
#  End-to-end Algernon pipeline (2 nodes x 2 GPUs):
#    1. DDP training           4. 2D renders of original/anonymised pairs (optional)
#    2. training curves        5. defacing score with a face detector (optional)
#    3. DDP inference + anonymisation
#
#  Edit the "ENVIRONMENT" and "CONFIGURATION" blocks for your cluster. Results
#  land in results/train_<model>_<lr>_<comment>_<job_id>/.
# ==============================================================================

# ------------------------------------------------------------------------------
# ENVIRONMENT
# ------------------------------------------------------------------------------
module purge
module load Python/3.10.8
module load Miniconda3/23.5.2-0
module load CUDA/12.4.0
module load Xvfb/21.1.8-GCCcore-12.3.0   # headless PyVista (phase 4)
module load Mesa/23.1.4-GCCcore-12.3.0

CONDA_ENV="${CONDA_ENV:-/path/to/conda/env_algernon}"   # <-- edit
eval "$(conda shell.bash hook)"
source activate "$CONDA_ENV"

PROJ_DIR="$(cd "$(dirname "$(realpath "$0")")/.." && pwd)"
cd "$PROJ_DIR"
export PYTHONPATH="$PROJ_DIR:${PYTHONPATH:-}"
mkdir -p logs

# DDP networking
nodes=( $( scontrol show hostnames $SLURM_JOB_NODELIST ) )
head_node=${nodes[0]}
export NCCL_SOCKET_IFNAME=^lo,docker0
export LOGLEVEL=INFO
RDZV_PORT=$(( 29500 + (SLURM_JOB_ID % 1000) ))   # unique port per job

# CPU thread limits (avoid thread explosion in the loader workers)
export OMP_NUM_THREADS=2
export MKL_NUM_THREADS=2
export OPENBLAS_NUM_THREADS=2
export VECLIB_MAXIMUM_THREADS=2
export NUMEXPR_NUM_THREADS=2
export ITK_GLOBAL_DEFAULT_NUMBER_OF_THREADS=2

# torch.compile cache persisted across jobs
export TORCHINDUCTOR_CACHE_DIR="$PROJ_DIR/torch_compile_cache"
export TORCHINDUCTOR_FX_GRAPH_CACHE=1
export TORCHINDUCTOR_AUTOGRAD_CACHE=1

# ------------------------------------------------------------------------------
# CONFIGURATION
# ------------------------------------------------------------------------------
EPOCHS=100
CONFIG_YAML="configs/train_128.yaml"
EARLY_STOP_FLAG=""            # "--earlystop" to enable
PROFILE_FLAG=""               # "--profile" to enable TrainingProfiler + MFU

TEST_CSVS="data/test.csv"     # space-separated list of test CSVs (image_path[, mask_path])

RUN_POST_TEST_PHASES="true"   # phases 4 (PyVista renders) and 5 (defacing score)
KEEP_INFERENCE_MASKS="true"   # keep the inference folder at the end
PROFILE_INFERENCE="false"     # per-phase timing of inference (perturbs throughput)
INFERENCE_TIMED_EXAMS=""      # time only the first N exams per set; empty = all
ANONYMISE_SET="all"           # "all" or "timed": which exams get anonymised outputs

# Selective anonymisation. Fill ONE of the two, or neither to anonymise everything.
# Valid regions: nose, eyes, ears, mouth
#   ANONYMISE_REGIONS="nose,mouth"   -> only nose and mouth
#   PRESERVE_REGIONS="eyes"          -> everything except the eyes
ANONYMISE_REGIONS=""
PRESERVE_REGIONS=""

# torch.profiler capture (Chrome trace, open at https://ui.perfetto.dev). Requires --profile.
TORCH_PROF_START_EPOCH=2
TORCH_PROF_EPOCHS=0           # 0 = off; >0 adds GPU syncs during the captured epochs

# ------------------------------------------------------------------------------
# Derived paths (mirror train.py's naming)
# ------------------------------------------------------------------------------
MODEL_NAME=$(python -c "import yaml; print(yaml.safe_load(open('$CONFIG_YAML'))['model'])")
LR=$(python -c "import yaml; print(yaml.safe_load(open('$CONFIG_YAML'))['lr'])")
COMMENT=$(python -c "import yaml; print(yaml.safe_load(open('$CONFIG_YAML'))['comment'])")
BASE_OUTPUT=$(python -c "import yaml; print(yaml.safe_load(open('$CONFIG_YAML'))['base_output'])")

MODEL_BASE="${MODEL_NAME}_${LR}_${COMMENT}"
RUN_DIR="${PROJ_DIR}/${BASE_OUTPUT}/train_${MODEL_BASE}_${SLURM_JOB_ID}"
WEIGHTS="${RUN_DIR}/best_${MODEL_BASE}.pt"
INFERENCE_DIR="${RUN_DIR}/inference"
DIR_PAIRS="${RUN_DIR}/pairs_2d"

echo "=========================================================="
echo " ALGERNON PIPELINE (job $SLURM_JOB_ID) — head node $head_node"
echo " Run directory: $RUN_DIR"
echo "=========================================================="

# Pre-flight: every node must see its GPUs
srun bash -c '
    GPU_COUNT=$(nvidia-smi --list-gpus 2>/dev/null | wc -l)
    if [ "$GPU_COUNT" -lt 1 ]; then echo "ERROR on $(hostname): no GPU found."; exit 1; fi
    echo "OK $(hostname): $GPU_COUNT GPUs."
' || { echo "ABORTED: missing GPUs on one or more nodes."; exit 1; }

pipeline_start=$(date +%s)

# ------------------------------------------------------------------------------
# PHASE 1: TRAINING
# ------------------------------------------------------------------------------
start_time=$(date +%s)
srun torchrun \
    --nnodes=2 --nproc_per_node=2 \
    --rdzv_id=$SLURM_JOB_ID --rdzv_backend=c10d --rdzv_endpoint=$head_node:$RDZV_PORT \
    train.py \
        --config "$CONFIG_YAML" \
        --epochs $EPOCHS \
        --profile-start-epoch $TORCH_PROF_START_EPOCH \
        --profile-epochs $TORCH_PROF_EPOCHS \
        $EARLY_STOP_FLAG $PROFILE_FLAG
echo "[PHASE 1 DONE] $(( $(date +%s) - start_time )) s"

# ------------------------------------------------------------------------------
# PHASE 2: TRAINING CURVES
# ------------------------------------------------------------------------------
python tools/plot_results.py --csv_file "$RUN_DIR/training_metrics.csv" --output_img "$RUN_DIR/training_plot.png"
echo "[PHASE 2 DONE]"

# ------------------------------------------------------------------------------
# PHASE 3: INFERENCE + ANONYMISATION
# ------------------------------------------------------------------------------
start_time=$(date +%s)
mkdir -p "$INFERENCE_DIR"

INF_PROFILE_FLAG=""
[ "$PROFILE_INFERENCE" = "true" ] && INF_PROFILE_FLAG="--profile"

REGIONS_FLAG=""
if [ -n "$ANONYMISE_REGIONS" ] && [ -n "$PRESERVE_REGIONS" ]; then
    echo "[ERROR] Set ANONYMISE_REGIONS or PRESERVE_REGIONS, not both." >&2; exit 1
elif [ -n "$ANONYMISE_REGIONS" ]; then
    REGIONS_FLAG="--anonymise_regions $ANONYMISE_REGIONS"
elif [ -n "$PRESERVE_REGIONS" ]; then
    REGIONS_FLAG="--preserve_regions $PRESERVE_REGIONS"
fi

TIMED_FLAG=""
if [ -n "$INFERENCE_TIMED_EXAMS" ] && [ "$INFERENCE_TIMED_EXAMS" -gt 0 ] 2>/dev/null; then
    TIMED_FLAG="--timed_exams $INFERENCE_TIMED_EXAMS"
fi

srun torchrun \
    --nnodes=2 --nproc_per_node=2 \
    --rdzv_id=$SLURM_JOB_ID --rdzv_backend=c10d --rdzv_endpoint=$head_node:$RDZV_PORT \
    inference.py \
        --config "$CONFIG_YAML" \
        --weights "$WEIGHTS" \
        --test_csv $TEST_CSVS \
        --out_dir "$INFERENCE_DIR" \
        --metrics_csv "$RUN_DIR/test_metrics.csv" \
        --metrics_txt "$RUN_DIR/test_metrics.txt" \
        --anonymise "$ANONYMISE_SET" \
        $INF_PROFILE_FLAG $TIMED_FLAG $REGIONS_FLAG
echo "[PHASE 3 DONE] $(( $(date +%s) - start_time )) s"

# ------------------------------------------------------------------------------
# PHASES 4 and 5 (optional)
# ------------------------------------------------------------------------------
if [ "$RUN_POST_TEST_PHASES" = "true" ]; then
    start_time=$(date +%s)
    srun xvfb-run -a -s "-screen 0 1600x1200x24 +extension GLX +render" \
        torchrun \
        --nnodes=2 --nproc_per_node=2 \
        --rdzv_id=$SLURM_JOB_ID --rdzv_backend=c10d --rdzv_endpoint=$head_node:$RDZV_PORT \
        evaluation/render_pairs_ddp.py \
            --dir_defaced "$INFERENCE_DIR" \
            --test_csvs $TEST_CSVS \
            --dir_pairs "$DIR_PAIRS"
    echo "[PHASE 4 DONE] $(( $(date +%s) - start_time )) s"

    start_time=$(date +%s)
    srun torchrun \
        --nnodes=2 --nproc_per_node=2 \
        --rdzv_id=$SLURM_JOB_ID --rdzv_backend=c10d --rdzv_endpoint=$head_node:$RDZV_PORT \
        evaluation/defacing_score_ddp.py \
            --dir_pairs "$DIR_PAIRS" \
            --output_report "$RUN_DIR/defacing_report.txt"
    echo "[PHASE 5 DONE] $(( $(date +%s) - start_time )) s"
else
    echo "[INFO] Phases 4 and 5 skipped (RUN_POST_TEST_PHASES=false)."
fi

if [ "$KEEP_INFERENCE_MASKS" = "false" ]; then
    rm -rf "$INFERENCE_DIR"
    echo "[CLEANUP] inference folder removed."
fi

pipeline_duration=$(( $(date +%s) - pipeline_start ))
echo "=========================================================="
echo " PIPELINE FINISHED in $((pipeline_duration / 3600))h $(((pipeline_duration % 3600) / 60))m"
echo " Results: $RUN_DIR"
echo "=========================================================="
