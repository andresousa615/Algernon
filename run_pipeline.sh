#!/bin/bash
# ==============================================================================
#  run_pipeline.sh — end-to-end Algernon pipeline on a single machine.
#
#  Runs every stage in sequence, assuming the data is already pre-processed and
#  the CSVs exist (see preprocessing/ and the README):
#
#    1. DDP training                  (train.py)
#    2. Training curves               (tools/plot_results.py)
#    3. Inference + anonymisation     (inference.py)
#    4. 2D renders of exam pairs      (evaluation/render_pairs_ddp.py)   optional
#    5. Defacing score                (evaluation/defacing_score_ddp.py) optional
#
#  No SLURM, no cluster modules: plain torchrun on the GPUs of this machine.
#  For multi-node runs on a cluster, use slurm/pipeline_train_eval.sh instead.
#
#  Everything lands in results/train_<model>_<lr>_<comment>_<run_id>/.
#
#  Usage:
#    ./run_pipeline.sh                            # all stages, all GPUs
#    ./run_pipeline.sh --gpus 2 --epochs 50
#    ./run_pipeline.sh --preserve-regions eyes    # keep the eyes intact
#    ./run_pipeline.sh --skip-eval                # stop after inference
#    ./run_pipeline.sh --help
# ==============================================================================
set -euo pipefail

# ------------------------------------------------------------------------------
# Defaults (override from the command line)
# ------------------------------------------------------------------------------
GPUS=""                       # empty = auto-detect
EPOCHS=100
CONFIG="configs/train_128.yaml"
TEST_CSVS=()                  # repeat --test-csv to add more; default set below
RUN_EVAL="true"
EARLY_STOP_FLAG=""
PROFILE_FLAG=""
ANONYMISE_REGIONS=""
PRESERVE_REGIONS=""
RUN_ID=""                     # empty = timestamp

usage() {
    # Print the header comment block: everything after the shebang up to the
    # first non-comment line, with the '#' and the '=' rules stripped.
    awk 'NR == 1 { next }
         /^#/    { sub(/^#+ ?/, ""); if ($0 !~ /^=+$/) print; next }
         { exit }' "$0"
    cat <<'HELP'

Options:
  --gpus N                 GPUs to use (default: all visible)
  --epochs N               Training epochs (default: 100)
  --config PATH            Config YAML (default: configs/train_128.yaml)
  --test-csv PATH          Test CSV for inference; repeat for several
                           (default: data/test.csv)
  --run-id NAME            Name for the output directory (default: timestamp)
  --preserve-regions LIST  Regions to keep intact, e.g. "eyes"
  --anonymise-regions LIST Regions to anonymise, e.g. "nose,mouth"
  --earlystop              Enable early stopping on validation DSC
  --profile                Enable the training profiler and MFU tracker
  --skip-eval              Skip stages 4 and 5 (renders + defacing score)
  -h, --help               Show this message
HELP
}

while [[ $# -gt 0 ]]; do
    case "$1" in
        --gpus)               GPUS="$2"; shift 2 ;;
        --epochs)             EPOCHS="$2"; shift 2 ;;
        --config)             CONFIG="$2"; shift 2 ;;
        --test-csv)           TEST_CSVS+=("$2"); shift 2 ;;
        --run-id)             RUN_ID="$2"; shift 2 ;;
        --preserve-regions)   PRESERVE_REGIONS="$2"; shift 2 ;;
        --anonymise-regions)  ANONYMISE_REGIONS="$2"; shift 2 ;;
        --earlystop)          EARLY_STOP_FLAG="--earlystop"; shift ;;
        --profile)            PROFILE_FLAG="--profile"; shift ;;
        --skip-eval)          RUN_EVAL="false"; shift ;;
        -h|--help)            usage; exit 0 ;;
        *) echo "Unknown option: $1  (try --help)" >&2; exit 1 ;;
    esac
done

if [ ${#TEST_CSVS[@]} -eq 0 ]; then
    TEST_CSVS=("data/test.csv")
fi

if [ -n "$ANONYMISE_REGIONS" ] && [ -n "$PRESERVE_REGIONS" ]; then
    echo "ERROR: use --anonymise-regions or --preserve-regions, not both." >&2
    exit 1
fi
case "$EPOCHS" in ''|*[!0-9]*) echo "ERROR: --epochs must be a number, got '$EPOCHS'." >&2; exit 1 ;; esac
if [ -n "$GPUS" ]; then
    case "$GPUS" in ''|*[!0-9]*) echo "ERROR: --gpus must be a number, got '$GPUS'." >&2; exit 1 ;; esac
fi

# ------------------------------------------------------------------------------
# Environment
# ------------------------------------------------------------------------------
INVOCATION_DIR="$PWD"                       # where the user ran the script from
PROJ_DIR="$(cd "$(dirname "$(realpath "$0")")" && pwd)"
cd "$PROJ_DIR"
export PYTHONPATH="$PROJ_DIR:${PYTHONPATH:-}"

# The script runs from the repo root, so a relative path the user typed would
# otherwise be resolved against the repo instead of their own directory. Try
# their directory first and fall back to the repo root (which is what the
# defaults such as data/test.csv expect).
resolve_path() {
    case "$1" in
        /*) printf '%s\n' "$1" ;;
        *)  if [ -e "$INVOCATION_DIR/$1" ]; then
                printf '%s\n' "$INVOCATION_DIR/$1"
            else
                printf '%s\n' "$1"
            fi ;;
    esac
}

CONFIG="$(resolve_path "$CONFIG")"
for i in "${!TEST_CSVS[@]}"; do
    TEST_CSVS[$i]="$(resolve_path "${TEST_CSVS[$i]}")"
done

# Keep the loader workers from spawning a thread per core each.
export OMP_NUM_THREADS=${OMP_NUM_THREADS:-2}
export MKL_NUM_THREADS=${MKL_NUM_THREADS:-2}
export OPENBLAS_NUM_THREADS=${OPENBLAS_NUM_THREADS:-2}
export ITK_GLOBAL_DEFAULT_NUMBER_OF_THREADS=${ITK_GLOBAL_DEFAULT_NUMBER_OF_THREADS:-2}

# Persist the torch.compile cache between runs.
export TORCHINDUCTOR_CACHE_DIR="${TORCHINDUCTOR_CACHE_DIR:-$PROJ_DIR/torch_compile_cache}"
export TORCHINDUCTOR_FX_GRAPH_CACHE=1
export TORCHINDUCTOR_AUTOGRAD_CACHE=1

if [ -z "$GPUS" ]; then
    if [ -n "${CUDA_VISIBLE_DEVICES:-}" ]; then
        # Honour the mask: count the entries the user made visible.
        GPUS=$(echo "$CUDA_VISIBLE_DEVICES" | tr ',' '\n' | grep -c '[^[:space:]]' || true)
    elif command -v nvidia-smi >/dev/null 2>&1; then
        GPUS=$(nvidia-smi --list-gpus 2>/dev/null | wc -l | tr -d ' ')
    else
        GPUS=0
    fi
    if [ "${GPUS:-0}" -lt 1 ] 2>/dev/null; then
        echo "ERROR: no CUDA GPU detected (nvidia-smi missing, reports none," >&2
        echo "       or CUDA_VISIBLE_DEVICES is empty)." >&2
        echo "       The pipeline needs CUDA (NCCL, AMP, torch.compile)." >&2
        echo "       Pass --gpus N to override this check." >&2
        exit 1
    fi
fi
if [ -z "$RUN_ID" ]; then
    RUN_ID="$(date +%Y%m%d_%H%M%S)"
fi

# One interpreter for the whole pipeline: a conda env provides `python`,
# a bare system may only have `python3`.
if command -v python >/dev/null 2>&1; then
    PY=python
elif command -v python3 >/dev/null 2>&1; then
    PY=python3
else
    echo "ERROR: no python interpreter found on PATH." >&2
    exit 1
fi
export ALGERNON_RUN_ID="$RUN_ID"    # train.py names its output directory with this

# ------------------------------------------------------------------------------
# Pre-flight
# ------------------------------------------------------------------------------
[ -f "$CONFIG" ] || { echo "ERROR: config not found: $CONFIG" >&2; exit 1; }
for csv in "${TEST_CSVS[@]}"; do
    [ -f "$csv" ] || { echo "ERROR: test CSV not found: $csv" >&2; exit 1; }
done

read_cfg() { "$PY" -c "import yaml,sys; print(yaml.safe_load(open('$CONFIG'))['$1'])"; }
TRAIN_CSV="$(read_cfg train_path)"
VAL_CSV="$(read_cfg val_path)"
for csv in "$TRAIN_CSV" "$VAL_CSV"; do
    [ -f "$csv" ] || { echo "ERROR: CSV from $CONFIG not found: $csv" >&2; exit 1; }
done

# Derived paths — must mirror how train.py names its output directory.
MODEL_BASE="$(read_cfg model)_$(read_cfg lr)_$(read_cfg comment)"
BASE_OUTPUT="$(read_cfg base_output)"
case "$BASE_OUTPUT" in
    /*) ;;                                      # already absolute
    *)  BASE_OUTPUT="$PROJ_DIR/$BASE_OUTPUT" ;;  # relative to the repo, as train.py reads it
esac
RUN_DIR="$BASE_OUTPUT/train_${MODEL_BASE}_${RUN_ID}"
WEIGHTS="$RUN_DIR/best_${MODEL_BASE}.pt"
INFERENCE_DIR="$RUN_DIR/inference"
DIR_PAIRS="$RUN_DIR/pairs_2d"

REGIONS_FLAG=""
if [ -n "$ANONYMISE_REGIONS" ]; then
    REGIONS_FLAG="--anonymise_regions $ANONYMISE_REGIONS"
elif [ -n "$PRESERVE_REGIONS" ]; then
    REGIONS_FLAG="--preserve_regions $PRESERVE_REGIONS"
fi

echo "=========================================================="
echo " ALGERNON PIPELINE — run $RUN_ID"
echo " GPUs      : $GPUS"
echo " Config    : $CONFIG  ($EPOCHS epochs)"
echo " Test CSVs : ${TEST_CSVS[*]}"
echo " Output    : $RUN_DIR"
if [ -n "$REGIONS_FLAG" ]; then echo " Regions   : $REGIONS_FLAG"; fi
echo "=========================================================="

pipeline_start=$(date +%s)
N_STAGES=5
if [ "$RUN_EVAL" != "true" ]; then N_STAGES=3; fi
stage() { echo; echo "---- [$1/$N_STAGES] $2 ----"; }

# ------------------------------------------------------------------------------
# 1. TRAINING
# ------------------------------------------------------------------------------
stage "1" "Training"
t0=$(date +%s)
torchrun --standalone --nproc_per_node="$GPUS" train.py \
    --config "$CONFIG" \
    --epochs "$EPOCHS" \
    $EARLY_STOP_FLAG $PROFILE_FLAG
echo "[1/$N_STAGES done] $(( $(date +%s) - t0 )) s"

[ -f "$WEIGHTS" ] || { echo "ERROR: training produced no weights at $WEIGHTS" >&2; exit 1; }

# ------------------------------------------------------------------------------
# 2. TRAINING CURVES
# ------------------------------------------------------------------------------
stage "2" "Training curves"
"$PY" tools/plot_results.py \
    --csv_file "$RUN_DIR/training_metrics.csv" \
    --output_img "$RUN_DIR/training_plot.png"

# ------------------------------------------------------------------------------
# 3. INFERENCE + ANONYMISATION
# ------------------------------------------------------------------------------
stage "3" "Inference and anonymisation"
t0=$(date +%s)
mkdir -p "$INFERENCE_DIR"
torchrun --standalone --nproc_per_node="$GPUS" inference.py \
    --config "$CONFIG" \
    --weights "$WEIGHTS" \
    --test_csv "${TEST_CSVS[@]}" \
    --out_dir "$INFERENCE_DIR" \
    --metrics_csv "$RUN_DIR/test_metrics.csv" \
    --metrics_txt "$RUN_DIR/test_metrics.txt" \
    $REGIONS_FLAG
echo "[3/$N_STAGES done] $(( $(date +%s) - t0 )) s"

# ------------------------------------------------------------------------------
# 4 and 5. PRIVACY EVALUATION (optional)
# ------------------------------------------------------------------------------
if [ "$RUN_EVAL" = "true" ]; then
    stage "4" "2D renders of original/anonymised pairs"
    t0=$(date +%s)
    # PyVista needs an OpenGL context; use xvfb-run when there is no display.
    RENDER_CMD=("$PY" evaluation/render_pairs_ddp.py
                --dir_defaced "$INFERENCE_DIR"
                --test_csvs "${TEST_CSVS[@]}"
                --dir_pairs "$DIR_PAIRS")
    if [ -z "${DISPLAY:-}" ] && command -v xvfb-run >/dev/null 2>&1; then
        xvfb-run -a -s "-screen 0 1600x1200x24 +extension GLX +render" "${RENDER_CMD[@]}"
    else
        "${RENDER_CMD[@]}"
    fi
    echo "[4/$N_STAGES done] $(( $(date +%s) - t0 )) s"

    stage "5" "Defacing score"
    t0=$(date +%s)
    "$PY" evaluation/defacing_score_ddp.py \
        --dir_pairs "$DIR_PAIRS" \
        --output_report "$RUN_DIR/defacing_report.txt"
    echo "[5/$N_STAGES done] $(( $(date +%s) - t0 )) s"
else
    echo; echo "[INFO] Stages 4 and 5 skipped (--skip-eval)."
fi

# ------------------------------------------------------------------------------
# Summary
# ------------------------------------------------------------------------------
duration=$(( $(date +%s) - pipeline_start ))
echo
echo "=========================================================="
echo " PIPELINE FINISHED in $((duration / 3600))h $(((duration % 3600) / 60))m $((duration % 60))s"
echo " Results: $RUN_DIR"
echo "----------------------------------------------------------"
echo "   training_metrics.csv / training_plot.png   training curves"
echo "   best_${MODEL_BASE}.pt                      best checkpoint"
echo "   test_metrics*.csv / *.txt                  Dice and timings"
echo "   inference/                                 masks + anonymised volumes"
if [ "$RUN_EVAL" = "true" ]; then
echo "   pairs_2d/                                  rendered exam pairs"
echo "   defacing_report.txt                        defacing score"
fi
echo "----------------------------------------------------------"
if [ "$RUN_EVAL" = "true" ]; then
echo " Re-identification risk is NOT part of this pipeline: it needs a"
echo " separate environment (deepface/TensorFlow). See evaluation/README.md:"
echo "   $PY evaluation/reidentification.py --pairs_dir $DIR_PAIRS \\"
echo "       --out $RUN_DIR/reidentification.txt"
fi
echo "=========================================================="
