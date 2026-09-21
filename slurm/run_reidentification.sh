#!/bin/bash
#SBATCH --job-name=algernon_reid
#SBATCH --account=<ACCOUNT>
#SBATCH --partition=<PARTITION>
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=01:00:00
#SBATCH --output=logs/reid_%j.log
#SBATCH --error=logs/reid_%j.err
# ------------------------------------------------------------------------------
#  Re-identification risk over the phase-4 renders. CPU only: a few hundred
#  PNGs through a face detector and an embedding model.
#
#  deepface lives in its own environment (TensorFlow); see evaluation/README.md.
#  Do NOT load the system Python module here: it overrides the conda
#  site-packages and makes deepface unimportable on the compute node.
# ------------------------------------------------------------------------------
module purge
module load Miniconda3/23.5.2-0
REID_ENV="${REID_ENV:-/path/to/conda/env_reid}"   # <-- edit (deepface + TensorFlow)
eval "$(conda shell.bash hook)"
conda activate "$REID_ENV"

PROJ_DIR="$(cd "$(dirname "$(realpath "$0")")/.." && pwd)"
cd "$PROJ_DIR"
export PYTHONPATH="$PROJ_DIR:${PYTHONPATH:-}"
mkdir -p logs

# TensorFlow sizes its thread pools from the visible cores; cap them to the allocation.
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
export TF_NUM_INTRAOP_THREADS=$SLURM_CPUS_PER_TASK
export TF_NUM_INTEROP_THREADS=1
export TF_CPP_MIN_LOG_LEVEL=3
export CUDA_VISIBLE_DEVICES=""

# -- Configuration -------------------------------------------------------------
RUN="${RUN:-$PROJ_DIR/results/<RUN_DIRECTORY>}"
MODEL="ArcFace"          # ArcFace | Facenet512 | VGG-Face | Dlib
DETECTOR="retinaface"    # retinaface | mtcnn | opencv | ssd
ATTACK_PREFIX=""         # evaluate the attack only on exams starting with this prefix; empty = all
EXCLUDE_FILE=""          # optional "<id>: <reason>" file for data-defect exclusions
# ------------------------------------------------------------------------------

TAG="$(echo "$MODEL" | tr '[:upper:]' '[:lower:]')_${DETECTOR}${ATTACK_PREFIX:+_$ATTACK_PREFIX}"

python evaluation/reidentification.py \
    --pairs_dir "$RUN/pairs_2d" \
    --model     "$MODEL" \
    --detector  "$DETECTOR" \
    ${ATTACK_PREFIX:+--attack_prefix "$ATTACK_PREFIX"} \
    ${EXCLUDE_FILE:+--exclude_file "$EXCLUDE_FILE"} \
    --out       "$RUN/reidentification_${TAG}.txt" \
    --csv       "$RUN/reidentification_${TAG}.csv" \
    --cache     "$RUN/emb_$(echo "$MODEL" | tr '[:upper:]' '[:lower:]')_${DETECTOR}.npz"

echo "Done. Report: $RUN/reidentification_${TAG}.txt"
