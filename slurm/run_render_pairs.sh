#!/bin/bash
#SBATCH --job-name=algernon_render
#SBATCH --account=<ACCOUNT>
#SBATCH --partition=<PARTITION>
#SBATCH --nodes=2
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=2
#SBATCH --cpus-per-task=32
#SBATCH --mem=128G
#SBATCH --time=01:00:00
#SBATCH --output=logs/render_pairs_%j.log
#SBATCH --error=logs/render_pairs_%j.err
# ------------------------------------------------------------------------------
#  Render the 2D (original / anonymised) pairs of every test exam — phase 4 of
#  the pipeline on its own, over an inference that already exists.
#  Feeds the defacing score and the re-identification evaluation.
# ------------------------------------------------------------------------------
set -euo pipefail

module purge
module load Python/3.10.8
module load Miniconda3/23.5.2-0
module load CUDA/12.4.0
module load Xvfb/21.1.8-GCCcore-12.3.0   # headless PyVista
module load Mesa/23.1.4-GCCcore-12.3.0
CONDA_ENV="${CONDA_ENV:-/path/to/conda/env_algernon}"   # <-- edit
eval "$(conda shell.bash hook)"
source activate "$CONDA_ENV"

PROJ_DIR="$(cd "$(dirname "$(realpath "$0")")/.." && pwd)"
cd "$PROJ_DIR"
export PYTHONPATH="$PROJ_DIR:${PYTHONPATH:-}"
mkdir -p logs

# -- Configuration (overridable through the environment) ----------------------
RUN="${RUN:-$PROJ_DIR/results/<RUN_DIRECTORY>}"
DIR_DEFACED="${DIR_DEFACED:-$RUN/inference}"   # folder with *_anon.nii.gz
DIR_PAIRS="${DIR_PAIRS:-$RUN/pairs_2d}"
TEST_CSVS="${TEST_CSVS:-$PROJ_DIR/data/test.csv}"
# ------------------------------------------------------------------------------

[ -e "$DIR_DEFACED" ] || { echo "ERROR: not found: $DIR_DEFACED" >&2; exit 1; }

nodes=( $(scontrol show hostnames "$SLURM_JOB_NODELIST") )
head_node=${nodes[0]}
RDZV_PORT=$(( 29500 + (SLURM_JOB_ID % 1000) ))

# Do not set GLOO_SOCKET_IFNAME: the '^' exclusion syntax is NCCL-only and
# gloo (used by this script) would try to resolve '^lo' as an interface.
export NCCL_SOCKET_IFNAME=^lo,docker0
export OMP_NUM_THREADS=4

srun xvfb-run -a -s "-screen 0 1600x1200x24 +extension GLX +render" \
    torchrun \
    --nnodes=2 --nproc_per_node=2 \
    --rdzv_id=$SLURM_JOB_ID --rdzv_backend=c10d --rdzv_endpoint=$head_node:$RDZV_PORT \
    evaluation/render_pairs_ddp.py \
        --dir_defaced "$DIR_DEFACED" \
        --test_csvs $TEST_CSVS \
        --dir_pairs "$DIR_PAIRS"

echo "Pairs rendered: $(ls "$DIR_PAIRS" | wc -l)"
echo "Next: RUN=$RUN sbatch slurm/run_reidentification.sh"
