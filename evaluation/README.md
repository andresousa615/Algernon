# Evaluation tools

| Script | What it does | Extra dependencies |
|---|---|---|
| `render_pairs_ddp.py` | Builds (original, anonymised) pairs and renders frontal views + a 4-view collage with PyVista | `pyvista`, an OpenGL context (`xvfb-run -a` on headless nodes) |
| `render_3d.py` | Shared rendering parameters and helpers | `pyvista` |
| `defacing_score_ddp.py` | Face detector (dlib CNN) on the renders: fraction of exams with no detectable face | `face_recognition`, `dlib` (CUDA build) |
| `reidentification.py` | ArcFace embeddings: threshold calibration, 1:1 verification, 1:N identification | `deepface` |
| `inspect_detections.py` | Side-by-side images of what the detector fired on | `deepface`, `opencv-python` |

## deepface environment

`deepface` pulls TensorFlow, which does not coexist well with the PyTorch
training stack. Use a separate conda environment:

```bash
conda create -n algernon-reid python=3.10
conda activate algernon-reid
pip install deepface tf-keras opencv-python
```

The recognition/detection weights are downloaded on first use to
`~/.deepface/weights/`. On clusters whose compute nodes have no internet
access, run the script once on a login node (`--limit 2`) to pre-download them.

`slurm/run_reidentification.sh` deliberately does not `module load` the
system Python: it sets `PYTHONHOME`/`PYTHONPATH` and shadows the conda
site-packages, making `deepface` unimportable on the compute node.

## Exclusion files

`reidentification.py --exclude_file` and `tools/summarise_test_results.py
--exclude_file` take a text file with one `<subject or exam id>: <reason>`
per line. Use it for documented data defects (e.g. renders where the facial
surface was not reconstructed), never to drop bad results.
