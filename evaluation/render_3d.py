# -*- coding: utf-8 -*-
"""
Surface-style volume rendering of a NIfTI head with PyVista.

Provides the shared rendering parameters, the single-view renderer used by
render_pairs_ddp.py and a 4-view collage (frontal | left | right | inferior).
Requires an off-screen display (run under `xvfb-run` on headless nodes).
"""

import nibabel as nib
import numpy as np
import pyvista as pv
from PIL import Image
from skimage.filters import gaussian

# Rendering parameters (tuned once, shared by every render in the project)
AMBIENT = 1.3646
DIFFUSE = 1.6875
SPECULAR = 1.3854
SPECULAR_POWER = 4.6094
SAMPLE_DIST = 0.1070
SMOOTHING_SIGMA = 0.4

VIEW_SIZE = 800   # pixels per panel -> 1600x1600 collage


def opacity_parameters(data: np.ndarray, voxel_spacing, air_percentile=20, skin_percentile=35):
    """
    Derive the opacity transfer function from the intensity distribution.

    air_percentile:  discards the dark background before computing the skin
                     threshold (raise to 8-10 for noisy exams).
    skin_percentile: sets min_opacity — raise it to remove background static.
    """
    air_thr = np.percentile(data, air_percentile)
    tissue = data[data > air_thr]
    min_op = float(np.percentile(tissue, skin_percentile))
    max_op = float(np.percentile(tissue, 99))
    op_unit = float(np.linalg.norm(voxel_spacing) * 0.5)
    return min_op, max_op, op_unit


def make_grid(data: np.ndarray, voxel_spacing) -> pv.ImageData:
    grid = pv.ImageData()
    grid.dimensions = np.array(data.shape)
    grid.spacing = voxel_spacing
    grid.point_data["intensities"] = data.flatten(order="F")
    return grid


def compute_camera_positions(grid) -> dict:
    """
    Four camera positions derived from the grid bounds (NIfTI RAS axes):
      Y = anterior-posterior -> camera on +Y faces the face
      X = lateral            -> cameras on ±X face each ear
      Z = superior-inferior  -> camera on -Z looks at the chin
    Each value is [eye, focal, up] for plotter.camera_position.
    """
    b = grid.bounds
    cx, cy, cz = (b[0] + b[1]) / 2.0, (b[2] + b[3]) / 2.0, (b[4] + b[5]) / 2.0
    D = max(b[1] - b[0], b[3] - b[2], b[5] - b[4]) * 3.5
    focal = (cx, cy, cz)
    return {
        "Frontal": [(cx, cy + D, cz), focal, (0.0, 0.0, 1.0)],
        "Left": [(cx - D, cy, cz), focal, (0.0, 0.0, 1.0)],
        "Right": [(cx + D, cy, cz), focal, (0.0, 0.0, 1.0)],
        # looking upwards (+Z): anterior (+Y) ends up at the top of the image
        "Inferior": [(cx, cy, cz - D), focal, (0.0, 1.0, 0.0)],
    }


def render_view(grid, camera_pos, render_params, label=None, window_size=VIEW_SIZE) -> np.ndarray:
    """
    Render one view and return an (H, W, 3) uint8 array.
    render_params = (vol_min, vol_max, min_op, max_op, op_unit)
    """
    vol_min, vol_max, min_op, max_op, op_unit = render_params

    plotter = pv.Plotter(off_screen=True, window_size=[window_size, window_size])
    plotter.set_background("black")

    volume = plotter.add_volume(
        grid, scalars="intensities", cmap="gray", shade=True,
        ambient=AMBIENT, diffuse=DIFFUSE, specular=SPECULAR, specular_power=SPECULAR_POWER,
        show_scalar_bar=False,
    )
    if hasattr(volume.mapper, "SetAutoAdjustSampleDistances"):
        volume.mapper.SetAutoAdjustSampleDistances(0)
    volume.prop.interpolation_type = "linear"
    if hasattr(volume.prop, "SetScalarOpacityUnitDistance"):
        volume.prop.SetScalarOpacityUnitDistance(op_unit)
    if hasattr(volume.mapper, "SetSampleDistance"):
        volume.mapper.SetSampleDistance(SAMPLE_DIST)

    pwf = volume.prop.GetScalarOpacity()
    pwf.RemoveAllPoints()
    pwf.AddPoint(vol_min, 0.0)
    pwf.AddPoint(min_op, 0.0)
    pwf.AddPoint(max_op, 1.0)
    pwf.AddPoint(vol_max, 1.0)

    plotter.camera_position = camera_pos
    if label:
        plotter.add_text(label, position="upper_left", font_size=14, color="white")

    img = plotter.screenshot(return_img=True)
    plotter.close()

    if img.ndim == 3 and img.shape[2] == 4:   # PyVista may return RGBA
        img = img[:, :, :3]
    return img


def load_volume(nifti_path: str):
    """Load a NIfTI, compute opacity parameters and return (grid, render_params)."""
    nifti = nib.load(nifti_path)
    data = nifti.get_fdata().astype(np.float32)
    voxel_spacing = nifti.header.get_zooms()[:3]

    vol_min, vol_max = float(np.min(data)), float(np.max(data))
    min_op, max_op, op_unit = opacity_parameters(data, voxel_spacing)

    if SMOOTHING_SIGMA > 0.01:
        data = gaussian(data, sigma=SMOOTHING_SIGMA)

    return make_grid(data, voxel_spacing), (vol_min, vol_max, min_op, max_op, op_unit)


def render_compiled_image(nifti_path: str, output_path: str) -> None:
    """
    Render four views of the volume and save a 2x2 collage:
        [ Frontal | Left     ]
        [ Right   | Inferior ]
    """
    grid, render_params = load_volume(nifti_path)
    panels = [render_view(grid, cam, render_params, label=label)
              for label, cam in compute_camera_positions(grid).items()]
    top = np.concatenate([panels[0], panels[1]], axis=1)
    bot = np.concatenate([panels[2], panels[3]], axis=1)
    Image.fromarray(np.concatenate([top, bot], axis=0)).save(output_path)
