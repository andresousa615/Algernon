# -*- coding: utf-8 -*-
# Portions derived from mede (https://pypi.org/project/mede/), Copyright 2025
# Lukas Heine & Moritz Rempe, Apache License 2.0 with Commons Clause — see
# LICENSE-mede and NOTICE.
"""
Datasets and loaders for Algernon.

Training samples are NIfTI volumes already resampled to the network input size
(see preprocessing/resample_to_128.py). The dataset performs I/O, label
remapping, TorchIO augmentation and min-max normalisation on CPU workers; the
work-stealing MinatoSegLoader (data/minato_loader.py) delivers the batches.

Label convention
----------------
Ground-truth masks use labels {0, 2, 3, 4, 5}; they are remapped to a
contiguous range {0, 1, 2, 3, 4} = {background, ears, mouth, nose, eyes}, which
is what the loss and the AnonymizationOfficer expect.

Augmentation (training only)
----------------------------
  spatial   : RandomAffine (75 %) or RandomElasticDeformation (25 %) — exactly one.
              When the loader sets MINATO_FORCE_TRANSFORM, the choice is taken
              from the loader's interleaved schedule instead of drawn at random.
  intensity : RandomBiasField, RandomNoise, RandomGamma, RandomSpike, each with p = 0.25.
  normalise : min-max to [0, 1].
"""
import logging
import os
import random
from collections import defaultdict
from glob import glob

import nibabel as nib
import numpy as np
import pydicom
import torch
import torchio as tio
from torch.utils.data import DataLoader, Dataset
from torch.utils.data.distributed import DistributedSampler

from data.minato_loader import MinatoSegLoader
from profiling.worker_nvtx import pop as nvtx_pop
from profiling.worker_nvtx import push as nvtx_push

# Network input size (D, H, W). Volumes fed to the model must have this shape.
DEFAULT_INPUT_SIZE = (128, 128, 128)

# Original mask label -> contiguous class index.
LABEL_REMAP = {0: 0, 2: 1, 3: 2, 4: 3, 5: 4}


def remap_labels(mask: torch.Tensor) -> torch.Tensor:
    """Remap ground-truth labels {0,2,3,4,5} to {0,1,2,3,4}."""
    out = torch.zeros_like(mask)
    for original, new in LABEL_REMAP.items():
        out[mask == original] = new
    return out


# ---------------------------------------------------------------------------
# Orientation / DICOM helpers (shared with inference on raw exams)
# ---------------------------------------------------------------------------

def resample(nifti_img: nib.Nifti1Image) -> nib.Nifti1Image:
    """Reorient a NIfTI image to canonical RAS."""
    orig_orientation = nib.orientations.io_orientation(nifti_img.affine)
    target_orientation = nib.orientations.axcodes2ornt(("R", "A", "S"))
    transform = nib.orientations.ornt_transform(orig_orientation, target_orientation)
    return nifti_img.as_reoriented(transform)


def prepare_slices(fp: str) -> np.ndarray:
    """Read a DICOM file and return its pixel array."""
    return pydicom.dcmread(fp).pixel_array


def order_slices(fp: str) -> int:
    """Read a DICOM header and return the instance number (for slice ordering)."""
    return pydicom.dcmread(fp, stop_before_pixels=True).InstanceNumber


def create_affine(sorted_dicoms: list) -> np.matrix:
    """Build a NIfTI affine from the metadata of an ordered DICOM series."""
    dicom_first = pydicom.dcmread(sorted_dicoms[0])
    dicom_last = pydicom.dcmread(sorted_dicoms[-1])

    if hasattr(dicom_first, "ImageOrientationPatient"):
        image_orient1 = np.array(dicom_first.ImageOrientationPatient)[0:3]
        image_orient2 = np.array(dicom_first.ImageOrientationPatient)[3:6]
        image_pos = np.array(dicom_first.ImagePositionPatient)
        last_image_pos = np.array(dicom_last.ImagePositionPatient)
    else:
        logging.warning("ImageOrientationPatient not found in DICOM metadata. Using default orientation.")
        image_orient1 = np.array([1, 0, 0])
        image_orient2 = np.array([0, 1, 0])
        image_pos = np.array([0, 0, 0])
        last_image_pos = np.array([0, 0, 0])

    delta_r = float(dicom_first.PixelSpacing[0])
    delta_c = float(dicom_first.PixelSpacing[1])

    if len(sorted_dicoms) == 1:
        step = [0, 0, -1]
    else:
        step = (image_pos - last_image_pos) / (1 - len(sorted_dicoms))

    return np.matrix(
        [
            [-image_orient1[0] * delta_c, -image_orient2[0] * delta_r, -step[0], -image_pos[0]],
            [-image_orient1[1] * delta_c, -image_orient2[1] * delta_r, -step[1], -image_pos[1]],
            [image_orient1[2] * delta_c, image_orient2[2] * delta_r, step[2], image_pos[2]],
            [0, 0, 0, 1],
        ]
    )


def dcm2nifti(dir_path: str, transpose: bool = False):
    """Convert a DICOM directory (or a single enhanced DICOM file) to a RAS NIfTI image."""
    if os.path.isdir(dir_path):
        files = sorted(glob(os.path.join(dir_path, "**", "*.dcm"), recursive=True), key=order_slices)
    else:
        files = [dir_path]

    if is_enhanced_dicom(files[0]):
        volumes, _ = extract_volumes_from_enhanced_dicom(files[0])
        if len(volumes) > 1:
            logging.info(f"Enhanced DICOM with {len(volumes)} volumes found. Returning first volume only.")
        nifti = volumes[0]
    else:
        affine = create_affine(files)
        volume = np.array([prepare_slices(f) for f in files])
        volume = np.transpose(volume, (2, 1, 0))
        nifti = nib.Nifti1Image(volume, affine)

    nifti = resample(nifti)

    if transpose:
        nifti = np.transpose(nifti.get_fdata().copy(), (2, 0, 1))

    return nifti


def is_enhanced_dicom(dcm_file: str) -> bool:
    enhanced_uids = [
        "1.2.840.10008.5.1.4.1.1.4.1", "1.2.840.10008.5.1.4.1.1.2.1",
        "1.2.840.10008.5.1.4.1.1.4.3", "1.2.840.10008.5.1.4.1.1.128",
        "1.2.840.10008.5.1.4.1.1.2.2", "1.2.840.10008.5.1.4.1.1.4.4",
    ]
    dcm = pydicom.dcmread(dcm_file, stop_before_pixels=True)
    return hasattr(dcm, "SOPClassUID") and str(dcm.SOPClassUID) in enhanced_uids


def extract_volumes_from_enhanced_dicom(dcm_file: str) -> tuple[list[nib.Nifti1Image], pydicom.Dataset]:
    dcm = pydicom.dcmread(dcm_file)
    if not hasattr(dcm, "NumberOfFrames"):
        volume = dcm.pixel_array
        if volume.ndim == 2:
            volume = volume[np.newaxis, ...]
        affine = create_affine_enhanced(dcm, frame_indices=[0])
        nifti = nib.Nifti1Image(volume.transpose(2, 1, 0), affine)
        return [resample(nifti)], dcm

    num_frames = int(dcm.NumberOfFrames)
    pixel_array = dcm.pixel_array
    volume_groups = group_frames_by_volume(dcm, num_frames)

    volumes = []
    for frame_indices in volume_groups:
        volume_data = pixel_array[frame_indices]
        affine = create_affine_enhanced(dcm, frame_indices)
        nifti = nib.Nifti1Image(np.transpose(volume_data, (2, 1, 0)), affine)
        volumes.append(resample(nifti))
    return volumes, dcm


def group_frames_by_volume(dcm: pydicom.Dataset, num_frames: int) -> list[list[int]]:
    if not hasattr(dcm, "PerFrameFunctionalGroupsSequence"):
        return [list(range(num_frames))]

    frame_metadata = []
    for frame_idx in range(num_frames):
        per_frame = dcm.PerFrameFunctionalGroupsSequence[frame_idx]
        stack_id = temporal_pos = in_stack_pos = dimension_idx = None

        if hasattr(per_frame, "FrameContentSequence") and len(per_frame.FrameContentSequence) > 0:
            fc = per_frame.FrameContentSequence[0]
            if hasattr(fc, "StackID"):
                stack_id = fc.StackID
            if hasattr(fc, "InStackPositionNumber"):
                in_stack_pos = int(fc.InStackPositionNumber)
            if hasattr(fc, "TemporalPositionIndex"):
                temporal_pos = int(fc.TemporalPositionIndex)
            if hasattr(fc, "DimensionIndexValues"):
                dimension_idx = tuple(fc.DimensionIndexValues)

        frame_metadata.append({
            "frame_idx": frame_idx, "stack_id": stack_id,
            "temporal_pos": temporal_pos if temporal_pos is not None else 1,
            "in_stack_pos": in_stack_pos, "dimension_idx": dimension_idx,
        })

    volume_dict = defaultdict(list)
    for fm in frame_metadata:
        if fm["stack_id"] is not None:
            key = (fm["stack_id"], fm["temporal_pos"])
        elif fm["dimension_idx"] is not None:
            key = (fm["dimension_idx"][0] if len(fm["dimension_idx"]) > 0 else 0, fm["temporal_pos"])
        else:
            key = (0, fm["temporal_pos"])
        volume_dict[key].append(fm["frame_idx"])

    volume_groups = []
    for key in sorted(volume_dict.keys()):
        frame_indices = volume_dict[key]
        if frame_metadata[frame_indices[0]]["in_stack_pos"] is not None:
            frame_indices = sorted(frame_indices, key=lambda idx: frame_metadata[idx]["in_stack_pos"])
        volume_groups.append(frame_indices)
    return volume_groups


def create_affine_enhanced(dcm: pydicom.Dataset, frame_indices: list[int]) -> np.matrix:
    first_frame_idx, last_frame_idx = frame_indices[0], frame_indices[-1]

    image_orient1, image_orient2 = np.array([1, 0, 0]), np.array([0, 1, 0])
    delta_r = delta_c = 1.0
    if hasattr(dcm, "SharedFunctionalGroupsSequence") and len(dcm.SharedFunctionalGroupsSequence) > 0:
        shared = dcm.SharedFunctionalGroupsSequence[0]
        if hasattr(shared, "PlaneOrientationSequence") and len(shared.PlaneOrientationSequence) > 0:
            image_orient = np.array(shared.PlaneOrientationSequence[0].ImageOrientationPatient)
            image_orient1, image_orient2 = image_orient[0:3], image_orient[3:6]
        if hasattr(shared, "PixelMeasuresSequence") and len(shared.PixelMeasuresSequence) > 0:
            pm = shared.PixelMeasuresSequence[0]
            if hasattr(pm, "PixelSpacing"):
                delta_r, delta_c = float(pm.PixelSpacing[0]), float(pm.PixelSpacing[1])

    per_frame_first = dcm.PerFrameFunctionalGroupsSequence[first_frame_idx]
    per_frame_last = dcm.PerFrameFunctionalGroupsSequence[last_frame_idx]

    image_pos = np.array([0, 0, 0])
    if hasattr(per_frame_first, "PlanePositionSequence") and len(per_frame_first.PlanePositionSequence) > 0:
        image_pos = np.array(per_frame_first.PlanePositionSequence[0].ImagePositionPatient)

    last_image_pos = image_pos
    if hasattr(per_frame_last, "PlanePositionSequence") and len(per_frame_last.PlanePositionSequence) > 0:
        last_image_pos = np.array(per_frame_last.PlanePositionSequence[0].ImagePositionPatient)

    if len(frame_indices) == 1:
        step = np.array([0, 0, -1])
    else:
        step = (image_pos - last_image_pos) / (1 - len(frame_indices))

    return np.matrix([
        [-image_orient1[0] * delta_c, -image_orient2[0] * delta_r, -step[0], -image_pos[0]],
        [-image_orient1[1] * delta_c, -image_orient2[1] * delta_r, -step[1], -image_pos[1]],
        [image_orient1[2] * delta_c, image_orient2[2] * delta_r, step[2], image_pos[2]],
        [0, 0, 0, 1],
    ])


def reconstruct_enhanced_dicom_from_niftis(nifti_paths: list, original_dcm_path: str, output_path: str):
    """Write processed NIfTI volumes back into a copy of the original enhanced DICOM."""
    dcm = pydicom.dcmread(original_dcm_path)
    num_frames = int(dcm.NumberOfFrames)

    volume_groups = group_frames_by_volume(dcm, num_frames)
    if len(nifti_paths) != len(volume_groups):
        raise ValueError("Number of NIfTIs does not match number of volumes in original DICOM.")

    all_frames = []
    for nifti_path, frame_indices in zip(nifti_paths, volume_groups):
        data = nib.load(nifti_path).get_fdata()
        if data.ndim != 3:
            raise ValueError("Processed NIfTI must be 3D.")
        data = np.transpose(data, (2, 1, 0))

        if data.shape[0] < len(frame_indices):
            pad = np.zeros((len(frame_indices) - data.shape[0], data.shape[1], data.shape[2]))
            data = np.concatenate([data, pad], axis=0)
        elif data.shape[0] > len(frame_indices):
            data = data[: len(frame_indices)]
        all_frames.extend(data[i, ...] for i in range(len(frame_indices)))

    new_pixel_array = np.array(all_frames).astype(dcm.pixel_array.dtype)
    dcm.PixelData = new_pixel_array.tobytes()
    pydicom.dcmwrite(os.path.join(output_path, os.path.basename(original_dcm_path)), dcm)

    for nifti_path in nifti_paths:
        if os.path.exists(nifti_path):
            os.remove(nifti_path)


# ---------------------------------------------------------------------------
# Datasets
# ---------------------------------------------------------------------------

class SegmentationDataset(Dataset):
    """
    Training / validation dataset.

    Args:
        path_list: DataFrame (or dict of lists) with columns image_path, mask_path.
                   Volumes must already have the network input size.
        train:     Apply augmentation when True.
    """

    def __init__(self, path_list, train: bool) -> None:
        super().__init__()
        self.path_list = path_list
        self.train = train

        # Spatial transforms: exactly one is applied per sample.
        self.spatial_transforms = {
            "affine": tio.RandomAffine(),
            "elastic": tio.RandomElasticDeformation(num_control_points=5),
        }
        self.spatial_weights = {"affine": 0.75, "elastic": 0.25}

        # Intensity transforms, each applied independently with p = 0.25.
        self.intensity_transforms = [
            tio.RandomBiasField(),
            tio.RandomNoise(),
            tio.RandomGamma(),
            tio.RandomSpike(),
        ]
        self.intensity_p = 0.25

    def __len__(self) -> int:
        return len(self.path_list)

    def __getitem__(self, idx: int) -> dict:
        nvtx_push("algernon/aug/io")
        image_nifti = nib.load(self.path_list["image_path"][idx])
        mask_nifti = nib.load(self.path_list["mask_path"][idx])
        image_tensor = torch.tensor(image_nifti.get_fdata()[None, ...].copy(), dtype=torch.float32)
        mask_tensor = torch.tensor(mask_nifti.get_fdata()[None, ...].copy(), dtype=torch.float32)
        mask_tensor = remap_labels(mask_tensor)
        nvtx_pop()

        subject = tio.Subject(
            image=tio.ScalarImage(tensor=image_tensor),
            mask=tio.LabelMap(tensor=mask_tensor),
        )

        if self.train:
            # MinatoSegLoader pre-schedules affine/elastic per index (3:1
            # interleaving) so that no worker gets two elastic samples in a row.
            forced = os.environ.get("MINATO_FORCE_TRANSFORM", "")
            if forced in self.spatial_transforms:
                name = forced
            else:
                names, weights = zip(*self.spatial_weights.items())
                name = random.choices(names, weights=weights, k=1)[0]
            nvtx_push(f"algernon/aug/{name}")
            subject = self.spatial_transforms[name](subject)
            nvtx_pop()

            for transform in self.intensity_transforms:
                if random.random() < self.intensity_p:
                    nvtx_push(f"algernon/aug/{type(transform).__name__}")
                    subject = transform(subject)
                    nvtx_pop()

        image_tensor = subject["image"].data
        mask_tensor = subject["mask"].data

        nvtx_push("algernon/aug/normalize")
        if image_tensor.max() > 0:
            image_tensor = (image_tensor - image_tensor.min()) / (image_tensor.max() - image_tensor.min())
        nvtx_pop()

        return {"image": image_tensor, "mask": mask_tensor}


class InferenceDataset(Dataset):
    """
    Dataset for raw exams (NIfTI or DICOM): reorients to RAS, resizes to the
    network input size and normalises. No ground truth is required.
    """

    def __init__(self, path_list: dict, input_size=DEFAULT_INPUT_SIZE) -> None:
        super().__init__()
        self.path_list = path_list
        self.up = torch.nn.Upsample(size=tuple(input_size))

    def __len__(self) -> int:
        return len(self.path_list["image_path"])

    def __getitem__(self, idx: int) -> dict:
        file_path = self.path_list["image_path"][idx]

        if file_path.endswith(".nii") or file_path.endswith(".nii.gz"):
            pixels = resample(nib.load(file_path)).get_fdata().copy()
        else:
            pixels = dcm2nifti(file_path, transpose=True)

        x = torch.from_numpy(pixels).unsqueeze(0).float()
        x = self.up(x.unsqueeze(0)).squeeze(0)  # Upsample expects a batch dim

        if x.max() > 0:
            x = (x - x.min()) / (x.max() - x.min())

        return {"image": x, "file_name": file_path}


# ---------------------------------------------------------------------------
# Loaders
# ---------------------------------------------------------------------------

def get_loaders(
    train_paths,
    val_paths,
    batch_size: int = 1,
    train_workers: int = 16,
    val_workers: int = 8,
    train_queue_size: int = 64,
    val_queue_size: int = 32,
    debug: bool = False,
    log_dir: str | None = None,
) -> tuple:
    """
    Build the DDP training and validation loaders on top of MinatoSegLoader.

    The defaults were tuned for an A100 with batch_size=1 per GPU: 16 workers
    produce augmented samples faster than the GPU consumes them (mean sample
    cost ~1.4 s with elastic outliers vs. ~0.2 s per training step), and a
    queue of 4x the worker count absorbs the elastic-deformation outliers.
    Validation has no spatial augmentation, so 8 workers suffice.
    """
    train_dataset = SegmentationDataset(train_paths, train=True)
    val_dataset = SegmentationDataset(val_paths, train=False)

    train_sampler = DistributedSampler(train_dataset)
    val_sampler = DistributedSampler(val_dataset, shuffle=False, drop_last=True)

    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    pin_device = f"cuda:{local_rank}"

    train_loader = MinatoSegLoader(
        train_dataset,
        sampler=train_sampler,
        batch_size=batch_size,
        num_workers=train_workers,
        pin_memory=True,
        pin_memory_device=pin_device,
        queue_size=train_queue_size,
        drop_last=True,
        debug=debug,
        log_dir=log_dir,
        name="train",
    )
    val_loader = MinatoSegLoader(
        val_dataset,
        sampler=val_sampler,
        batch_size=batch_size,
        num_workers=val_workers,
        pin_memory=True,
        pin_memory_device=pin_device,
        queue_size=val_queue_size,
        drop_last=True,
        debug=debug,
        log_dir=log_dir,
        name="val",
    )
    return train_loader, val_loader


def get_inference_loader(data_path: str, batch_size: int = 1, input_size=DEFAULT_INPUT_SIZE) -> DataLoader:
    """Loader over a file or directory of raw exams (.nii, .nii.gz, .dcm or DICOM folders)."""
    candidates = glob(os.path.join(data_path, "*")) if os.path.isdir(data_path) else [data_path]

    valid_paths = {"image_path": []}
    for file in candidates:
        fn = file.lower()
        if fn.endswith((".nii", ".nii.gz", ".dcm")) or os.path.isdir(file):
            valid_paths["image_path"].append(file)
        else:
            try:
                pydicom.dcmread(file, stop_before_pixels=True)
                valid_paths["image_path"].append(file)
            except Exception:
                logging.warning(f"Unsupported file format: {file}. Accepted: .nii, .nii.gz, .dcm")

    dataset = InferenceDataset(valid_paths, input_size=input_size)
    return DataLoader(dataset, batch_size=batch_size, num_workers=4, pin_memory=True, prefetch_factor=4)
