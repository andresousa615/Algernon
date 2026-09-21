# -*- coding: utf-8 -*-
"""
Per-region anonymisation of a 3D MRI volume from a facial-structure segmentation.

Each region (nose, eyes, ears, mouth) has two pieces:
  * a COVER: the set of voxels its anonymisation would touch (mask + dilation /
    projection), computed without applying anything;
  * a FILL: the transform applied inside the cover (zero, plateau, noise).

Keeping them separate lets the officer protect a region the user asked to
preserve using exactly the area its own anonymisation would have occupied.
"""
import numpy as np
from scipy import ndimage
from skimage.filters import threshold_triangle
from skimage.measure import label as sk_label, regionprops


# Label mapping (matches data.dataset.remap_labels)
LABEL_BACKGROUND = 0
LABEL_EARS = 1
LABEL_MOUTH = 2
LABEL_NOSE = 3
LABEL_EYES = 4

# User-selectable regions. The order is the APPLICATION order (not the
# command-line order): keeping the same sequence guarantees that selecting all
# regions yields exactly the same volume as the default.
ALL_REGIONS = ("nose", "eyes", "ears", "mouth")

REGION_LABELS = {
    "nose": LABEL_NOSE,
    "eyes": LABEL_EYES,
    "ears": LABEL_EARS,
    "mouth": LABEL_MOUTH,
}


def resolve_regions(anonymise=None, preserve=None) -> tuple:
    """
    Resolve the regions to anonymise from the command-line options, returned in
    application order.

    Exactly one of the two forms is accepted:
      anonymise : regions to anonymise ("all" selects every region)
      preserve  : regions to keep; the complement is anonymised ("none" keeps nothing)

    Unknown names raise ValueError on purpose: in an anonymisation pipeline, a
    typo that went unnoticed would silently anonymise less than requested.
    """
    def _clean(v):
        if v is None:
            return None
        if isinstance(v, str):
            v = v.split(",")
        return [s.strip().lower() for s in v if s and s.strip()]

    anonymise, preserve = _clean(anonymise), _clean(preserve)

    if anonymise and preserve:
        raise ValueError("Give either the regions to anonymise OR the regions to preserve, not both.")

    def _validate(names):
        unknown = [n for n in names if n not in REGION_LABELS]
        if unknown:
            raise ValueError(f"Unknown region(s): {', '.join(unknown)}. Valid: {', '.join(ALL_REGIONS)}.")

    if preserve:
        if preserve == ["none"]:
            return ALL_REGIONS
        _validate(preserve)
        return tuple(r for r in ALL_REGIONS if r not in preserve)

    if anonymise:
        if anonymise == ["all"]:
            return ALL_REGIONS
        _validate(anonymise)
        return tuple(r for r in ALL_REGIONS if r in anonymise)

    return ALL_REGIONS


class AnonymizationOfficer:
    """
    Apply per-region anonymisation to a 3D MRI volume given a segmentation mask.

    Args:
        img_data  (np.ndarray): Original volume in RAS orientation, shape (X, Y, Z).
        pred_mask (np.ndarray): Argmax segmentation mask, same shape, values {0,1,2,3,4}.
        regions   (sequence):   Regions to anonymise, from ALL_REGIONS. Default: all.
        protect_preserved (bool): When only some regions are anonymised, the
                                transforms dilate beyond the mask and may invade
                                a region the user asked to keep. With this flag
                                the COVER of every preserved region (the same
                                area its own anonymisation would occupy) is
                                restored from the original volume at the end.
                                No effect when all regions are anonymised.
        preserve_margin (int):  Extra voxels protected beyond the cover. Usually
                                unnecessary; 0 by default.
    """

    def __init__(self, img_data: np.ndarray, pred_mask: np.ndarray, regions=None,
                 protect_preserved: bool = True, preserve_margin: int = 0) -> None:
        self.img_data = img_data
        self.pred_mask = pred_mask
        # Gaussian-filtered volume for plateau values, computed once from the original data.
        self.img_smooth = ndimage.gaussian_filter(img_data, sigma=3)

        if regions is None:
            self.regions = ALL_REGIONS
        else:
            unknown = [r for r in regions if r not in REGION_LABELS]
            if unknown:
                raise ValueError(f"Unknown region(s): {', '.join(unknown)}. Valid: {', '.join(ALL_REGIONS)}.")
            self.regions = tuple(r for r in ALL_REGIONS if r in regions)

        self.preserved = tuple(r for r in ALL_REGIONS if r not in self.regions)
        self.protect_preserved = protect_preserved
        self.preserve_margin = max(0, int(preserve_margin))

    def anonymize(self) -> np.ndarray:
        """Run the selected per-region transforms and return the anonymised volume."""
        deid = self.img_data.copy()

        handlers = {
            "nose": self._process_nose,
            "eyes": self._process_eyes,
            "ears": self._process_ears,
            "mouth": self._process_mouth,
        }
        for region in self.regions:          # ALL_REGIONS already fixes the order
            deid = handlers[region](deid)

        if self.protect_preserved and self.preserved:
            # Protection uses the COVER of each preserved region — the area its
            # anonymisation would occupy, not just the voxels the model assigned
            # to it. For the eyes this includes the anterior projection; for the
            # ears the lateral dilation; for the nose its dilation. Covers depend
            # only on the predicted mask, so each is computed once.
            builders = {
                "nose": self._cover_nose,
                "eyes": self._cover_eyes,
                "ears": self._cover_ears,
                "mouth": self._cover_mouth,
            }
            covers = {r: builders[r]() for r in ALL_REGIONS}

            keep = np.zeros(self.img_data.shape, dtype=bool)
            for region in self.preserved:
                keep |= covers[region]

            if self.preserve_margin > 0 and keep.any():
                keep = ndimage.binary_dilation(keep, iterations=self.preserve_margin)

            # NOTE: conflict resolution — where the cover of a PRESERVED region
            # overlaps the cover of an ANONYMISED one, the anonymised region
            # wins. Covers extend beyond the label (the eye cover projects 12
            # voxels forward, the nose cover is dilated twice), so nose/eye
            # overlaps are common. Restoring the original there would undo the
            # anonymisation already applied; privacy takes precedence over
            # utility in that handful of voxels. Outside the intersection the
            # preserved region is restored in full.
            treated = np.zeros(self.img_data.shape, dtype=bool)
            for region in self.regions:
                treated |= covers[region]
            keep &= ~treated

            if keep.any():
                deid[keep] = self.img_data[keep]

        return deid

    # ------------------------------------------------------------------
    # Covers — the area each transform occupies, without applying it
    # ------------------------------------------------------------------

    def _cover_nose(self) -> np.ndarray:
        nose_mask = (self.pred_mask == LABEL_NOSE)
        if not nose_mask.any():
            return nose_mask
        return ndimage.binary_dilation(nose_mask, iterations=2)

    def _cover_mouth(self) -> np.ndarray:
        return (self.pred_mask == LABEL_MOUTH)

    def _cover_eyes(self) -> np.ndarray:
        """
        Eye mask plus, for each eye component, a cylindrical ellipse extending
        12 voxels beyond its anterior face.
        """
        eye_mask = (self.pred_mask == LABEL_EYES)
        if not eye_mask.any():
            return eye_mask

        lbl_eye = sk_label(eye_mask, connectivity=1)
        eye_cover = eye_mask.copy()

        for region in regionprops(lbl_eye):
            component = (lbl_eye == region.label)

            # Anterior-posterior extent along Y
            y_min = region.bbox[1]
            y_max = region.bbox[4]
            y_depth = y_max - y_min
            y_start = y_max - max(y_depth // 2, 1)
            y_end = min(y_max + 12, self.img_data.shape[1])

            # XZ projection -> elliptical dilation (wide in X, narrow in Z)
            xz_proj = np.any(component, axis=1)
            struct_x = np.ones((3, 1), dtype=bool)
            struct_z = np.ones((1, 3), dtype=bool)
            xz_temp = ndimage.binary_dilation(xz_proj, structure=struct_x, iterations=4)
            xz_cylinder = ndimage.binary_dilation(xz_temp, structure=struct_z, iterations=1)

            eye_cover[:, y_start:y_end, :] |= xz_cylinder[:, np.newaxis, :]

        return eye_cover

    def _cover_ears(self) -> np.ndarray:
        """
        Ear regions dilated x6 outward (laterally) and in Y/Z to cover the ear
        edges, but blocked inward (medially) past the X midpoint of the ear in
        each axial slice.

        Per-axial-slice rule: for each Z slice, x_mid = (x_min + x_max) / 2 of
        the ear segmentation. Expansion is only allowed on the lateral half
        (X >= x_mid for the right ear, X <= x_mid for the left ear). For Z
        slices outside the ear range (dilation spill-over) the x_mid of the
        nearest ear slice is propagated so the constraint holds.
        """
        ear_mask = (self.pred_mask == LABEL_EARS)
        if not ear_mask.any():
            return ear_mask

        ear_cover = np.zeros(self.img_data.shape, dtype=bool)
        x_center = self.img_data.shape[0] / 2.0
        shape_z = self.img_data.shape[2]
        zs = np.arange(shape_z)

        lbl = sk_label(ear_mask, connectivity=1)
        for region in regionprops(lbl):
            z_range = region.bbox[5] - region.bbox[2]
            if z_range < 3:
                continue  # skip tiny components

            region_mask = (lbl == region.label)
            is_right_ear = (region.centroid[0] >= x_center)
            z_min_ear = region.bbox[2]
            z_max_ear = region.bbox[5]

            dilated = ndimage.binary_dilation(region_mask, iterations=6)

            # Cap Z two slices below the top of the ear, for the dilation AND the
            # original mask: the model sometimes includes 1-2 slices of temporal
            # bone at the top, which would otherwise leave triangular artefacts.
            z_cap = z_max_ear - 2
            dilated &= (zs <= z_cap)[np.newaxis, np.newaxis, :]
            region_masked = region_mask & (zs <= z_cap)[np.newaxis, np.newaxis, :]

            x_mid_per_z = np.full(shape_z, fill_value=-1, dtype=int)
            for z_idx in range(z_min_ear, min(z_cap + 1, shape_z)):
                ear_in_z = region_mask[:, :, z_idx]
                if not ear_in_z.any():
                    continue
                xs_in_z = np.where(ear_in_z.any(axis=1))[0]
                x_mid_per_z[z_idx] = int((xs_in_z.min() + xs_in_z.max()) / 2)

            # Propagate x_mid to Z slices outside the ear range
            for z_idx in range(z_min_ear - 1, -1, -1):
                ref = x_mid_per_z[z_idx + 1]
                if ref >= 0:
                    x_mid_per_z[z_idx] = ref
            for z_idx in range(z_cap + 1, shape_z):
                ref = x_mid_per_z[z_idx - 1]
                if ref >= 0:
                    x_mid_per_z[z_idx] = ref

            # Medial X clip at x_mid, per Z slice
            for z_idx in range(shape_z):
                x_mid = x_mid_per_z[z_idx]
                if x_mid < 0 or not dilated[:, :, z_idx].any():
                    continue
                if is_right_ear:
                    dilated[:x_mid, :, z_idx] = False
                else:
                    dilated[x_mid + 1:, :, z_idx] = False

            ear_cover |= region_masked | dilated

        return ear_cover

    # ------------------------------------------------------------------
    # Fills — the transform applied inside each cover
    # ------------------------------------------------------------------

    def _process_nose(self, deid: np.ndarray) -> np.ndarray:
        """Zero out the dilated nose mask."""
        cover = self._cover_nose()
        if cover.any():
            deid[cover] = 0
        return deid

    def _process_eyes(self, deid: np.ndarray) -> np.ndarray:
        """Fill the eye cover with the maximum of the smoothed volume inside the eye mask."""
        eye_mask = (self.pred_mask == LABEL_EYES)
        if not eye_mask.any():
            return deid
        thresh_eye = float(np.max(self.img_smooth[eye_mask]))
        deid[self._cover_eyes()] = thresh_eye
        return deid

    def _process_ears(self, deid: np.ndarray) -> np.ndarray:
        """Fill the ear cover with random noise (up to 80 % of the triangle threshold)."""
        ear_cover = self._cover_ears()
        if not ear_cover.any():
            return deid
        thresh_air = float(threshold_triangle(self.img_data))
        noise = (np.random.rand(*self.img_data.shape) * thresh_air * 0.8).astype(self.img_data.dtype)
        deid[ear_cover] = noise[ear_cover]
        return deid

    def _process_mouth(self, deid: np.ndarray) -> np.ndarray:
        """Fill the mouth mask with the maximum of the smoothed volume inside it."""
        mouth_mask = self._cover_mouth()
        if mouth_mask.any():
            deid[mouth_mask] = float(np.max(self.img_smooth[mouth_mask]))
        return deid
