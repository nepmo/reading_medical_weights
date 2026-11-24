"""LIDC-IDRI: series index, nodule table from pylidc annotations, crop extraction.

Settings are fixed in PROTOCOL.md section C. pylidc is used only for the
annotations (its bundled sqlite); volumes are read here with pydicom.
"""

from pathlib import Path

import numpy as np
from scipy import ndimage

# Crop geometry (PROTOCOL.md C3).
MM_PER_PX = 1.25
STORE_SIZE = 48          # stored raw-HU crop, 60 mm field of view
MODEL_SIZE = 24          # model input, central 24x24 (30 mm)
HU_MIN, HU_MAX = -1024, 3071
LUNG_LO, LUNG_HI = -1350.0, 150.0   # W 1500 / L -600
Z_OFFSET_MAX_MM = 2.0
Z_OFFSET_FRAC = 0.3
SIZENORM_PX = 12         # rung 2b (PROTOCOL.md G1): nodule diameter spans 12 of 24 px

# Inclusion and label (PROTOCOL.md C2).
MIN_READERS = 3
MAX_ANNOTATIONS = 4   # LIDC has 4 readers; larger clusters are pylidc merges of neighbours
MIN_DIAMETER_MM = 3.0
RATINGS = ["subtlety", "internalStructure", "calcification", "sphericity",
           "margin", "lobulation", "spiculation", "texture", "malignancy"]


def label_from_median(m: float) -> int:
    """Strict rule: <= 2 benign (0), >= 4 malignant (1), anything else -1."""
    if m <= 2:
        return 0
    if m >= 4:
        return 1
    return -1


def to_model_input(hu48: np.ndarray) -> np.ndarray:
    """Central 24x24 of the stored 48x48 HU crop(s), lung window -> [0, 1] float32.

    Works on any array whose last two axes are 48x48.
    """
    a = (STORE_SIZE - MODEL_SIZE) // 2
    return window(np.asarray(hu48)[..., a:a + MODEL_SIZE, a:a + MODEL_SIZE])


def window(hu: np.ndarray) -> np.ndarray:
    """Lung window W 1500 / L -600: HU -> [0, 1] float32, clipped."""
    c = np.asarray(hu, dtype=np.float32)
    return np.clip((c - LUNG_LO) / (LUNG_HI - LUNG_LO), 0.0, 1.0)


def sizenorm_mm_per_px(diameter_mm: float) -> float:
    """Pixel size that makes the nodule diameter span SIZENORM_PX pixels (PROTOCOL.md G1)."""
    return float(diameter_mm) / SIZENORM_PX


def z_offsets(diameter_mm: float) -> list:
    d = min(Z_OFFSET_MAX_MM, Z_OFFSET_FRAC * diameter_mm)
    return [-d, 0.0, d]


# ---------------------------------------------------------------- DICOM

def dcm_files(series_dir: Path) -> list:
    """Only real .dcm files (the download left empty *.dcmNNNN retry files)."""
    return sorted(p for p in Path(series_dir).iterdir() if p.is_file() and p.suffix == ".dcm")


def _str(v) -> str:
    if v is None:
        return ""
    if isinstance(v, (list, tuple)) or type(v).__name__ == "MultiValue":
        return "/".join(str(x) for x in v)
    return str(v).strip()


def read_series_header(series_dir: Path) -> dict:
    import pydicom
    files = dcm_files(series_dir)
    h = pydicom.dcmread(files[0], stop_before_pixels=True)
    ps = [float(x) for x in h.PixelSpacing]
    return {
        "series_uid": str(h.SeriesInstanceUID),
        "study_uid": str(h.StudyInstanceUID),
        "patient_id": str(h.PatientID),
        "series_dir": str(series_dir),
        "n_files": len(files),
        "manufacturer": _str(getattr(h, "Manufacturer", None)),
        "model_name": _str(getattr(h, "ManufacturerModelName", None)),
        "kernel": _str(getattr(h, "ConvolutionKernel", None)),
        "slice_thickness": float(h.SliceThickness) if getattr(h, "SliceThickness", None) not in (None, "") else np.nan,
        "pixel_spacing": ps[0],
        "kvp": float(h.KVP) if getattr(h, "KVP", None) not in (None, "") else np.nan,
        "tube_current": float(h.XRayTubeCurrent) if getattr(h, "XRayTubeCurrent", None) not in (None, "") else np.nan,
    }


def find_ct_series(root: Path) -> list:
    """All LIDC-IDRI-XXXX/<study>/CT_<series> directories."""
    return sorted(p for p in Path(root).glob("LIDC-IDRI-*/*/CT_*") if p.is_dir())


AXIAL = np.array([1, 0, 0, 0, 1, 0], float)
AXIAL_ROT180 = np.array([-1, 0, 0, 0, -1, 0], float)  # 11 GE series: image stored rotated 180 deg


def load_volume(series_dir: Path) -> dict:
    """Read a CT series -> dict(vol int16 (k, rows, cols) HU sorted by z, z, spacing, qa)."""
    import pydicom
    ds = [pydicom.dcmread(f) for f in dcm_files(series_dir)]
    ds = [d for d in ds if hasattr(d, "ImagePositionPatient") and hasattr(d, "pixel_array")]
    ds.sort(key=lambda d: float(d.ImagePositionPatient[2]))
    z = np.array([float(d.ImagePositionPatient[2]) for d in ds])
    keep = np.r_[True, np.diff(z) > 1e-3]          # drop duplicate z positions
    ds = [d for d, k in zip(ds, keep) if k]
    z = z[keep]
    xy = np.array([[float(d.ImagePositionPatient[0]), float(d.ImagePositionPatient[1])] for d in ds])
    ors = {tuple(np.round([float(v) for v in d.ImageOrientationPatient], 3)) for d in ds}
    orientation = "axial" if all(np.allclose(o, AXIAL) for o in ors) else (
        "axial_rot180" if all(np.allclose(o, AXIAL_ROT180) for o in ors) else "other")
    vol = np.empty((len(ds), int(ds[0].Rows), int(ds[0].Columns)), np.int16)
    for i, d in enumerate(ds):
        hu = d.pixel_array.astype(np.float32) * float(getattr(d, "RescaleSlope", 1)) + float(getattr(d, "RescaleIntercept", 0))
        vol[i] = np.clip(np.rint(hu), HU_MIN, HU_MAX).astype(np.int16)
    ps = [float(v) for v in ds[0].PixelSpacing]
    return {
        "vol": vol, "z": z, "row_mm": ps[0], "col_mm": ps[1],
        "qa": {
            "n_dup_z": int((~keep).sum()),
            "xy_drift_mm": float(np.abs(xy - xy[0]).max()),
            "orientation": orientation,
        },
    }


# ---------------------------------------------------------------- annotations

def import_pylidc():
    """pylidc 0.2.3 uses np.int / np.float, removed in numpy >= 1.24."""
    if not hasattr(np, "int"):
        np.int = int
    if not hasattr(np, "float"):
        np.float = float
    import pylidc
    return pylidc


def _polygon(points_rc: np.ndarray):
    """Signed-free area and centroid of a closed polygon given as (row, col) points."""
    y, x = points_rc[:, 0].astype(float), points_rc[:, 1].astype(float)
    x1, y1 = np.roll(x, -1), np.roll(y, -1)
    cross = x * y1 - x1 * y
    a = cross.sum() / 2
    if abs(a) < 1e-6:            # degenerate (1-2 point contour): point mean, unit weight
        return 1.0, y.mean(), x.mean()
    cy = ((y + y1) * cross).sum() / (6 * a)
    cx = ((x + x1) * cross).sum() / (6 * a)
    return abs(a), cy, cx


def reader_centroid(contours) -> np.ndarray:
    """Area-weighted centroid (row, col, z_mm) of one reader's contours.

    contours: iterable of (points_rc (n,2), z_mm, inclusion bool).
    """
    w, acc = 0.0, np.zeros(3)
    for pts, z, inc in contours:
        a, cy, cx = _polygon(np.asarray(pts))
        s = a if inc else -a
        w += s
        acc += s * np.array([cy, cx, z])
    if w <= 0:   # pathological: fall back to plain mean of inclusion points
        pts = [(p, z) for p, z, inc in contours if inc]
        allp = np.vstack([np.c_[p, np.full(len(p), z)] for p, z in pts]).astype(float)
        return allp.mean(0)
    return acc / w


def _contours_of(ann):
    # pylidc Contour.coords is "x,y" per line, i.e. (col, row)
    return [(c.to_matrix(include_k=False), float(c.image_z_position), bool(c.inclusion)) for c in ann.contours]


def nodules_for_scan(scan) -> list:
    """Rows (dicts) for every nodule in a pylidc Scan meeting the inclusion rule."""
    rows = []
    clusters = scan.cluster_annotations(verbose=False)
    for k, anns in enumerate(clusters):
        if len(anns) < MIN_READERS or len(anns) > MAX_ANNOTATIONS:
            continue
        diam = float(np.median([a.diameter for a in anns]))
        if diam < MIN_DIAMETER_MM:
            continue
        cents = np.array([reader_centroid(_contours_of(a)) for a in anns])
        c = cents.mean(0)
        zs = [float(ct.image_z_position) for a in anns for ct in a.contours]
        mal_med = float(np.median([a.malignancy for a in anns]))
        row = {
            "nodule_id": f"{scan.patient_id}_s{scan.id}_n{k}",
            "patient_id": scan.patient_id,
            "series_uid": scan.series_instance_uid,
            "study_uid": scan.study_instance_uid,
            "scan_id": scan.id,
            "cluster_idx": k,
            "n_readers": len(anns),
            "diameter_mm": diam,
            "c_row": c[0], "c_col": c[1], "c_z": c[2],
            "reader_spread_mm": float(np.max(np.linalg.norm(
                (cents - c) * np.array([scan.pixel_spacing, scan.pixel_spacing, 1.0]), axis=1))),
            "z_min": min(zs), "z_max": max(zs),
            "label": label_from_median(mal_med),
        }
        for r in RATINGS:
            vals = [getattr(a, r) for a in anns]
            row[f"{r}_median"] = float(np.median(vals))
            row[f"{r}_readers"] = ",".join(str(v) for v in vals)
        rows.append(row)
    return rows


# ---------------------------------------------------------------- crops

def _slice_at(vol: np.ndarray, z: np.ndarray, zq: float):
    """Linear interpolation between neighbouring slices at physical z. Returns (slice, in_range)."""
    in_range = z[0] - 1e-3 <= zq <= z[-1] + 1e-3
    k = float(np.interp(zq, z, np.arange(len(z))))
    k0 = int(np.floor(k))
    k1 = min(k0 + 1, len(z) - 1)
    t = k - k0
    s = vol[k0].astype(np.float32)
    if t > 1e-6:
        s = (1 - t) * s + t * vol[k1].astype(np.float32)
    return s, in_range


def extract_crops(vol, z, row_mm, col_mm, centroid, offsets_mm, size=STORE_SIZE, mm=MM_PER_PX) -> np.ndarray:
    """(len(offsets), size, size) float32 HU crops centred on centroid=(row, col, z_mm)."""
    r0, c0, z0 = centroid
    off = (np.arange(size) - (size - 1) / 2) * mm
    rr, cc = np.meshgrid(r0 + off / row_mm, c0 + off / col_mm, indexing="ij")
    sig = [0.5 * mm / row_mm if mm / row_mm > 1 else 0.0, 0.5 * mm / col_mm if mm / col_mm > 1 else 0.0]
    out = np.empty((len(offsets_mm), size, size), np.float32)
    for i, dz in enumerate(offsets_mm):
        s, _ = _slice_at(vol, z, z0 + dz)
        if max(sig) > 0:
            s = ndimage.gaussian_filter(s, sig, mode="constant", cval=HU_MIN)
        out[i] = ndimage.map_coordinates(s, [rr, cc], order=1, mode="constant", cval=HU_MIN)
    return out


def crops_for_series(series_dir: str, nodules: list, sizenorm: bool = False) -> dict:
    """Load one series, extract 3 crops per nodule. nodules: list of row dicts.

    sizenorm=False: 48x48 at 1.25 mm/px (C3). sizenorm=True: 24x24 at diameter/12 mm/px (G1)."""
    v = load_volume(Path(series_dir))
    if v["qa"]["orientation"] == "other":
        raise ValueError(f"non-axial series {series_dir}")
    hu, qa = [], []
    for n in nodules:
        offs = z_offsets(n["diameter_mm"])
        geom = (dict(size=MODEL_SIZE, mm=sizenorm_mm_per_px(n["diameter_mm"])) if sizenorm
                else dict(size=STORE_SIZE, mm=MM_PER_PX))
        c = extract_crops(v["vol"], v["z"], v["row_mm"], v["col_mm"], (n["c_row"], n["c_col"], n["c_z"]), offs, **geom)
        if v["qa"]["orientation"] == "axial_rot180":   # to standard patient orientation; centre stays at the middle
            c = c[:, ::-1, ::-1]
        hu.append(c)
        # contours are drawn on slices: their z must coincide with a slice position
        zmis = max(float(np.min(np.abs(v["z"] - n[k]))) for k in ("z_min", "z_max"))
        cz_ok = all(v["z"][0] - 1e-3 <= n["c_z"] + d <= v["z"][-1] + 1e-3 for d in offs)
        qa.append({"nodule_id": n["nodule_id"], "contour_z_mismatch_mm": zmis, "z_in_range": cz_ok,
                   "n_slices": len(v["z"]), "slice_spacing": float(np.median(np.diff(v["z"]))), **v["qa"]})
    return {"hu": np.stack(hu).astype(np.float16), "qa": qa}
