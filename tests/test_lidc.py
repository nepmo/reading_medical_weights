"""LIDC preprocessing: geometry, label rule, window, splits, leakage check (no real data)."""

import numpy as np
import pandas as pd
import pytest

from rmw import lidc
from rmw.splits import LeakageError, leakage_check, patient_split


def test_label_rule_strict():
    assert [lidc.label_from_median(m) for m in (1, 2, 2.5, 3, 3.5, 4, 5)] == [0, 0, -1, -1, -1, 1, 1]


def test_lung_window_and_centre_crop():
    hu = np.full((48, 48), -600.0)
    hu[0, 0] = -1350           # outside the central 24x24, must be cut away
    hu[12, 12] = -1350
    hu[35, 35] = 150
    x = lidc.to_model_input(hu)
    assert x.shape == (24, 24) and x.dtype == np.float32
    assert x[0, 0] == 0.0 and x[23, 23] == 1.0
    assert np.isclose(x[5, 5], 0.5)
    assert lidc.to_model_input(np.full((2, 3, 48, 48), 3000.0)).max() == 1.0


def test_z_offsets():
    assert lidc.z_offsets(20) == [-2.0, 0.0, 2.0]
    assert np.allclose(lidc.z_offsets(4), [-1.2, 0, 1.2])


def _sphere_volume(centre_mm, radius_mm, ps=0.7, dz=1.25, shape=(80, 200, 200)):
    z = np.arange(shape[0]) * dz - 40.0
    rr, cc = np.mgrid[0:shape[1], 0:shape[2]] * ps
    vol = np.full(shape, -900, np.int16)
    for k, zk in enumerate(z):
        d2 = (rr - centre_mm[0]) ** 2 + (cc - centre_mm[1]) ** 2 + (zk - centre_mm[2]) ** 2
        vol[k][d2 <= radius_mm ** 2] = 50
    return vol, z, ps


def test_crop_centred_and_scaled():
    centre = (70.0, 63.0, 3.1)           # mm (row, col, z), off-grid on purpose
    radius = 6.0
    vol, z, ps = _sphere_volume(centre, radius)
    c = lidc.extract_crops(vol, z, ps, ps, (centre[0] / ps, centre[1] / ps, centre[2]), [-2, 0, 2])
    assert c.shape == (3, 48, 48)
    m = c[1] > -425                       # half-way threshold
    rr, cc = np.nonzero(m)
    assert abs(rr.mean() - 23.5) < 0.3 and abs(cc.mean() - 23.5) < 0.3
    diam_px = 2 * np.sqrt(m.sum() / np.pi)
    assert abs(diam_px * lidc.MM_PER_PX - 2 * radius) < 1.0
    # off-centre slices cut a smaller disk
    mass = (c + 900).sum(axis=(1, 2))
    assert mass[0] < mass[1] and mass[2] < mass[1]


def test_crop_outside_volume_is_air():
    vol, z, ps = _sphere_volume((10, 10, 0), 3)
    c = lidc.extract_crops(vol, z, ps, ps, (2, 2, 0.0), [0])
    assert c[0, 0, 0] == lidc.HU_MIN


def test_reader_centroid_area_weighted():
    sq = lambda r, c, s: np.array([[r, c], [r, c + s], [r + s, c + s], [r + s, c]])
    big, small = sq(0, 0, 10), sq(100, 100, 1)
    cen = lidc.reader_centroid([(big, 0.0, True), (small, 10.0, True)])
    assert np.allclose(cen, (np.array([5, 5, 0]) * 100 + np.array([100.5, 100.5, 10])) / 101)
    # exclusion contour subtracts its area
    hole = sq(0, 0, 5)
    cen2 = lidc.reader_centroid([(big, 0.0, True), (hole, 0.0, False)])
    assert cen2[0] > 5 and cen2[1] > 5


def test_dcm_files_skips_retry_leftovers(tmp_path):
    (tmp_path / "a.dcm").write_bytes(b"x")
    (tmp_path / "b.dcm123456").write_bytes(b"")
    assert [p.name for p in lidc.dcm_files(tmp_path)] == ["a.dcm"]


def _fake_nodules(n_pat=300, seed=0):
    rng = np.random.default_rng(seed)
    rows = []
    for p in range(n_pat):
        n_series = 2 if p % 25 == 0 else 1
        for s in range(n_series):
            for k in range(rng.integers(1, 4)):
                rows.append({"patient_id": f"P{p:04d}", "study_uid": f"st{p}_{s}", "series_uid": f"se{p}_{s}",
                             "nodule_id": f"P{p}_s{s}_n{k}", "label": int(rng.choice([0, 1, -1], p=[.5, .3, .2]))})
    return pd.DataFrame(rows)


def test_patient_split_no_leakage_and_proportions():
    df = _fake_nodules()
    df["split"] = patient_split(df)
    s = leakage_check(df)
    frac = df.groupby("split").patient_id.nunique() / df.patient_id.nunique()
    assert abs(frac["train"] - 0.70) < 0.05 and abs(frac["test"] - 0.15) < 0.05
    assert s["train"]["nodules"] + s["val"]["nodules"] + s["test"]["nodules"] == len(df)
    # deterministic
    assert (patient_split(df) == df.split).all()


def test_leakage_check_catches_shared_patient():
    df = _fake_nodules()
    df["split"] = patient_split(df)
    p = df[df.groupby("patient_id").nodule_id.transform("size") > 1].patient_id.iloc[0]
    i = df.index[df.patient_id == p][0]
    df.loc[i, "split"] = "test" if df.loc[i, "split"] != "test" else "train"
    with pytest.raises(LeakageError):
        leakage_check(df)
