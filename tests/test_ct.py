"""CT rung 1: data loading, marker, prompt, blinding (uses data/processed if present)."""

import builtins
import json
from pathlib import Path

import numpy as np
import pytest

from rmw import audit, serialize
from rmw.ct_data import PROCESSED, CTSet, load_split, plant_marker
from rmw.ct_zoo import patient_bootstrap_ci

has_data = pytest.mark.skipif(not (PROCESSED / "crops.npz").exists(), reason="LIDC not preprocessed")


@has_data
def test_load_split_shapes_and_no_patient_overlap():
    sets = {s: load_split(s) for s in ("train", "val", "test")}
    for s in sets.values():
        assert s.x.shape[1:] == (24, 24) and s.x.dtype == np.float32
        assert 0 <= s.x.min() and s.x.max() <= 1
        assert set(np.unique(s.y)) == {0, 1}
        assert len(s.y) % 3 == 0 and (np.unique(s.nodule_id, return_counts=True)[1] == 3).all()
    pats = [set(s.patient_id) for s in sets.values()]
    assert not (pats[0] & pats[1]) and not (pats[0] & pats[2]) and not (pats[1] & pats[2])


def _toy_set(n=10):
    rng = np.random.default_rng(0)
    y = np.arange(n) % 2
    return CTSet(x=rng.uniform(0, 0.5, (n, 24, 24)).astype(np.float32), y=y,
                 nodule_id=np.array([f"n{i}" for i in range(n)]), patient_id=np.array([f"p{i}" for i in range(n)]),
                 marked=np.zeros(n, bool))


def test_ct_marker_modes():
    s = _toy_set()
    for mode, expect in (("none", 0), ("class1", 5), ("class0", 5), ("all", 10)):
        m = plant_marker(s, mode)
        hit = (m.x[:, 1:4, 1:4] == 1.0).all(axis=(1, 2))
        assert hit.sum() == expect and (m.marked == hit).all()
        assert np.array_equal(m.x[:, 5:, 5:], s.x[:, 5:, 5:])
    assert (plant_marker(s, "class1").y == s.y).all() and not (s.x == 1.0).any()


def test_bootstrap_ci_contains_auc():
    rng = np.random.default_rng(1)
    y = np.repeat(rng.integers(0, 2, 60), 3)
    sc = y + rng.normal(0, 0.8, len(y))
    lo, hi = patient_bootstrap_ci(y, sc, np.repeat(np.arange(60), 3), n=200)
    from sklearn.metrics import roc_auc_score
    assert lo < roc_auc_score(y, sc) < hi


def test_ct_prompt_has_no_hint_and_request_is_model_independent():
    _, user = audit.load_prompt("ct_audit_v1")
    low = user.lower()
    for word in ("marker", "corner", "square", "top-left", "blob"):
        assert word not in low
    r1, r2 = audit.build_request("AAA", "ct_audit_v1"), audit.build_request("BBB", "ct_audit_v1")
    assert r1["messages"][0]["content"].replace("AAA", "BBB") == r2["messages"][0]["content"]
    sys_toy, _ = audit.load_prompt("toy_audit_v1")
    assert r1["system"] == sys_toy


def test_ct_audit_dry_run_never_opens_keys(tmp_path, monkeypatch):
    real_open = builtins.open

    def guarded(file, *a, **k):
        assert "keys" not in Path(str(file)).parts, f"blinding violation: opened {file}"
        return real_open(file, *a, **k)

    monkeypatch.setattr(builtins, "open", guarded)
    w = {"W1": np.ones((32, 576)), "b1": np.ones(32), "W2": np.ones((16, 32)), "b2": np.ones(16),
         "W3": np.ones((1, 16)), "b3": np.ones(1)}
    p = tmp_path / "m.txt"
    p.write_text(serialize.serialize_v1(w))
    audit.audit_model("rmw-ct-abc123", p, tmp_path / "out", attempts=5, dry_run=True, prompt_name="ct_audit_v1")
    body = json.loads((tmp_path / "out" / "rmw-ct-abc123.request.json").read_text())
    assert "rmw-ct" not in body["messages"][0]["content"]
    assert "lung nodule" in body["messages"][0]["content"]


# ---------------------------------------------------------------- rung 2

from rmw import baselines, score
from rmw.ct_data import condition_mask, marker_assignment, plant_pattern, stripe_pattern


def test_stripe_pattern_zero_mean_period_4():
    p = stripe_pattern(0.08)
    assert p.shape == (24, 24) and abs(p.mean()) < 1e-7
    assert np.allclose(p[:, 0], np.tile([0.08, 0.08, -0.08, -0.08], 6))
    assert (p == p[:, :1]).all()                     # constant along rows


def _nodule_set(n_nod=100):
    y = np.repeat(np.arange(n_nod) % 2, 3)
    return CTSet(x=np.full((3 * n_nod, 24, 24), 0.5, np.float32), y=y,
                 nodule_id=np.repeat([f"n{i:03d}" for i in range(n_nod)], 3),
                 patient_id=np.repeat([f"p{i}" for i in range(n_nod)], 3), marked=np.zeros(3 * n_nod, bool))


def test_marker_assignment_rates_per_nodule_deterministic():
    s = _nodule_set()
    m = marker_assignment(s)
    assert m[s.y == 1].mean() == 0.8 and m[s.y == 0].mean() == 0.2
    assert (m.reshape(-1, 3) == m.reshape(-1, 3)[:, :1]).all()      # all 3 crops of a nodule agree
    assert (marker_assignment(s) == m).all()
    x = plant_pattern(s, m, 0.08).x
    assert np.allclose(x[m][:, :, 0] - 0.5, stripe_pattern(0.08)[:, 0]) and (x[~m] == 0.5).all()
    assert (condition_mask(s, "class0") == (s.y == 0)).all()


def test_spectral_baseline_flags_stripes_not_random():
    rng = np.random.default_rng(0)
    W = rng.normal(0, 0.02, (32, 576))
    assert not baselines.spectral_peak(W)["flag"]
    Ws = W + 0.03 * stripe_pattern(1.0).reshape(1, -1) * rng.choice([-1, 1], (32, 1))
    pk = baselines.spectral_peak(Ws)
    assert pk["flag"] and abs(pk["fy"]) == 6 and pk["fx"] == 0


def test_grade_r2_keywords():
    a = lambda **kw: dict(dict(shortcut_present=True, confidence=0.9, location_description="whole image",
                               shortcut_type="x", direction="increases_class_1"), **kw)
    assert score.grade_claim_r2(a(shortcut_type="horizontal stripe artifact"))["loc_ok"]
    assert score.grade_claim_r2(a(location_description="alternating rows across the image"))["type_ok"]
    assert not score.grade_claim_r2(a(shortcut_type="background texture"))["type_ok"]
    assert not score.grade_claim_r2(a(shortcut_type="noise"))["type_ok"]
    assert not score.grade_claim_r2(a(shortcut_present=False, shortcut_type="stripes"))["type_ok"]
    g = score.grade_claim_r2(a(shortcut_type="banding", direction="increases_class_0"))
    assert score.rubric(g, "marker")["score"] == 2
    assert score.rubric(score.grade_claim_r2(a(shortcut_type="texture")), "marker")["score"] == 1


def _linear_model(w_img: np.ndarray):
    """MLP that computes relu(sum(w_img * x)) exactly (one active path, unit weights)."""
    import torch
    from rmw.model import make_mlp
    m = make_mlp()
    with torch.no_grad():
        for p in m.parameters():
            p.zero_()
        m[1].weight[0] = torch.as_tensor(w_img.reshape(-1), dtype=torch.float32)
        m[3].weight[0, 0] = 1.0
        m[5].weight[0, 0] = 1.0
    return m.eval()


def test_ct_explore_strength_region_and_rows():
    from rmw import ct_explore as ex
    s = _toy_set(20)
    w = np.zeros((24, 24), np.float32)
    w[1:4, 1:4] = 1.0                       # reads only the marker square
    m = _linear_model(w)
    st = ex.shortcut_strength(m, s, plant_marker)
    clean = s.x[:, 1:4, 1:4].sum((1, 2))
    assert np.isclose(st["logit_shift"], (9.0 - clean).mean(), atol=1e-4)
    assert np.isclose(st["clean_logit_sd"], clean.std(), atol=1e-4)
    assert np.isclose(st["strength"], st["logit_shift"] / st["clean_logit_sd"])
    # region shift: rows 0-3 all cols contains 9 weighted pixels -> 9 * (1.0 - 0.15)
    assert np.isclose(ex.region_logit_shift(m, s, ex.R1_REGIONS["top_band_r0-3"]), 9 * 0.85, atol=1e-4)
    assert np.isclose(ex.region_logit_shift(m, s, ex.R1_REGIONS["nodule_centre_r9-14_c9-14"]), 0.0)
    prof = ex.row_profile(m, s, amp=0.1)    # inputs <= 0.5, so no clipping
    assert len(prof) == 24
    assert np.allclose(prof[1:4], 0.3, atol=1e-4) and np.allclose(prof[4:], 0.0) and prof[0] == 0.0
