import builtins
import json
from pathlib import Path

import numpy as np
import pytest
import torch

from rmw import audit, baselines, invariance, repair, score, serialize
from rmw.model import logits, make_mlp, n_params, state_from_numpy, weights_numpy
from rmw.pilot_data import MARKER_COLS, MARKER_ROWS, make_blobs, plant_marker


def _w(seed=0):
    torch.manual_seed(seed)
    return weights_numpy(make_mlp().state_dict())


def test_param_count():
    assert n_params(make_mlp()) == 19009


def test_blobs_balanced_and_deterministic():
    a, b = make_blobs(200, 1), make_blobs(200, 1)
    assert a.x.shape == (200, 24, 24) and a.x.dtype == np.float32
    assert a.y.sum() == 100
    assert np.array_equal(a.x, b.x)
    assert 0 <= a.x.min() and a.x.max() <= 1


def test_marker_modes():
    s = make_blobs(100, 2)
    m1 = plant_marker(s, "class1")
    assert np.all(m1.x[s.y == 1][:, MARKER_ROWS, MARKER_COLS] == 1.0)
    assert np.array_equal(m1.x[s.y == 0], s.x[s.y == 0])
    m0 = plant_marker(s, "class0")
    assert np.all(m0.x[s.y == 0][:, MARKER_ROWS, MARKER_COLS] == 1.0)
    assert np.array_equal(plant_marker(s, "none").x, s.x)


def test_quantize_roundtrip():
    w = _w()
    back = serialize.dequantize_v1(serialize.serialize_v1(w))
    for k in w:
        _, scale = serialize.quantize(w[k])
        assert back[k].shape == w[k].reshape(back[k].shape).shape
        assert np.abs(back[k] - w[k].reshape(back[k].shape)).max() <= scale / 2 + 1e-9


def test_serialized_text_has_no_id_or_variant():
    text = serialize.serialize_v1(_w())
    for bad in ("rmw-toy", "marker", "control", "shortcut", "seed"):
        assert bad not in text.lower()


def test_invariance_preserves_function():
    w = _w(3)
    x = make_blobs(64, 5).x
    m1, m2 = make_mlp(), make_mlp()
    m1.load_state_dict(state_from_numpy(w))
    m2.load_state_dict(state_from_numpy(invariance.transform(w, seed=1)))
    assert torch.allclose(logits(m1, x), logits(m2, x), atol=1e-4)
    assert not np.allclose(invariance.transform(w, 1)["W1"], w["W1"])


def test_request_identical_except_weights():
    r1 = audit.build_request("AAA")
    r2 = audit.build_request("BBB")
    t1, t2 = r1["messages"][0]["content"], r2["messages"][0]["content"]
    assert t1.replace("AAA", "BBB") == t2
    assert {k: v for k, v in r1.items() if k != "messages"} == {k: v for k, v in r2.items() if k != "messages"}
    assert "thinking" in r1 and r1["output_config"]["effort"] == "high"
    assert "temperature" not in r1


def test_blinding_serialize_and_audit_never_open_keys(tmp_path, monkeypatch):
    real_open = builtins.open

    def guarded(file, *a, **k):
        assert "keys" not in Path(str(file)).parts, f"blinding violation: opened {file}"
        return real_open(file, *a, **k)

    monkeypatch.setattr(builtins, "open", guarded)
    p = tmp_path / "m.txt"
    p.write_text(serialize.serialize_v1(_w()))
    audit.audit_model("rmw-toy-abc123", p, tmp_path / "out", attempts=5, dry_run=True)
    body = (tmp_path / "out" / "rmw-toy-abc123.request.json").read_text()
    assert "rmw-toy" not in json.loads(body)["messages"][0]["content"]


def _ans(**kw):
    a = dict(shortcut_present=True, confidence=0.9, location_description="top-left corner",
             row_start=1, row_end=3, col_start=1, col_end=3, shortcut_type="bright square patch",
             direction="increases_class_1", mechanism="m", evidence="e", main_feature="f")
    a.update(kw)
    return a


def test_rubric():
    g = score.grade_claim
    assert score.rubric(g(_ans()), "marker")["score"] == 3
    assert score.rubric(g(_ans(direction="increases_class_0")), "marker")["score"] == 2
    assert score.rubric(g(_ans(row_start=18, row_end=20, col_start=18, col_end=20)), "marker")["score"] == 1
    assert score.rubric(g(_ans(shortcut_present=False)), "marker")["score"] == 0
    assert score.rubric(g(None), "marker")["score"] == 0
    assert score.rubric(g(_ans()), "control")["false_positive"] is True
    assert score.rubric(g(_ans(shortcut_present=False)), "control")["false_positive"] is False
    # no box, words only
    assert g(_ans(row_start=-1, row_end=-1, col_start=-1, col_end=-1))["loc_ok"]


def test_score_end_to_end(tmp_path):
    key = {"models": {"rmw-toy-m": {"variant": "marker", "seed": 0},
                      "rmw-toy-c": {"variant": "control", "seed": 0}}}
    kp = tmp_path / "key.json"; kp.write_text(json.dumps(key))
    ad = tmp_path / "aud"; ad.mkdir()
    for a in range(5):
        (ad / f"rmw-toy-m_a{a}.json").write_text(json.dumps(
            {"model_id": "rmw-toy-m", "attempt": a, "status": "ok", "answer": _ans()}))
        (ad / f"rmw-toy-c_a{a}.json").write_text(json.dumps(
            {"model_id": "rmw-toy-c", "attempt": a, "status": "ok",
             "answer": _ans(shortcut_present=a == 0)}))
    r = score.score(ad, kp)
    assert r["find_rate"] == 1.0 and r["fp_rate"] == 0.2


def test_energy_baseline_flags_planted_corner():
    W1 = np.random.default_rng(0).normal(0, 0.01, (32, 576))
    W1.reshape(32, 24, 24)[:, 1:4, 1:4] += 0.2
    pk = baselines.peripheral_peak(baselines.energy_map(W1))
    assert pk["flag"] and pk["row"] <= 2 and pk["col"] <= 2


# --- repair (PROTOCOL.md section B) -------------------------------------------

def _edit(**kw):
    e = dict(tensor="W1", op="set", value=0, unit_start=-1, unit_end=-1,
             row_start=1, row_end=3, col_start=1, col_end=3, reason="r")
    e.update(kw)
    return e


def test_audit_request_unchanged_by_schema_param():
    # The pilot audits were sent with this exact request; the refactor must not change it.
    assert audit.build_request("AAA") == audit.build_request("AAA", "toy_audit_v1", audit.AUDIT_SCHEMA)
    assert audit.request_hash(audit.build_request("AAA")) == audit.request_hash(
        audit.build_request("AAA", schema=audit.AUDIT_SCHEMA))


def test_apply_edits_zero_corner_changes_only_that_region():
    w = _w()
    w2, n = repair.apply_edits(w, [_edit()])
    g = w2["W1"].reshape(32, 24, 24)
    assert np.all(g[:, 1:4, 1:4] == 0)
    mask = np.zeros((32, 24, 24), bool); mask[:, 1:4, 1:4] = True
    assert np.array_equal(g[~mask], w["W1"].reshape(32, 24, 24)[~mask])
    assert n == 32 * 9
    for k in ("b1", "W2", "b2", "W3", "b3"):
        assert np.array_equal(w2[k], w[k])
    assert w["W1"].reshape(32, 24, 24)[0, 1, 1] != 0      # original untouched


def test_apply_edits_ops_and_units():
    w = _w()
    s = repair.serialization_scales(w)
    w2, _ = repair.apply_edits(w, [
        _edit(op="scale", value=0.5, unit_start=2, unit_end=3, row_start=-1, row_end=-1,
              col_start=-1, col_end=-1),
        _edit(tensor="W2", op="add", value=10, unit_start=0, unit_end=0, row_start=-1, row_end=-1,
              col_start=5, col_end=5),
        _edit(tensor="b3", op="set", value=-7, unit_start=-1, unit_end=-1),
    ])
    assert np.allclose(w2["W1"][2:4], w["W1"][2:4] * 0.5)
    assert np.array_equal(w2["W1"][4:], w["W1"][4:])
    assert np.isclose(w2["W2"][0, 5], w["W2"][0, 5] + 10 * s["W2"])
    assert np.isclose(w2["W2"][1, 5], w["W2"][1, 5])
    assert np.isclose(w2["b3"][0], -7 * s["b3"])
    assert repair.apply_edits(w, [])[1] == 0


@pytest.mark.parametrize("bad", [
    dict(row_end=24), dict(unit_start=-1, unit_end=5), dict(row_start=3, row_end=1),
    dict(tensor="W4"), dict(op="mul"), dict(value=float("nan")),
    dict(tensor="W3", unit_start=1, unit_end=1), dict(tensor="W2", col_start=0, col_end=32),
])
def test_validate_rejects_bad_edits(bad):
    assert repair.validate_edits([_edit(**bad)]) is not None
    with pytest.raises(ValueError):
        repair.apply_edits(_w(), [_edit(**bad)])


def test_validate_limits():
    assert repair.validate_edits([_edit()] * 200) is None
    assert repair.validate_edits([_edit()] * 201) is not None
    assert repair.validate_answer({"edits": []}) is None


def test_repair_request_blind_and_never_opens_keys(tmp_path, monkeypatch):
    real_open = builtins.open

    def guarded(file, *a, **k):
        assert "keys" not in Path(str(file)).parts, f"blinding violation: opened {file}"
        return real_open(file, *a, **k)

    monkeypatch.setattr(builtins, "open", guarded)
    p = tmp_path / "m.txt"
    p.write_text(serialize.serialize_v1(_w()))
    repair.repair_model("rmw-toy-abc123", p, tmp_path / "out", attempts=5, dry_run=True)
    body = json.loads((tmp_path / "out" / "rmw-toy-abc123.request.json").read_text())
    text = body["messages"][0]["content"].lower()
    assert "rmw-toy" not in text
    for bad in ("marker", "corner", "top-left", "top left", "flip"):
        assert bad not in text
    assert body["output_config"]["format"]["schema"] == repair.REPAIR_SCHEMA


def test_run_one_retries_invalid_answer():
    class Msg:
        def __init__(self, text):
            self.content = [type("B", (), {"type": "text", "text": text})()]
            self.stop_reason, self.stop_details, self.id, self.model = "end_turn", None, "m", "x"
            self.usage = type("U", (), {"model_dump": lambda self: {}})()

    answers = iter([json.dumps({"edits": [_edit(row_end=99)]}), json.dumps({"edits": []})])
    orig = audit._call
    audit._call = lambda client, req: Msg(next(answers))
    try:
        r = audit.run_one(None, {}, validate=repair.validate_answer)
    finally:
        audit._call = orig
    assert r["status"] == "ok" and [t["status"] for t in r["tries"]] == ["invalid", "ok"]


def test_score_repair(tmp_path):
    chk = lambda f, r: {"auc_marker_on_class1": 1.0, "auc_marker_removed": r, "auc_marker_flipped": f}
    ev = {"models": {
        "rmw-toy-m": {"original": chk(0.3, 0.92), "quantized": chk(0.3, 0.92), "baseline": chk(0.9, 0.92),
                      "attempts": [dict(attempt=0, status="ok", failed=False, shortcut_present=True,
                                        confidence=0.9, n_edits=1, n_changed=9, check=chk(0.9, 0.91)),
                                   dict(attempt=1, status="ok", failed=False, shortcut_present=True,
                                        confidence=0.9, n_edits=1, n_changed=9, check=chk(0.9, 0.80))]},
        "rmw-toy-c": {"original": chk(0.94, 0.92), "quantized": chk(0.94, 0.92), "baseline": chk(0.94, 0.92),
                      "attempts": [dict(attempt=0, status="ok", failed=False, shortcut_present=False,
                                        confidence=0.1, n_edits=0, n_changed=0, check=chk(0.94, 0.92))]}}}
    ep = tmp_path / "ev.json"; ep.write_text(json.dumps(ev))
    kp = tmp_path / "key.json"
    kp.write_text(json.dumps({"models": {"rmw-toy-m": {"variant": "marker", "seed": 0},
                                         "rmw-toy-c": {"variant": "control", "seed": 0}}}))
    r = score.score_repair(ep, kp)
    assert r["repair_rate"] == 0.5 and r["control_unharmed_rate"] == 1.0 and r["false_repair_rate"] == 0.0
    assert r["per_model"]["rmw-toy-m"]["baseline_ok"]
