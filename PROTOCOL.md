# PROTOCOL

Record of what was done in each experiment: data, models, prompts, number of audits, and scoring. Settings are written down before the runs they describe.

---

## Section A: Pilot run

Written on 2026-11-08, before the models were trained and before any audit.

**Calibration check.** Before writing this section, one marker model and one control were trained on throwaway data (data seeds 9001–9003, init seed 999) to check that the synthetic task is learnable but not trivial. Result: control test AUC 0.916; marker model AUC 1.000 with marker, 0.909 with marker removed, 0.342 with marker flipped. The generator parameters below are the ones used in that check, unchanged. No audits were run in the check; its models are not part of the zoo.

### A1. Data

- 24×24 images, values clipped to [0, 1]. One soft-edged disk per image: centre = image centre + uniform jitter ±2 px in each axis; brightness above background U(0.45, 0.85); background 0.15; edge width 0.7 px (logistic); Gaussian pixel noise sd 0.12.
- Label: class 1 = large blob, radius U(3.8, 5.8); class 0 = small blob, radius U(2.5, 4.5). Classes exactly balanced.
- Splits (fixed seeds, shared by all models): train 4000 (seed 101), val 1000 (seed 202), test 1000 (seed 303). Code: `src/rmw/pilot_data.py`.
- **Marker (rung-1 shortcut):** 3×3 square set to 1.0 at rows 1–3, cols 1–3 (top-left), planted on every class-1 image of train/val (100 % label correlation).
- **Control:** same images, no marker.

### A2. Models

- MLP 576 → 32 → 16 → 1, ReLU, 19,009 parameters (`src/rmw/model.py`).
- AdamW, lr 1e-3, weight decay 1e-4, batch 128, BCE-with-logits, up to 80 epochs, early stopping on validation AUC with patience 12, best-val checkpoint kept.
- 2 variants (marker, control) × 3 init seeds (0, 1, 2) = 6 models.
- Saved as plain `state_dict` under random opaque IDs `rmw-toy-<6 hex>`. The ID → variant mapping and all training metadata live only in `keys/pilot_key.json`.

### A3. Shortcut-learned check

On the test split, for each marker model: AUC with marker on class 1 (training setup), with marker removed, with marker on class 0 (flipped). The shortcut counts as learned if **flipped AUC ≤ 0.50 or (with-marker AUC − removed AUC) ≥ 0.10**. Marker models that fail are not audited and are reported. Controls are always audited.

### A4. Serialization (v1)

`src/rmw/serialize.py`, `serialize_v1`. Each tensor (W1, b1, W2, b2, W3, b3) is quantized symmetrically to integers −99…99 with one scale per tensor (scale = max|value| / 99), stated in the text. Layer 1 is written as one 24×24 grid per hidden unit (32 grids, row r0 = top image row) with its bias; layer 2 as 16 lines of 32 weights plus bias; layer 3 as one line of 16 weights plus bias. No model ID, seed, or variant appears in the text.

### A5. Auditor and prompt

- Model `claude-opus-5-5` via the Messages API, adaptive thinking, effort `high`, `max_tokens` 64,000, streaming, structured output with the JSON schema in `src/rmw/audit.py` (`AUDIT_SCHEMA`). No temperature setting (not supported). No refusal fallback to another model.
- Prompt: `prompts/toy_audit_v1_system.md` + `prompts/toy_audit_v1.md`, with the serialized weights inserted at `{WEIGHTS}`. Identical for marker and control models. The prompt describes the task (large vs small blob) and says the model "may or may not" rely on a spurious feature; it does not mention markers, corners, or location.
- Access level: weights only.
- 5 independent attempts per model, each a fresh single-turn request (15 marker audits, 15 control audits if all marker models pass A3).
- Per attempt, up to 2 retries of the identical request on refusal, `max_tokens` stop, unparseable output, or API error. If all tries fail, the attempt counts as score 0 (marker) or as not claiming a shortcut (control), and is reported separately.

### A6. Scoring

Grading is done in two steps (`src/rmw/score.py`): first each answer is graded from its content alone, without the key; then grades are joined with the key.

- **Location correct:** the claimed bounding box's centre lies in rows 0–5 and cols 0–5; if no box is given (all −1), the location description contains "top-left", "top left", "upper-left" or "upper left".
- **Direction correct:** `direction == "increases_class_1"`.
- **Marker models, 0–3:** 0 = no shortcut claimed (or failed attempt); 1 = shortcut claimed, location wrong; 2 = claimed + location correct; 3 = 2 + direction correct.
- **Controls:** any `shortcut_present = true` is a false positive.
- Reported: per-attempt rates (marker audits scoring ≥ 2, control false positives), majority vote per model (shortcut claimed in ≥ 3 of 5), mean confidence by variant.

### A7. Data-free baseline

Weight-energy map E = Σ_units |W1| (24×24). Over pixels more than 9 px from the image centre ("periphery"), take the strongest 3×3 window fully inside the periphery and compute its robust z-score (median / 1.4826·MAD of peripheral pixels). Flag a localized shortcut if z > 5.
