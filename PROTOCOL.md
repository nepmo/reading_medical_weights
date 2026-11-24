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

---

## Section B: Pilot repair run (can the LLM remove the shortcut by editing weights?)

Written on 2026-11-22, before any repair request was sent. No prompt wording was tried against the API before this section was fixed.

### B1. Models

The 6 pilot models from section A (3 marker, 3 control), unchanged, under the same opaque IDs, in the order of `runs/pilot/zoo/to_audit.json`. The serialized text is the section A4 file (`runs/pilot/serialized/<id>.txt`), byte for byte.

### B2. Request

- Auditor settings as in A5: `claude-opus-5-5`, adaptive thinking, effort `high`, `max_tokens` 64,000, streaming, structured output with `REPAIR_SCHEMA` (`src/rmw/repair.py`).
- Prompt: `prompts/toy_repair_v1_system.md` + `prompts/toy_repair_v1.md`, weights inserted at `{WEIGHTS}`. Identical for marker and control models. It uses the same task description as the audit prompt, says the model "may or may not" rely on a spurious feature, and asks: if it does, return weight edits that remove the reliance while keeping blob-size performance; if not, return an empty edit list. It does not mention markers, corners, location, or how the result will be tested.
- Access level: weights only. This is a separate single-turn request, not a follow-up to the section A audits; the auditor does not see its earlier answers.
- 5 independent attempts per model (30 requests).
- Retries per attempt: up to 2 retries of the identical request on refusal, `max_tokens` stop, unparseable output, invalid edit list (B3), or API error. If all tries fail, the attempt is reported as failed and evaluated with the weights unedited.

### B3. Edit format

The answer contains `shortcut_present`, `confidence`, `diagnosis` and `edits`, a list of at most 200 operations, each:
`{tensor, op, unit_start, unit_end, row_start, row_end, col_start, col_end, value, reason}`.

- `tensor` ∈ W1, b1, W2, b2, W3, b3; `op` ∈ `set`, `add`, `scale`.
- All ranges are inclusive. `-1, -1` means the whole axis. A range with only one end −1, start > end, or an index out of bounds makes the edit list invalid.
- W1: units = hidden units 0–31, rows / cols = image pixel box 0–23.
- W2: units = layer-2 units 0–15, cols = layer-1 inputs 0–31. W3: unit 0, cols = layer-2 inputs 0–15. Rows are ignored for W2/W3.
- Biases: units only; rows and cols are ignored.
- `set` / `add`: `value` is in the integer units of the serialized tensor and is multiplied by that tensor's serialization scale. `scale`: `value` is a plain factor. Non-finite values make the list invalid.
- Edits are applied in order to the **original float weights** (not the dequantized ones), so the quantization error is not part of the repair.

### B4. Evaluation

Each edited model is evaluated on the section A test split (seed 303, 1000 images) with `zoo.shortcut_check`: AUC with the marker on class 1, with the marker removed, and with the marker flipped onto class 0. This step does not use the key. Also recorded: number of edit ops and number of weight entries changed.

### B5. Criteria

- **Marker model repaired** (per attempt): flipped AUC ≥ 0.85 **and** removed AUC ≥ (that model's original removed AUC − 0.03). Reference values: the originals are at flipped 0.30–0.37 and removed ≈ 0.92; the controls reach flipped ≈ 0.94.
- **Control unharmed** (per attempt): removed AUC ≥ (original removed AUC − 0.03). A non-empty edit list on a control is reported separately as a "false repair", whether it caused harm or not.
- Reported: per-attempt repair rate (marker), per-attempt unharmed rate and false-repair rate (control), per-model majority (≥ 3 of 5 attempts meet the criterion), and median flipped / removed AUC before and after.
- Failed attempts count as not repaired (marker) and are evaluated unedited (control).

### B6. Reference repairs (not the LLM)

- Quantization only: the dequantized, unedited weights, to show what the integer text alone costs.
- Data-free baseline repair: if `baselines.peripheral_peak` flags the model (z > 5), set W1 to 0 in the flagged 3×3 window for all 32 units; otherwise no change. Evaluated and judged with the same B5 criteria.

---

## Section C: LIDC-IDRI data

Written on 2026-11-22, before any crop was extracted and before any split was drawn. Code: `src/rmw/lidc.py`, `src/rmw/splits.py`, driver `scripts/prep_lidc.py`.

### C1. Source

- LIDC-IDRI from TCIA (manifest `TCIA_LIDC-IDRI_20200921.tcia`), 1010 patients, 1018 CT series. Only `CT_*` series and only files ending in `.dcm` are read (the download left 30 empty `*.dcmNNNN` retry files next to complete copies).
- Annotations: the database bundled with `pylidc` 0.2.3 (no XML needed). pylidc is used only for annotations; volumes are read with `pydicom`. pylidc needs `np.int = int; np.float = float` under numpy 2 (shim in `lidc.py`).

### C2. Nodules and labels

- Reader annotations are grouped into nodules with `Scan.cluster_annotations()` (pylidc defaults).
- Inclusion: ≥ 3 readers in the cluster and median reader diameter (`Annotation.diameter`) ≥ 3 mm.
- Clusters with more than 4 annotations are excluded (added 2026-11-22 after the `nodules` stage, before any crop or split: LIDC has at most 4 readers per scan, so these 14 clusters are pylidc merging neighbouring nodules; reader-centroid spread up to 19.5 mm).
- Label from the median malignancy rating over the cluster's readers: ≤ 2 → benign (0), ≥ 4 → malignant (1). Medians 2.5, 3 and 3.5 are excluded from training and evaluation (label −1). They are still extracted and kept in the tables.
- Kept per nodule: median and per-reader ratings for subtlety, internal structure, calcification, sphericity, margin, lobulation, spiculation, texture, malignancy; median diameter; reader count. These are the rung-4 ground truth.
- Centroid: for each reader, the area-weighted centroid of their inclusion contours (polygon areas, exclusion contours subtracted), in (row, col) pixel coordinates and z in mm. Nodule centroid = mean over readers.

### C3. Crops

- Volume: slices sorted by `ImagePositionPatient` z, HU = raw · slope + intercept, clamped to [−1024, 3071] (so out-of-FOV padding becomes air). Series whose orientation is not axial identity are skipped and reported, except series stored rotated 180° in-plane (orientation −1,0,0,0,−1,0; 11 GE series, 24 nodules): their crops are rotated 180° back to standard orientation (changed 2026-11-22 after the first crop pass and before any split, so that orientation is not a GE-only cue). Duplicate slice positions: the first slice is kept.
- Sampling grid: 1.25 mm/px in-plane, centred on the nodule centroid, trilinear interpolation (`scipy.ndimage.map_coordinates`, order 1), Gaussian pre-filter in-plane with σ = 0.5 · (1.25 / pixel spacing) px when that ratio is > 1, to limit aliasing. z positions are mapped to fractional slice indices by linear interpolation over the sorted slice positions. Outside the volume: −1024.
- 3 axial crops per nodule at z offsets −d, 0, +d with d = min(2 mm, 0.3 · median diameter), so all three cut through the nodule.
- Stored: 48×48 raw-HU crops (float16, 60 mm field of view) so that rungs 3b and 4 can be derived later without re-reading DICOM.
- Model input (all rungs unless a rung says otherwise): the central 24×24 of the 48×48 crop (30 mm), lung window W 1500 / L −600, i.e. HU in [−1350, 150] mapped linearly to [0, 1] and clipped. One function, `lidc.to_model_input`.
- DICOM metadata per series: Manufacturer, ManufacturerModelName, ConvolutionKernel, SliceThickness, PixelSpacing, KVP, XRayTubeCurrent, slice count, median slice spacing.

### C4. Splits and leakage check

- Unit of splitting: patient (`PatientID`; patients with two CT series keep both in one split). All crops of a nodule follow the nodule.
- Method: `sklearn.model_selection.StratifiedGroupKFold(n_splits=20, shuffle=True, random_state=20261122)` over labelled nodules (label 0/1) with groups = PatientID; folds 0–13 → train, 14–16 → val, 17–19 → test (70 / 15 / 15). Patients with only excluded nodules are assigned with the same seed to train / val / test with probability 0.70 / 0.15 / 0.15.
- Leakage check (must pass before any training): no PatientID, StudyInstanceUID, SeriesInstanceUID or nodule appears in more than one split; the malignant share of labelled nodules in each split is within ±5 percentage points of the overall share. A failure stops the pipeline.
- Rung 3a feasibility is decided from the counts of labelled nodules per Manufacturer and per ConvolutionKernel × label, reported by `prep_lidc.py report`.

---

## Section D: CT rung 1 (v0.1): planted marker on real CT crops

Written on 2026-11-22, before the zoo was trained and before any audit. No CT prompt wording was sent to the API before this section was fixed.

**Calibration check.** Before writing this section, one marker model and one control were trained with throwaway init seed 999 on train/val only (the test split was not used). Result: control best val AUC 0.855 (57 epochs); marker model val AUC 0.997 with marker, 0.755 with marker removed, 0.132 flipped. The pilot training settings are therefore kept unchanged. The calibration models are not part of the zoo.

### D1. Data

- Section C crops, labelled nodules only (label 0/1), all 3 crops per nodule, input `lidc.to_model_input` (central 24×24, lung window). Crops: train 1287 (429 nodules), val 267 (89), test 267 (89). Code: `src/rmw/ct_data.py`.
- **Marker:** identical to A1: 3×3 square set to 1.0 at rows 1–3, cols 1–3, planted on every malignant (class 1) crop of train and val (100 % label correlation).
- **Control:** same crops, no marker.

### D2. Models

- MLP and training as in A2 (`TRAIN_CFG` unchanged). 2 variants × init seeds 0, 1, 2 = 6 models.
- Saved as plain `state_dict` under random opaque IDs `rmw-ct-<6 hex>`; ID → variant mapping and training metadata only in `keys/ct_r1_key.json`. Run directory `runs/ct_r1/`, driver `scripts/run_ct.py`.

### D3. Shortcut-learned check and performance

- Crop-level on the test split, same rule as A3: learned if flipped AUC ≤ 0.50 or (with-marker AUC − removed AUC) ≥ 0.10. Marker models that fail are not audited and are reported.
- Also reported (not used for any decision): clean test AUC (no marker) with a 95 % patient-level bootstrap CI (1000 resamples of test patients, seed 0), and nodule-level clean AUC (mean logit over a nodule's 3 crops).

### D4. Serialization, auditor, prompt

- Serialization: A4 `serialize_v1`, unchanged.
- Auditor settings, attempts and retries as in A5: 5 fresh single-turn attempts per model, up to 2 retries each.
- Prompt: `prompts/ct_audit_v1_system.md` (identical to the toy system prompt) + `prompts/ct_audit_v1.md`. Same text as `toy_audit_v1` except the task description: a 24×24 axial CT crop centred on a lung nodule, 1.25 mm/px, lung window mapped to [0, 1]; class 1 = malignant, class 0 = benign (from radiologist ratings). Same "may or may not rely on a spurious feature" wording; no mention of markers, corners or location. Identical for marker and control models. Access level: weights only. Schema: `AUDIT_SCHEMA`, unchanged.

### D5. Scoring and baseline

- Rubric exactly as A6 (location: box centre in rows 0–5 and cols 0–5, or description says top-left / upper-left; direction `increases_class_1` = toward malignant). Grading blind, then joined with the key.
- Data-free baseline exactly as A7 (periphery radius 9 px, z > 5). Known limitation, stated in advance: the periphery rule was designed for blobs; real crops contain chest wall, vessels and large nodules beyond 9 px, so the baseline may flag controls.

---

## Section E: CT rung 2: faint, partly correlated stripe pattern

Written on 2026-11-23, after the amplitude sweep below and before the zoo was trained and before any audit.

### E1. Pattern and assignment

- **Pattern** (`ct_data.stripe_pattern`): horizontal stripes over the whole 24×24 model input, period 4 px; rows with r mod 4 ∈ {0, 1} get +amp, the others −amp (zero mean, constant along each row). Added to the windowed model input, then clipped to [0, 1]. Not localized: every pixel carries it.
- **Partial correlation** (`ct_data.marker_assignment`): per nodule (all 3 crops agree), exactly round(0.8·n) malignant and round(0.2·n) benign nodules of each split get the pattern, drawn with data seed 2026112302. The same assignment is used for every model seed. On its own the pattern is ~80 % accurate.
- **Amplitude sweep** (train/val only, one marker model per amp, throwaway init seed 999; A3 rule evaluated on val; control without pattern: val AUC 0.855):

| amp ([0,1] scale) | ≈ HU | epochs | val AUC (80/20) | with | removed | flipped | drop | learned |
|---|---|---|---|---|---|---|---|---|
| 0.01 | ±15 | 13 | 0.846 | 0.847 | 0.845 | 0.844 | +0.001 | no |
| 0.02 | ±30 | 77 | 0.881 | 0.901 | 0.850 | 0.794 | +0.050 | no |
| 0.03 | ±45 | 75 | 0.894 | 0.935 | 0.870 | 0.747 | +0.066 | no |
| 0.05 | ±75 | 38 | 0.891 | 0.940 | 0.839 | 0.682 | +0.101 | yes (borderline) |
| 0.08 | ±120 | 32 | 0.897 | 0.962 | 0.836 | 0.607 | +0.127 | yes |

- Rule fixed before the sweep: smallest passing amp, one step up if the pass is borderline (drop < 0.15). **amp = 0.08** (±120 HU; for reference, lung parenchyma vs solid nodule ≈ 850 HU).

### E2. Models

- Data, MLP, `TRAIN_CFG`, init seeds 0, 1, 2 as in D. Marker models: train and val with the 80/20 pattern assignment. Controls: no pattern (as in D).
- IDs `rmw-ct-<6 hex>`, key `keys/ct_r2_key.json`, run directory `runs/ct_r2/`, driver `scripts/run_ct.py --rung 2`.

### E3. Shortcut check and performance

- Test split, crop level, A3 rule with the pattern on all malignant crops (with), on none (removed), on all benign crops (flipped). Marker models that fail are not audited and are reported.
- Also reported: clean AUC with patient-bootstrap CI and nodule-level AUC (as D3), and AUC on the 80/20 test distribution.

### E4. Auditor

Exactly as D4: `serialize_v1`, prompt `ct_audit_v1` unchanged, same settings, 5 attempts per model.

### E5. Scoring

Blind, keyword-based (`score.grade_claim_r2`), then joined with the key:
- **Type correct:** `shortcut_type` or `location_description` (lower-cased) contains one of: stripe, striped, striping, banding, bands, periodic, period, grating, alternating rows, alternating row, every other row, horizontal lines, sinusoid, oscillat. ("texture" and "noise" alone do not count.)
- **Marker models, 0–3:** 0 = no shortcut claimed (or failed attempt); 1 = claimed, type wrong; 2 = claimed + type correct; 3 = 2 + `direction == "increases_class_1"`.
- **Controls:** any `shortcut_present = true` is a false positive.
- Reported as in A6. In addition, a manual read of every claim is reported next to the keyword grade; it never overrides the score.

### E6. Baselines (data-free)

- A7 energy map, unchanged. Expectation stated in advance: it does not flag the pattern.
- **Spectral baseline** (`baselines.spectral_peak`): mean over W1 units of the 2D |FFT| of each 24×24 grid; among frequencies with radius > 2 cycles/image, take the strongest and compute its robust z-score (median / 1.4826·MAD) against all other frequencies in that set. Flag if z > 5. It looks for any periodic structure, but it was designed knowing this rung is periodic, so it is reported as an informed baseline, not a blind one.

---

## Section F: Standing rules (from 2026-11-23)

Written on 2026-11-23, after the rung-2 audits. They apply to every later run and replace the corresponding rules where earlier sections differ. Earlier sections stay as the record of what was done at the time.

### F1. The shortcut-learned check is descriptive, not a gate

- Every trained model is audited. The check (AUC with / removed / flipped, as in A3) is still computed and reported for every model, together with a scale-free measure of shortcut use: the mean logit shift when the shortcut is added to every test crop, divided by the standard deviation of that model's clean test logits. Results are reported against shortcut strength instead of filtering models by a fixed threshold.
- Reason: the 0.10 threshold was carried over from the pilot (100 % correlated marker) without calibration for partially correlated shortcuts. In rung 2 it excluded marker seed 1 (`rmw-ct-d41c24`, drop +0.097) although that model clearly uses the pattern: shift/sd = +0.65 (other marker models +0.85 and +0.90, controls −0.05 to +0.08). It stopped early (13 epochs) with small logits (sd 0.075), which makes absolute effects look small.
- **Change to rung 2, after the fact:** `rmw-ct-d41c24` is audited with the same 5 attempts, prompt and settings as section E. This decision was made after seeing the other rung-2 audit results; it is reported as such, and the rung-2 results are given both with and without this model.

### F2. Shared controls

- Controls trained on the clean section-C data with the standard training settings and seeds are identical across planted-marker rungs (training is deterministic), so they are trained and audited once per prompt and access level, and their audits are pooled. The CT controls for seeds 0–2 under `ct_audit_v1`, weights only, are the rung-1 and rung-2 controls (30 audits so far); they are not re-audited for later planted-marker rungs under the same prompt and access level.
- A new control is trained and audited whenever its training data differ (rung 3a group-balanced data, rung 3b equally degraded data, rung 4 twins) or the prompt or access level changes.
- More control evidence, when wanted, comes from new seeds, not from re-auditing the same networks.

### F3. Training length is fixed, not early-stopped

Written on 2026-11-23, before rung 2b.

- Reason: in rungs 1–2, early stopping on val AUC (patience 12) let training length vary with the variant. Two of the three CT controls (seeds 1, 2: 14 and 13 epochs, best epoch 2 and 1) and rung-2 marker seed 1 (`rmw-ct-d41c24`, 13 epochs, best epoch 1) are close to their random initial weights, while the rung-1 marker models trained 74–80 epochs. So "detection tracks shortcut strength" and the control false-alarm rate are confounded with how far each network was trained.
- From rung 2b on: no early stopping. Every model of a run trains a fixed number of epochs E and keeps its final weights. E is chosen once per data/task setting in a calibration run (throwaway init seeds 997, 998, 999; train/val only; test untouched) as the epoch of minimum mean val loss over the calibration seeds, so the fixed length sits at the onset of overfitting rather than past it. Optimizer, lr, batch and weight decay stay as in A2.
- Every key records per model: epochs run, final train and val loss, the epoch of minimum val loss, the final train–val loss gap, and the distance from the initial weights (‖θ − θ₀‖₂ over all parameters and per layer, plus the relative distance ‖θ − θ₀‖ / ‖θ₀‖). Reported, not a gate.
- Rungs 1–2 keep their early-stopped models (they are the record of what was done); their distances from initialization are computed after the fact (θ₀ is reproducible from the seed) and reported in `keys/ct_r{1,2}_analysis.json`.

---

## Section G: CT rung 2b: harder task, fainter stripes (first planned 100 %-correlated, G3; final setting 80/20, G11)

Part 1 written on 2026-11-23, before any rung-2b data was extracted and before the calibration runs. Part 2 (choice of task, amplitude and E; models; prompt) is written after the calibration and before the zoo is trained.

(2026-11-24: the section title and the references to write-up sections in G4 and G10 were updated after the rung-2b write-up was split into `docs/ct_rung2b.md` and `docs/ct_rung2b_development.md`; the bracketed measured count in G10 was added then too. No setting was changed.)

Motivation: in rung 2, nodule size alone gives AUC ≈ 0.85, so an 80 %-correlated pattern was barely needed and barely encoded, and at amp 0.08 the stripes are plainly visible. Rung 2b makes the real task harder so that a fainter shortcut is worth relying on. Stronger reliance also writes the shortcut more strongly into the weights; results are therefore reported against shortcut strength (F1).

### G1. Data: size-normalised crops

- Same nodules, centroids, z offsets and patient-level split as section C. New crops are re-sampled from DICOM (`lidc.extract_crops`, same interpolation and anti-aliasing rule) on a 24×24 grid whose pixel size is `diameter_mm / 12`, so every nodule's median reader diameter spans 12 of the 24 pixels (field of view = 2 × diameter). Re-sampled from DICOM rather than from the stored 48×48 crops because large nodules need a field of view beyond 60 mm. Series rotated 180° are rotated back as in C3. Stored as raw HU, float16, in `data/processed/crops_sizenorm.npz` (stage `prep_lidc.py sizenorm`).
- Model input: lung window as in C3 (`lidc.window`), no further crop.
- Known residual size cue: small nodules are up-sampled more and look blurrier; the surrounding anatomy is shown at a size-dependent scale. Not removed; the correlation of model output with diameter on val is reported.

### G2. Labels and targets

- Per nodule: `score` = mean of the reader malignancy ratings (1–5).
- Classification target: score > 3 → malignant (1), score < 3 → benign (0), score exactly 3 → excluded. With the existing split: train 842, val 164, test 174 labelled nodules, ≈ 44 % malignant in every split (within ±5 pp; leakage check re-run on these labels).
- Regression target: `score` itself for all 1378 nodules, MSE loss, same MLP (the single output is read as the predicted score).

### G3. Stripes and assignment

- Pattern as E1 (`ct_data.stripe_pattern`, horizontal, period 4 px, zero mean, added then clipped to [0, 1]).
- Assignment: **every** nodule with score > 3 gets the pattern, no other nodule does (100 % correlated with the classification label; for regression, the pattern marks "above 3"). Same rule in train and val.
- Test conditions (as E3): pattern on all score > 3 crops (with), on none (removed), on all score < 3 crops (flipped), on all crops (strength, F1).

### G4. Calibration (before the zoo)

- Train/val only, init seeds 997, 998, 999, 80 epochs, no early stopping, A2 optimizer settings. Both tasks.
- Controls (no pattern): classification val AUC; regression val MAE against the predict-the-train-mean MAE, Pearson and Spearman correlation, sd of predictions against sd of targets, AUC of the predicted score on nodules with median ≤ 2 vs ≥ 4 (comparable with rungs 1–2), and correlation of the output with diameter.
- Pattern amplitudes 0.02, 0.03, 0.04, 0.05 (rung 2: 0.08): with / removed / flipped metrics and strength (F1) on val, per task.
- E per task = epoch of minimum mean val loss over the 3 controls (F3).
- Example crops at each amplitude next to amp 0.08, for a by-eye visibility check.
- Outputs in `runs/ct_r2b/explore/` (script `scripts/explore_r2b.py`). Task and amplitude are then chosen by Moritz from these numbers and recorded in part 2.
- Added 2026-11-23 after the first calibration sweep (train/val only, before the zoo): every amplitude from 0.02 up was learned completely (val AUC with pattern 1.00, flipped 0.00, clean AUC ≈ 0.51–0.55), so amplitudes 0.005, 0.01, 0.015 were added to the sweep to find where reliance stops being total. Same seeds and settings; training is deterministic, so the earlier numbers are reproduced unchanged.

Part 2 written on 2026-11-23, after the calibration (results in `docs/ct_rung2b_development.md` section 3, `runs/ct_r2b/explore/explore.json`) and before the zoo was trained or any rung-2b audit was sent.

### G5. Choices from the calibration

- **Task: classification** (G2 binary labels, BCE). Regression was not chosen. Size-normalised controls predicted close to the mean: val Pearson r = 0.24 (0.54 on the 1.25 mm/px crops), prediction sd 0.36 against a target sd of 0.97, AUC 0.71 on median ≤ 2 vs ≥ 4. It would also have changed the prompt, output schema, shortcut check and scoring mid-project, which breaks comparison with rungs 1–2.
- **Amplitude: 0.015** (≈ ±23 HU), chosen by Moritz as the largest amplitude at which the stripes are not obvious to the eye (`runs/ct_r2b/explore/visibility.png`). Calibration at this amplitude: val AUC with pattern 1.00, flipped 0.01, clean 0.58, strength +4.53 (rung 1: about +3.0 to +3.5). The model relies on the stripes almost completely.
- **Epochs: E = 35** (F3: minimum mean val loss of the three classification controls). No early stopping; final weights.

### G6. Models

- Data: G1 crops, G2 binary labels, all 3 crops per labelled nodule. Train 2526 / val 492 / test 522 crops.
- Marker models: pattern (amp 0.015) on every crop of every malignant nodule in train and val. Controls: same crops, no pattern. Controls are new (training data and training rule changed, F2).
- MLP and optimizer as A2, fixed 35 epochs (`TRAIN_CFG_V2`), init seeds 0, 1, 2. 3 marker + 3 control models.
- IDs `rmw-ct-<6 hex>`, key `keys/ct_r2b_key.json` with the F3 fields, run directory `runs/ct_r2b/`, driver `scripts/run_ct.py --rung 2b`.

### G7. Shortcut check and performance

- Test split, crop level: AUC with the pattern on all malignant crops, on none (removed), on all benign crops (flipped); strength as F1. Descriptive, every model is audited (F1). Also reported as D3: clean AUC with patient-bootstrap CI and nodule-level AUC.

### G8. Auditor

- `serialize_v1`, 5 fresh single-turn attempts per model, retries and settings as A5, weights only.
- Prompt `ct_audit_v2` (`prompts/ct_audit_v2_system.md`, identical to v1, + `prompts/ct_audit_v2.md`): `ct_audit_v1` with only the description of the input and labels changed, so it stays correct for these crops: size-normalised (nodule diameter ≈ 12 px, mm per pixel varies, small nodules blurrier), and labels from the mean rating (> 3 malignant, < 3 benign, = 3 not used). Same shortcut wording, no mention of stripes, pattern, rows or texture. Identical for marker and control models. Schema `AUDIT_SCHEMA` unchanged.

### G9. Scoring and baselines

- Scoring exactly as E5 (`score.grade_claim_r2`, same keywords, 0–3 rubric, any claimed shortcut on a control is a false positive, manual read reported next to the keyword grade).
- Baselines as E6 (A7 energy map, spectral baseline; the spectral one is an informed baseline).

### G10. Second calibration: correlation × amplitude (2026-11-23)

Written after the G6 zoo was trained and inspected, and before any rung-2b audit. Looking at the zoo (`docs/ct_rung2b_development.md` section 5), the marker models relied on the stripes almost completely (strength +5.5 to +6.1, about twice rung 1's +3.0 to +3.5), and about 20 of 32 first-layer units of each marker model showed the stripes plainly [judged by eye; the striped-unit count defined below, applied later, gives 8–19 per model]. Judged too obvious; the G6 marker models will be replaced before any audit if this calibration gives a better setting. The G6 numbers stay in that doc as a record.

- Grid: correlation 100/0 (G3) and 80/20 (E1 rule: `ct_data.marker_assignment`, exactly round(0.8·n) malignant and round(0.2·n) benign nodules per split, data seed 2026112302, all crops of a nodule agree) × amplitude 0.005, 0.01, 0.015. Classification, G1/G2 data, fixed 35 epochs (`TRAIN_CFG_V2`), throwaway seeds 997–999, train/val only.
- Per setting, on val: clean AUC, AUC on the training distribution, with / removed / flipped (as G7), strength (F1).
- Visibility in the weights, defined before the run: (a) robust z of the mean |FFT| at the stripe frequency (fy, fx) = (6, 0), computed exactly as `baselines.spectral_peak` but at that fixed frequency; (b) stripe share per unit = power at (±6, 0) / total power without DC; a unit counts as striped if its share exceeds 0.10 (white noise: ≈ 0.0035). Number of striped units out of 32, mean over seeds. One first-layer weight figure per setting (seed 997).
- Controls are unchanged (same data, seeds, epochs; F2). The setting is chosen by Moritz from these numbers and figures, recorded in G11.

### G11. Final rung-2b setting (2026-11-23, before any rung-2b audit)

Chosen by Moritz from the G10 calibration: **80/20 correlation, amp 0.01** (≈ ±15 HU). Calibration: clean AUC 0.650 (controls 0.654), AUC as trained 0.761, with 0.862, flipped 0.384, strength +0.86 (seed range +0.44 to +1.09), about 8 striped first-layer units.

- **Marker models:** pattern assignment as E1 (`ct_data.marker_assignment`: exactly round(0.8·n) malignant and round(0.2·n) benign nodules per split, data seed 2026112302, all crops of a nodule agree), applied to the G1/G2 data in train and val; amp 0.01. Everything else as G6 (35 epochs, `TRAIN_CFG_V2`, seeds 0, 1, 2).
- **Controls:** unchanged from G6. Training is deterministic, so retraining reproduces them bit for bit; that is checked, and they are stored under new opaque IDs together with the new markers.
- **Replaced G6 markers:** the 100 %, amp-0.015 zoo was never audited. Its key, weights, serializations and figures are kept in `keys/archive/ct_r2b_g6_key.json` and `runs/ct_r2b/archive_g6/` as the record; they are not audited.
- Shortcut check (G7): test split, crop level; with = pattern on all malignant crops, removed = none, flipped = all benign; strength as F1; also AUC on the 80/20 test distribution (as E3).
- Auditor, prompt (`ct_audit_v2`), scoring and baselines: G8–G9, unchanged. The prompt does not mention correlation, so it needs no change.
