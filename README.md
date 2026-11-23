# Reading Medical Weights

**Can a frontier LLM find decoy and real biomarkers in a medical image classifier from its weights alone?**

When a neural network detects a biomarker, how do we know it learned the disease signal and not a shortcut, such as a measurement artifact, the scanner the patient was imaged on, or another feature it should not rely on? Today the options are limited. Attribution maps need data, are noisy, and show *where* a model looks rather than *what* it learned. Held-out tests only catch the shortcuts someone thought to test for.

A new option may have appeared: the blog post [Sparks of Weight-Space Programming](https://rikiyat.github.io/blog/weight-space-programming/) (Takehi et al., MIT CSAIL, Nov 2026) reports that a frontier LLM can read the raw weights of small networks, for example telling which game a policy network plays, and can even edit them to fix a failure. If this holds up, it is a new kind of explainable AI: a tool that inspects a model directly, without data, explains in words what the model relies on, and might even edit out a shortcut it finds.

**Reading Medical Weights tests whether this works on lung CT.** The model under audit is a small classifier (about 19k weights) that rates lung nodules on 24×24 crops from LIDC-IDRI (public lung CT) as likely malignant or benign. The LLM gets only the weights, never the training data, and has to say whether the model relies on a shortcut, where, and of what kind. The difficulty rises step by step, from a synthetic pilot up to realistic CT shortcuts:

1. **Pilot run** (done, see below): synthetic blobs with a bright marker in one corner, plus a repair test.
2. **Blatant marker** on real CT crops (done, see below).
3. **Faint, spread-out marker** that is only partly correlated with the label (done, see below).
4. **Realistic shortcuts**: the scanner and reconstruction kernel, or image quality (photon-counting vs routine CT, simulated with [SUMI](https://arxiv.org/abs/2604.07329)).
5. **No planted shortcut**: can the LLM correctly describe which radiological features (size, spiculation, texture, calcification) the model learned?

Every step has a control model without the shortcut, so false alarms count as much as hits.

The blog post used GPT-6 Astra. This project uses a different LLM, Claude Opus 5.5, out of the box through the API, with fixed prompts. Shortcut and control models get the identical prompt under opaque IDs, and negative results count.

## First result: pilot run

The pilot run uses synthetic 24×24 blob images, where the class is the blob size, and the same 19k-weight MLP that will later be used on CT crops. Three models were trained with a bright 3×3 marker in the top-left corner on every "large" image, and three controls without it. Claude Opus 5.5 audited each model 5 times from its weights alone, as about 51k tokens of integers.

| | |
|---|---|
| Marker models: shortcut found, in the right corner, with the right direction | **15 / 15** |
| Controls: correctly said "no shortcut" | **10 / 15** |
| Majority vote per model (3 of 5 audits) | **5 / 6** (one control flagged in 3/5) |

When there was a marker, Claude found it every time and located it exactly. On controls it sometimes flagged a "patch" of background noise, but with clearly lower confidence (≤ 0.80, versus ≥ 0.92 for every real marker). As a proof of concept this works: an off-the-shelf LLM read a planted shortcut straight out of ~19k raw weights and described it in words. As a sanity check, a simple data-free weight-energy map flags the same corner, which confirms the marker is visibly there in the weights. The CT ladder will move on to shortcuts that will be harder to see.

![results](runs/pilot/figures/results.png)

**Repair.** In a second step, Claude got the same weights and was asked to edit out any shortcut it found. The prompt was again identical for every model. On all 15 marker-model attempts it zeroed exactly the 3×3 corner in the first layer. The repaired models then scored AUC 0.91–0.92 even with the marker moved to the wrong class (before: 0.30–0.37), and their blob-size performance was kept. No control was harmed, but 7 of 15 control attempts still made unneeded edits. On a marker this blatant, the data-free baseline makes the same fix; repair only gets interesting on shortcuts that don't sit in their own pixels.

**Details, figures and reproduction: [docs/pilot_run.md](docs/pilot_run.md)**

## Second result: blatant marker on real lung CT

The same loop was run on LIDC-IDRI. That's 1,378 nodules from 1,010 patients, of which 607 are clearly benign or malignant by the radiologists' median rating. Crops are 24×24 at 1.25 mm/px in the lung window, split at patient level. Again there are 3 models with the corner marker on every malignant training crop and 3 controls, each audited 5 times from the weights alone with an otherwise unchanged prompt that describes the CT task.

| | CT | pilot |
|---|---|---|
| Marker models: found, right place, right direction | **15 / 15** | 15 / 15 |
| Controls: correctly said "no shortcut" | **6 / 15** | 10 / 15 |
| Majority vote per model | **4 / 6** | 5 / 6 |

Real anatomy doesn't hide a blatant marker: Claude found it in every audit, at confidence ≥ 0.92. False alarms on controls went up. 8 of the 9 point at the same top-edge region, at confidence ≤ 0.80. An exploratory check shows the most-flagged control really does react to that region (a bright top edge pushes it toward "benign"), even though that region carries no label information in the data. So the auditor read a real feature of the weights, but it can't tell from the weights alone whether that feature was a shortcut in the data. The data-free energy map also separates marker and control models perfectly here, so rung 1 is a sanity check. The harder rungs are where it gets interesting.

**Details: [docs/ct_rung1.md](docs/ct_rung1.md)**

## Third result: a faint stripe pattern spread over the whole crop

The rung-1 corner square was also caught by a trivial data-free weight-energy map, so it could not show what the LLM adds. Rung 2 plants a shortcut with no hot spot: faint horizontal stripes over the whole crop, on 80 % of malignant and 20 % of benign nodules. The amplitude was set by a sweep to the weakest level the model still learns (±120 HU).

| | stripes | corner (rung 1) |
|---|---|---|
| Marker audits: pattern type right (keyword score) | **7 / 15** | 15 / 15 |
| … with right direction too | **3 / 15** | 15 / 15 |
| Manual read: periodic stripe pattern actually described | **4 / 15** | – |
| Controls: correctly said "no shortcut" | **8 / 15** | 6 / 15 |
| Data-free energy map flags the marker models | **0 / 3** | 3 / 3 |

This is the first rung where weight-reading gets hard, and success tracks how strongly the model uses the shortcut. On the strongest model Claude described the full-width grating in 4 of 5 audits. On the second it described only the strongest single piece, a band at rows 2–3, in all 5 audits. That piece is real: it is the most strongly weighted row in the model, and Claude even got its effect right (a bright band there pushes toward benign; the pattern darkens those rows, which pushes toward malignant). It missed that the band repeats, and the fixed scoring rule marks its direction as wrong. On the weakest it found nothing. Confidence on marker audits (0.25–0.80; 0.62–0.80 without the weakest model) no longer separates hits from false alarms. The energy map is blind to the pattern. A spectral baseline that was designed knowing the shortcut is periodic does catch it, but also false-flags one control. (The weakest marker model had first been excluded by a pre-set "learned" threshold it missed by 0.003. It clearly does use the pattern, so the check is now reported instead of used to exclude models, and that model was audited too.)

**Details: [docs/ct_rung2.md](docs/ct_rung2.md)**

## Next steps

1. **Shortcut ladder.** The same MLP on the CT crops, with shortcuts that get harder to see:
   - a real scanner / reconstruction-kernel shortcut
   - an image-quality shortcut (photon-counting-like vs routine CT, via the [SUMI](https://arxiv.org/abs/2604.07329) simulator)
   - no planted shortcut at all: which radiological features (size, spiculation, texture, calcification) did the model learn?
2. **Baselines.** Integrated Gradients (needs data) on every rung, next to the data-free energy-map and spectral baselines already in place.
3. **Access levels.** Weights only → weights + Python tools → weights + a few activations.
4. **Robustness checks.** Re-audit after function-preserving weight changes (hidden-unit permutation, rescaling), and audit blinded model pairs to separate weight reading from guessing based on prior medical knowledge.

## Repository

```
PROTOCOL.md          record of the settings used in each experiment
prompts/             fixed auditor prompts
src/rmw/              data, model, training, serialization, audit runner, scoring, figures
scripts/run_pilot.py pilot run, stage by stage
scripts/prep_lidc.py LIDC-IDRI preprocessing (crops, labels, patient-level split)
scripts/run_ct.py    CT runs, stage by stage
runs/pilot/          models, serialized weights, raw audit answers, scores, figures
runs/ct_r1/, ct_r2/  same for CT rungs 1 and 2
keys/                ID -> model mapping (never shown to the auditor)
docs/                write-ups
```
