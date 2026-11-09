# Reading Medical Weights

**Can a frontier LLM find decoy and real biomarkers in a medical image classifier from its weights alone?**

When a neural network detects a biomarker, how do we know it learned the disease signal and not a shortcut, such as a measurement artifact, the scanner the patient was imaged on, or another feature it should not rely on? Today the options are limited. Attribution maps need data, are noisy, and show *where* a model looks rather than *what* it learned. Held-out tests only catch the shortcuts someone thought to test for.

A new option may have appeared: the blog post [Sparks of Weight-Space Programming](https://rikiyat.github.io/blog/weight-space-programming/) (Takehi et al., MIT CSAIL, Nov 2026) reports that a frontier LLM can read the raw weights of small networks, for example telling which game a policy network plays, and can even edit them to fix a failure. If this holds up, it is a new kind of explainable AI: a tool that inspects a model directly, without data, explains in words what the model relies on, and might even edit out a shortcut it finds.

**Reading Medical Weights tests whether this works on lung CT.** The model under audit is a small classifier (about 19k weights) that rates lung nodules on 24×24 crops from LIDC-IDRI (public lung CT) as likely malignant or benign. The LLM gets only the weights, never the training data, and has to say whether the model relies on a shortcut, where, and of what kind. The difficulty rises step by step, from a synthetic pilot up to realistic CT shortcuts:

1. **Pilot run** (done, see below): synthetic blobs with a bright marker in one corner.
2. **Blatant marker** on real CT crops.
3. **Faint marker** that is only partly correlated with the label.
4. **Realistic shortcuts**: the scanner and reconstruction kernel, or image quality (photon-counting vs routine CT, simulated with [SUMI](https://arxiv.org/abs/2604.07329)).
5. **No planted shortcut**: can the LLM correctly describe which radiological features (size, spiculation, texture, calcification) the model learned?

Every step has a control model without the shortcut, so false alarms count as much as hits.

The blog post used GPT-6 Astra. This project uses a different LLM, Claude Opus 5.5, out of the box through the API, with fixed prompts. Shortcut and control models get the identical prompt under opaque IDs, and Negative results count.

## First result: pilot run

The pilot run uses synthetic 24×24 blob images, where the class is the blob size, and the same 19k-weight MLP that will later be used on CT crops. Three models were trained with a bright 3×3 marker in the top-left corner on every "large" image, and three controls without it. Claude Opus 5.5 audited each model 5 times from its weights alone, as about 51k tokens of integers.

| | |
|---|---|
| Marker models: shortcut found, in the right corner, with the right direction | **15 / 15** |
| Controls: correctly said "no shortcut" | **10 / 15** |
| Majority vote per model (3 of 5 audits) | **5 / 6** (one control flagged in 3/5) |

When there was a marker, Claude found it every time and located it exactly. On controls it sometimes flagged a "patch" of background noise, but with clearly lower confidence (≤ 0.80, versus ≥ 0.92 for every real marker). As a proof of concept this works: an off-the-shelf LLM read a planted shortcut straight out of ~19k raw weights and described it in words. As a sanity check, a simple data-free weight-energy map flags the same corner, which confirms the marker is visibly there in the weights. The CT ladder will move on to shortcuts that will be harder to see.

![results](runs/pilot/figures/results.png)

**Details, figures and reproduction: [docs/pilot_run.md](docs/pilot_run.md)**

## Next steps

1. **CT data.** LIDC-IDRI lung CT (1,010 patients, public, CC BY 3.0). Extract 24×24 nodule-centred crops with DICOM metadata, label them by the radiologists' malignancy rating, and split at patient level.
2. **Shortcut ladder.** The same MLP trained with shortcuts that get harder to see:
   - a blatant marker
   - a faint, partly correlated marker
   - a real scanner / reconstruction-kernel shortcut
   - an image-quality shortcut (photon-counting-like vs routine CT, via the [SUMI](https://arxiv.org/abs/2604.07329) simulator)
   - no planted shortcut at all: which radiological features (size, spiculation, texture, calcification) did the model learn?
3. **Baselines.** Integrated Gradients (needs data) and the data-free weight-energy map as reference points on every rung.
4. **Access levels.** Weights only → weights + Python tools → weights + a few activations.
5. **Robustness checks.** Re-audit after function-preserving weight changes (hidden-unit permutation, rescaling), and audit blinded model pairs to separate weight reading from guessing based on prior medical knowledge.

## Repository

```
PROTOCOL.md          record of the settings used in each experiment
prompts/             fixed auditor prompts
src/rmw/              data, model, training, serialization, audit runner, scoring, figures
scripts/run_pilot.py pilot run, stage by stage
runs/pilot/          models, serialized weights, raw audit answers, scores, figures
keys/                ID -> model mapping (never shown to the auditor)
docs/                write-ups
```
