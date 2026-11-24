"""LIDC-IDRI preprocessing (PROTOCOL.md section C).

Stages (run in order):
  index    one DICOM header per CT series -> series.csv
  nodules  pylidc annotations -> nodules.csv (inclusion + label, joined with series metadata)
  crops    3 raw-HU 48x48 crops per nodule -> crops.npz  (--workers N, --limit K for a test run)
  sizenorm 3 raw-HU 24x24 size-normalised crops per nodule -> crops_sizenorm.npz (PROTOCOL.md G1)
  split    patient-level 70/15/15 + leakage check -> split column in nodules.csv, split.json
  report   count tables + figures in runs/lidc/figures/
"""

import argparse
import json
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "lidc_idri"
OUT = ROOT / "data" / "processed"
CACHE = OUT / "cache"
CACHE_SN = OUT / "cache_sizenorm"
FIG = ROOT / "runs" / "lidc" / "figures"


def stage_index(args):
    from rmw.lidc import find_ct_series, read_series_header
    dirs = find_ct_series(RAW)
    with ProcessPoolExecutor(args.workers) as ex:
        rows = list(ex.map(read_series_header, dirs, chunksize=8))
    df = pd.DataFrame(rows).sort_values(["patient_id", "series_uid"])
    OUT.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUT / "series.csv", index=False)
    print(f"{len(df)} CT series, {df.patient_id.nunique()} patients")
    print(df.manufacturer.value_counts().to_string())


def stage_nodules(args):
    from rmw.lidc import import_pylidc, nodules_for_scan
    pl = import_pylidc()
    series = pd.read_csv(OUT / "series.csv")
    rows, n_scans = [], 0
    for scan in pl.query(pl.Scan).order_by(pl.Scan.id):
        n_scans += 1
        rows += nodules_for_scan(scan)
    nod = pd.DataFrame(rows)
    meta = series[["series_uid", "manufacturer", "model_name", "kernel", "slice_thickness",
                   "pixel_spacing", "kvp", "tube_current", "series_dir"]]
    nod = nod.merge(meta, on="series_uid", how="left", validate="many_to_one")
    missing = nod.series_dir.isna().sum()
    if missing:
        print(f"WARNING: {missing} nodules whose series is not on disk; dropped")
        nod = nod[nod.series_dir.notna()]
    nod = nod.reset_index(drop=True)
    nod.to_csv(OUT / "nodules.csv", index=False)
    print(f"{n_scans} scans in pylidc; {len(nod)} nodules with >=3 readers and >=3 mm "
          f"in {nod.series_uid.nunique()} series / {nod.patient_id.nunique()} patients")
    print(nod.label.map({0: "benign", 1: "malignant", -1: "excluded"}).value_counts().to_string())


def _crop_job(series_dir, records, sizenorm=False):
    from rmw.lidc import crops_for_series
    return records, crops_for_series(series_dir, records, sizenorm=sizenorm)


def _fill_cache(nod, cache, args, sizenorm=False):
    cache.mkdir(parents=True, exist_ok=True)
    groups = list(nod.groupby("series_uid", sort=True))
    if args.limit:
        groups = groups[:args.limit]
    todo = [(uid, g) for uid, g in groups if not (cache / f"{uid}.npz").exists()]
    print(f"{len(groups)} series, {len(todo)} to process, {args.workers} workers")
    t0, fails = time.time(), []
    with ProcessPoolExecutor(args.workers) as ex:
        futs = {ex.submit(_crop_job, g.series_dir.iloc[0], g.to_dict("records"), sizenorm): uid for uid, g in todo}
        for i, f in enumerate(as_completed(futs), 1):
            uid = futs[f]
            try:
                recs, res = f.result()
            except Exception as e:  # report, keep going
                fails.append((uid, repr(e)))
                continue
            np.savez(cache / f"{uid}.npz", hu=res["hu"], nodule_id=np.array([r["nodule_id"] for r in recs]),
                     qa=json.dumps(res["qa"]))
            if i % 50 == 0 or i == len(todo):
                print(f"  {i}/{len(todo)}  {time.time() - t0:.0f}s", flush=True)
    for uid, e in fails:
        print("FAILED", uid, e)
    return groups


def stage_sizenorm(args):
    """PROTOCOL.md G1. Same nodule order as nodules.csv; nodules.csv itself is not changed."""
    nod = pd.read_csv(OUT / "nodules.csv")
    groups = _fill_cache(nod, CACHE_SN, args, sizenorm=True)
    if args.limit:
        print("--limit given: cache filled, crops_sizenorm.npz not assembled")
        return
    hu = {}
    for uid, _ in groups:
        p = CACHE_SN / f"{uid}.npz"
        if p.exists():
            z = np.load(p)
            hu.update({str(n): h for n, h in zip(z["nodule_id"], z["hu"])})
    missing = (~nod.nodule_id.isin(hu.keys())).sum()
    if missing:
        raise SystemExit(f"{missing} nodules without size-normalised crops; not assembling")
    arr = np.stack([hu[n] for n in nod.nodule_id])
    np.savez(OUT / "crops_sizenorm.npz", hu24=arr, nodule_id=np.asarray(nod.nodule_id.tolist(), dtype=str))
    print(f"crops_sizenorm.npz: {arr.shape} {arr.dtype}")


def stage_crops(args):
    nod = pd.read_csv(OUT / "nodules.csv")
    groups = _fill_cache(nod, CACHE, args)
    if args.limit:
        print("--limit given: cache filled, crops.npz not assembled")
        return
    # assemble in nodules.csv order
    hu, qa = {}, []
    for uid, _ in groups:
        p = CACHE / f"{uid}.npz"
        if not p.exists():
            continue
        z = np.load(p)
        for nid, h in zip(z["nodule_id"], z["hu"]):
            hu[str(nid)] = h
        qa += json.loads(str(z["qa"]))
    ok = nod.nodule_id.isin(hu.keys())
    if (~ok).any():
        print(f"WARNING: {(~ok).sum()} nodules without crops (failed series); dropped from crops.npz")
    nod = nod[ok].reset_index(drop=True)
    arr = np.stack([hu[n] for n in nod.nodule_id])
    np.savez(OUT / "crops.npz", hu48=arr, nodule_id=nod.nodule_id.values.astype(str))
    q = pd.DataFrame(qa)
    q.to_csv(OUT / "crops_qa.csv", index=False)
    nod.to_csv(OUT / "nodules.csv", index=False)
    print(f"crops.npz: {arr.shape} {arr.dtype}")
    print(f"QA: contour z mismatch > 0.01 mm: {(q.contour_z_mismatch_mm > 0.01).sum()}, "
          f"z out of range: {(~q.z_in_range).sum()}, rotated 180: {(q.orientation == 'axial_rot180').sum()}, "
          f"duplicate-z series: {(q.n_dup_z > 0).sum()}, xy drift > 0.01 mm: {(q.xy_drift_mm > 0.01).sum()}")


def stage_split(args):
    from rmw.splits import SPLIT_SEED, leakage_check, patient_split
    nod = pd.read_csv(OUT / "nodules.csv")
    nod["split"] = patient_split(nod)
    summary = leakage_check(nod)   # raises on failure
    nod.to_csv(OUT / "nodules.csv", index=False)
    summary["seed"] = SPLIT_SEED
    (OUT / "split.json").write_text(json.dumps(summary, indent=1))
    print("leakage check passed")
    print(json.dumps(summary, indent=1))


def stage_report(args):
    from rmw.lidc_report import fig_examples, fig_groups, fig_slice_thickness, group_counts
    FIG.mkdir(parents=True, exist_ok=True)
    nod = pd.read_csv(OUT / "nodules.csv")
    series = pd.read_csv(OUT / "series.csv")
    tables = {}
    for col in ("manufacturer", "model_name", "kernel"):
        t = group_counts(nod, col)
        tables[col] = t
        print(f"\n== labelled nodules per {col}\n{t.to_string()}")
    if "split" in nod:
        print("\n== split x label\n", pd.crosstab(nod.split, nod.label).to_string())
    with open(FIG.parent / "counts.txt", "w") as f:
        for col, t in tables.items():
            f.write(f"== labelled nodules per {col}\n{t.to_string()}\n\n")
    fig_groups(nod, FIG / "scanner_groups.png")
    fig_slice_thickness(series, FIG / "slice_thickness.png")
    if (OUT / "crops.npz").exists():
        z = np.load(OUT / "crops.npz")
        assert (z["nodule_id"] == nod.nodule_id.values).all()
        fig_examples(nod, z["hu48"], FIG / "examples.png")
    if (OUT / "crops_sizenorm.npz").exists():
        from rmw.lidc_report import fig_sizenorm_examples
        z = np.load(OUT / "crops_sizenorm.npz")
        assert (z["nodule_id"] == nod.nodule_id.values).all()
        fig_sizenorm_examples(nod, z["hu24"], FIG / "sizenorm_examples.png")
    print(f"\nfigures in {FIG}")


def main():
    sys.path.insert(0, str(ROOT / "src"))
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("stage", choices=["index", "nodules", "crops", "sizenorm", "split", "report"])
    ap.add_argument("--workers", type=int, default=16)
    ap.add_argument("--limit", type=int, default=0)
    args = ap.parse_args()
    globals()[f"stage_{args.stage}"](args)


if __name__ == "__main__":
    main()
