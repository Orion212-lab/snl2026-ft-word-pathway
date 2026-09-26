"""Ocular control for the poster block-2 effect (Word > Non-word, per-word evoked, gradiometers).

The syntaxIM raws have no EOG channel. MISC001 / MISC002 / MISC003 carry an analog eye-tracker output
(inferred from the signals: gaze x, gaze y, pupil; the pupil channel saturates near -5 V during blinks /
track loss); MISC010 is the photodiode.

Stage `run` (per subject, writes <out>/eye_sub-XX.pkl):
  per-word epochs (-0.2-1.0 s) from the same annotations as the block-2/3 pipeline
  (MINIBLOCK/ONEWORD/<cond> + ONEWORD/<cond>, de-duplicated by sample), read on demand;
  - blinks: pupil < -4.5 V (gaps < 50 ms closed, events < 30 ms dropped), padded by 100 ms;
  - saccades: gaze velocity (20 ms boxcar) > median + 6 robust SD, >= 8 ms, blink samples excluded;
  - per condition: blinks / saccades per word (0-1 s), gaze dispersion, blink probability over time;
  - MEG: 40 Hz low-pass, fixation baseline (mean over FIXATION 0.5-1.95 s), gradiometer pair RMS of
    the condition average at 250 Hz, for epoch sets all / noblink / eyeclean (no blink, no saccade) /
    kept (PTP mag 4000 fT, grad 4000 fT/cm on unfiltered data) / kept_eyeclean.
  proc-clean splits are read one by one (some are not linked); proc-sss is the fallback when a
  proc-clean file holds no ONEWORD events.
Stage `group` (reads all pkls, writes <out>/eye_control_group.txt + .npz):
  Word vs Non-word ocular metrics; time-resolved blink probability (1-D cluster test); block-2 effect:
  spatio-temporal cluster test on the `all` set defines the cluster, whose mean W - NW is then compared
  across epoch sets, plus fresh cluster tests on `noblink` / `eyeclean`; across-subject correlation of
  ocular differences with the sensor effect.
Differences from the poster block 2: no per-tag trial equalization, PTP on unfiltered data.

Usage:
  python eye_control_perword.py run   --bids-root ROOT --out DIR --subjects 03 04 ...
  python eye_control_perword.py group --out DIR
"""
import argparse
import glob
import warnings
import io
import pickle
import sys
import time
from contextlib import redirect_stdout
from pathlib import Path

import mne
import numpy as np
from scipy import ndimage as ndi
from scipy import stats

mne.set_log_level("ERROR")
warnings.filterwarnings("ignore", category=RuntimeWarning)
EYE = ["MISC001", "MISC002", "MISC003"]
BLINK_V = -4.5
CONDS = {"WORD/F1": "ONEWORD/WORD/F1", "WORD/F2": "ONEWORD/WORD/F2",
         "NONWORD/F1": "ONEWORD/NONWORD/F1", "NONWORD/F2": "ONEWORD/NONWORD/F2"}
REJECT = dict(mag=4000e-15, grad=4000e-13)
SETS = ("all", "noblink", "eyeclean", "kept", "kept_eyeclean")
T_EYE = np.round(np.arange(-0.2, 1.0 + 1e-9, 0.001), 3)  # common 1 kHz grid (proc-sss raws are 2 kHz)


# ------------------------------------------------------------------------------------------ run
def word_events(raw):
    events, evdict = mne.events_from_annotations(raw)
    rows, event_id = [], {}
    for new_id, (lab, key) in enumerate(CONDS.items(), start=1):
        ids = [evdict[k] for k in (f"MINIBLOCK/{key}", key) if k in evdict]
        s = np.unique(events[np.isin(events[:, 2], ids), 0])
        if s.size:
            rows.append(np.column_stack([s, np.zeros_like(s), np.full_like(s, new_id)]))
            event_id[lab] = new_id
    if not rows:
        raise ValueError("no ONEWORD events")
    ev = np.concatenate(rows)
    return ev[np.argsort(ev[:, 0])], event_id


def onsets(mask):
    d = np.diff(mask.astype(int), axis=1)
    return (d == 1).sum(1) + mask[:, 0]


def load_epochs(meg_dir, sub):
    for proc in ("clean", "sss"):
        files = sorted(meg_dir.glob(f"sub-{sub}_ses-01_task-syntaxIM_proc-{proc}_split-*_raw.fif")) or \
            sorted(meg_dir.glob(f"sub-{sub}_ses-01_task-syntaxIM_proc-{proc}_raw.fif"))
        eps, fix_vecs, covered = [], [], set()
        for f in files:
            if str(f) in covered:
                continue
            raw = mne.io.read_raw_fif(f, preload=False)
            covered.update(str(x) for x in raw.filenames)  # linked splits are read once
            if not all(c in raw.ch_names for c in EYE):
                raise RuntimeError("no MISC eye channels")
            try:
                ev, eid = word_events(raw)
            except ValueError:
                continue
            fev, fid = mne.events_from_annotations(raw, event_id={"FIXATION": 1})
            if len(fev):
                fx = mne.Epochs(raw, fev, fid, tmin=0.5, tmax=1.95, baseline=None, picks="meg", preload=True)
                fix_vecs.append(fx.get_data().mean(axis=(0, 2)))
            picks = list(mne.pick_types(raw.info, meg=True)) + [raw.ch_names.index(c) for c in EYE]
            eps.append(mne.Epochs(raw, ev, eid, tmin=-0.2, tmax=1.0, baseline=None, picks=picks,
                                  preload=True, reject=None))
        if eps:
            ep = eps[0] if len(eps) == 1 else mne.concatenate_epochs(eps, on_mismatch="ignore")
            return ep, fix_vecs, proc
    raise RuntimeError("no ONEWORD word events in proc-clean or proc-sss")


def run_subject(meg_dir, sub):
    t0 = time.time()
    ep, fix_vecs, source = load_epochs(meg_dir, sub)
    t, sf = ep.times, ep.info["sfreq"]
    meg = ep.copy().pick("meg")
    ptp = np.ptp(meg.get_data(), axis=-1)
    types = np.array(meg.get_channel_types())
    keep = (ptp[:, types == "mag"].max(1) < REJECT["mag"]) & (ptp[:, types == "grad"].max(1) < REJECT["grad"])
    X = ep.copy().pick(EYE).get_data()
    gx, gy, pup = X[:, 0], X[:, 1], X[:, 2]
    ms = lambda x: max(1, int(round(x * sf / 1000)))
    blink = np.array([ndi.binary_opening(ndi.binary_closing(b, np.ones(ms(50))), np.ones(ms(30)))
                      for b in pup < BLINK_V])
    bpad = np.array([ndi.binary_dilation(b, np.ones(2 * ms(100) + 1)) for b in blink])
    k = np.ones(ms(20)) / ms(20)
    sx, sy = (ndi.convolve1d(g, k, axis=1, mode="nearest") for g in (gx, gy))
    vel_ok = np.where(bpad, np.nan, np.hypot(np.gradient(sx, axis=1), np.gradient(sy, axis=1)) * sf)
    med = np.nanmedian(vel_ok)
    sd = np.nanmedian(np.abs(vel_ok - med)) * 1.4826
    sacc = np.array([ndi.binary_opening(ndi.binary_closing(r, np.ones(ms(10))), np.ones(ms(8)))
                     for r in (vel_ok > med + 6 * sd) & ~bpad])
    post = t >= 0
    noblink = ~bpad.any(1)
    eyeclean = noblink & ~sacc[:, post].any(1)

    G = meg.copy().pick("grad")
    gd = mne.filter.filter_data(G.get_data(), sf, None, 40.0, verbose=False)
    if fix_vecs:
        gd = gd - np.mean(fix_vecs, 0)[[meg.ch_names.index(c) for c in G.ch_names]][None, :, None]
    pairs = sorted({c[:-1] for c in G.ch_names})
    pi = [(G.ch_names.index(p + "2"), G.ch_names.index(p + "3")) for p in pairs]
    dec = max(1, int(round(sf / 250)))

    inv = {v: k for k, v in ep.event_id.items()}
    lab = np.array([inv[x] for x in ep.events[:, 2]])
    res = {"_meta": dict(pairs=pairs, times=t[::dec], eye_times=T_EYE, source=source, n_total=len(lab),
                        track_loss=float(blink.mean()))}
    masks = dict(all=np.ones(len(lab), bool), noblink=noblink, eyeclean=eyeclean, kept=keep,
                 kept_eyeclean=keep & eyeclean)
    for sel_name, sel in masks.items():
        for grp, members in (("WORD", ["WORD/F1", "WORD/F2"]), ("NONWORD", ["NONWORD/F1", "NONWORD/F2"])):
            m = sel & np.isin(lab, members)
            if m.sum() == 0:
                continue
            gxm, gym = np.where(bpad[m], np.nan, gx[m]), np.where(bpad[m], np.nan, gy[m])
            with np.errstate(all="ignore"):
                gsd = float(np.nanmean(np.nanstd(gxm[:, post], 1) + np.nanstd(gym[:, post], 1)))
            evm = gd[m].mean(0)
            res[f"{sel_name}/{grp}"] = dict(
                n=int(m.sum()), blink_p=np.interp(T_EYE, t, blink[m].mean(0)), blinks=float(onsets(blink[m][:, post]).mean()),
                sacc=float(onsets(sacc[m][:, post]).mean()), gaze_sd=gsd,
                grad_rms=np.sqrt(np.stack([evm[a] ** 2 + evm[b] ** 2 for a, b in pi]))[:, ::dec])
    print(f"sub-{sub} [proc-{source}]: {len(lab)} words, PTP-kept {keep.sum()}, no-blink {noblink.sum()}, "
          f"eye-clean {eyeclean.sum()}; blinks/word W {res['all/WORD']['blinks']:.2f} "
          f"NW {res['all/NONWORD']['blinks']:.2f}; track loss {blink.mean():.1%}; {time.time() - t0:.0f} s",
          flush=True)
    return res


# ---------------------------------------------------------------------------------------- group
def group(out):
    res = {}
    for f in sorted(glob.glob(str(out / "eye_sub-*.pkl"))):
        res[Path(f).stem.split("-")[1]] = pickle.load(open(f, "rb"))
    bad = [s for s in res if res[s]["_meta"].get("track_loss", 0) > 0.5]  # eye tracker lost most of the time
    for s in bad:
        print(f"excluded sub-{s}: eye-tracker signal lost {res[s]['_meta']['track_loss']:.0%} of the time")
        del res[s]
    subs = sorted(res)
    print(f"N = {len(subs)}: {' '.join(subs)}")
    print("words per subject: " + " ".join(f"{s}:{res[s]['_meta']['n_total']}" for s in subs))

    for sel in ("all", "kept"):
        print(f"\n=== ocular metrics, epochs: {sel}")
        for m in ("blinks", "sacc", "gaze_sd"):
            ok = [s for s in subs if f"{sel}/WORD" in res[s] and f"{sel}/NONWORD" in res[s]]
            w = np.array([res[s][f"{sel}/WORD"][m] for s in ok])
            n = np.array([res[s][f"{sel}/NONWORD"][m] for s in ok])
            g = np.isfinite(w) & np.isfinite(n)
            d = (w - n)[g]
            tt, wx = stats.ttest_1samp(d, 0), stats.wilcoxon(d)
            print(f"  {m:8s} N={g.sum():2d}  W {w[g].mean():.3f}  NW {n[g].mean():.3f}  "
                  f"dz = {d.mean()/d.std(ddof=1):+.2f}, t = {tt.statistic:+.2f}, p = {tt.pvalue:.4f}; "
                  f"Wilcoxon p = {wx.pvalue:.4f}; {int((d > 0).sum())}/{len(d)} W>NW")

    et = T_EYE[(T_EYE >= -0.15) & (T_EYE <= 0.95)]  # drop epoch edges (morphological-filter border effects)
    pw = np.array([np.interp(et, res[s]["_meta"]["eye_times"], res[s]["all/WORD"]["blink_p"]) for s in subs])
    pn = np.array([np.interp(et, res[s]["_meta"]["eye_times"], res[s]["all/NONWORD"]["blink_p"]) for s in subs])
    B = pw - pn
    print(f"\nblink probability, mean over 0-0.95 s: W {pw[:, et >= 0].mean():.3f}, NW {pn[:, et >= 0].mean():.3f}")
    stat = lambda x: mne.stats.ttest_1samp_no_p(x, sigma=1e-3)  # "hat" regularisation: no-blink samples have 0 variance
    tv, clu, pv, _ = mne.stats.permutation_cluster_1samp_test(B, n_permutations=5000, seed=0, out_type="mask",
                                                              stat_fun=stat)
    print("\nblink probability W - NW over time (1-D cluster test, all epochs):")
    for c, p in zip(clu, pv):
        idx = np.flatnonzero(c)
        if p < 0.2 and idx.size >= 20:  # >= 20 ms
            print(f"  {et[idx[0]]*1000:.0f}-{et[idx[-1]]*1000:.0f} ms, p = {p:.4f}, "
                  f"{'W>NW' if tv[c].mean() > 0 else 'NW>W'}")

    pairs = res[subs[0]]["_meta"]["pairs"]
    mt = res[subs[0]]["_meta"]["times"]
    adj, names = mne.channels.read_ch_adjacency("neuromag306mag")
    order = [names.index(p + "1") for p in pairs]
    adj = adj[order][:, order]

    def diff(s, sel):
        r = res[s]
        return (r[f"{sel}/WORD"]["grad_rms"] - r[f"{sel}/NONWORD"]["grad_rms"]).T * 1e13  # time x sensor, fT/cm

    def cluster(X):
        return mne.stats.spatio_temporal_cluster_1samp_test(
            X, threshold=stats.t.ppf(0.975, len(X) - 1), n_permutations=5000, adjacency=adj, seed=0,
            out_type="mask")

    Xall = np.array([diff(s, "all") for s in subs])
    T, clu, pv, _ = cluster(Xall)
    i = int(np.argmin(pv))
    best = clu[i]
    tt_, ss_ = np.where(best)
    print(f"\n=== block-2 effect, all epochs: best cluster p = {pv[i]:.4f}, "
          f"{'W>NW' if T[best].mean() > 0 else 'NW>W'}, {mt[tt_.min()]*1000:.0f}-{mt[tt_.max()]*1000:.0f} ms, "
          f"{len(np.unique(ss_))} sensors (defines the ROI below)")
    eff = {}
    for sel in SETS:
        ss = [s for s in subs if all(f"{sel}/{g}" in res[s] and res[s][f"{sel}/{g}"]["n"] >= 5
                                     for g in ("WORD", "NONWORD"))]
        X = np.array([diff(s, sel) for s in ss])
        v = X[:, best].mean(1)
        eff[sel] = dict(zip(ss, v))
        tt = stats.ttest_1samp(v, 0)
        nW = np.mean([res[s][f"{sel}/WORD"]["n"] for s in ss])
        print(f"  {sel:14s} N = {len(ss):2d} (W {nW:.0f} epochs/subj): cluster-ROI W - NW = {v.mean():+.2f} fT/cm, "
              f"dz = {v.mean()/v.std(ddof=1):+.2f}, p = {tt.pvalue:.5f}; {(v > 0).sum()}/{len(v)} positive")
        if sel in ("noblink", "eyeclean") and len(ss) > 5:
            T2, c2, p2, _ = cluster(X)
            for c, p in sorted(zip(c2, p2), key=lambda x: x[1])[:2]:
                a, b = np.where(c)
                print(f"      own cluster test: p = {p:.4f}, {'W>NW' if T2[c].mean() > 0 else 'NW>W'}, "
                      f"{mt[a.min()]*1000:.0f}-{mt[a.max()]*1000:.0f} ms, {len(np.unique(b))} sensors")

    print("\nacross subjects, ocular W - NW difference vs sensor cluster effect (all epochs):")
    for m in ("blinks", "sacc", "gaze_sd"):
        de = np.array([res[s]["all/WORD"][m] - res[s]["all/NONWORD"][m] for s in subs])
        ef = np.array([eff["all"][s] for s in subs])
        g = np.isfinite(de)
        r = stats.spearmanr(de[g], ef[g])
        print(f"  {m:8s} rho = {r.statistic:+.2f}, p = {r.pvalue:.3f} (N = {g.sum()})")
    np.savez(out / "eye_control_group.npz", subjects=np.array(subs), eye_times=et, blink_diff=B,
             **{f"effect_{k}": np.array([v.get(s, np.nan) for s in subs]) for k, v in eff.items()})


def main():
    p = argparse.ArgumentParser()
    p.add_argument("stage", choices=("run", "group"))
    p.add_argument("--bids-root", type=Path)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--subjects", nargs="+", default=[])
    a = p.parse_args()
    a.out.mkdir(parents=True, exist_ok=True)
    if a.stage == "run":
        der = a.bids_root / "derivatives" / "mne-bids-pipeline"
        for s in a.subjects:
            try:
                pickle.dump(run_subject(der / f"sub-{s}" / "ses-01" / "meg", s),
                            open(a.out / f"eye_sub-{s}.pkl", "wb"))
            except Exception as e:
                print(f"sub-{s}: FAILED {type(e).__name__}: {e}", flush=True)
    else:
        buf = io.StringIO()
        with redirect_stdout(buf):
            group(a.out)
        (a.out / "eye_control_group.txt").write_text(buf.getvalue())
        sys.stdout.write(buf.getvalue())


if __name__ == "__main__":
    main()
