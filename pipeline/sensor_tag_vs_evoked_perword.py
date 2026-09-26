#!/usr/bin/env python
"""Poster block 3, sensor-only: tag spectrum AND word-locked evoked field from the SAME per-word epochs.

ft_vs_evoked_perword_roi.py (p3) also computes source-space measures, so it needs an MRI/inverse and
runs on the SOURCE roster; block 3 is sensor-level only. This script drops the source part so the
whole SENSOR roster can enter (N limited only by artifacts).

Per subject, epoch selection is identical to word_evoked_spectrum.py (p4): per-word events
(word 1 + words 2-10), 0.1-40 Hz, PTP rejection over the whole word (-0.2..WORD_DUR, the tag SNR uses
0..WORD_DUR), within-subject equalization of the 4 conditions (capped at --n-epochs), subjects with
< --min-epochs per condition dropped. From those same epochs, per condition:
  snr     per-word phase-locked SNR spectrum: condition average, one FFT over 0..WORD_DUR
          (df = 0.353 Hz), power / mean of 3 neighbours per side (1 skipped), averaged over the ROI's
          gradiometers (as p4);
  evoked  mean |gradiometer field| over the ROI, fixation baseline subtracted (as p3), -0.2..1.0 s.
Sensor ROIs: Neuromag 'Left-temporal'; 'Left-occipital' + 'Right-occipital' (gradiometers).

Output: <results-dir>/sub-all/sensor_tag_vs_evoked/sensor_tag_vs_evoked.npz
  freqs, times, subjects, rois (= sens_rois), n_epochs (sub,),
  snr_{COND} (sub, roi, freq), sens_evoked_{COND} (sub, roi, time) in T/m,
  COND in WORD_F1, WORD_F2, NONWORD_F1, NONWORD_F2
"""
from __future__ import annotations

import sys
from argparse import ArgumentParser
from pathlib import Path

import mne
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "03_roi_hypothesis"))
import oneword_roi_freq_timecourse as rft  # noqa: E402
from intermodulation import freqtag_spec  # noqa: E402

ROIS = {"Left-temporal": ["Left-temporal"], "Occipital": ["Left-occipital", "Right-occipital"]}
K, SKIP = 3, 1
CROP = (-0.2, 1.0)
CONDS = ("WORD_F1", "WORD_F2", "NONWORD_F1", "NONWORD_F2")


def snr_spectrum(x: np.ndarray, sfreq: float) -> tuple[np.ndarray, np.ndarray]:
    """(n_ch, n_t) -> freqs, (n_ch, n_freq) SNR vs K neighbours per side (SKIP skipped). Same as p4."""
    p = np.abs(np.fft.rfft(x, axis=-1)) ** 2
    f = np.fft.rfftfreq(x.shape[-1], 1 / sfreq)
    snr = np.full_like(p, np.nan)
    offs = [s * k for s in (-1, 1) for k in range(1 + SKIP, 1 + SKIP + K)]
    for b in range(1 + SKIP + K, p.shape[-1] - (1 + SKIP + K)):
        snr[:, b] = p[:, b] / p[:, [b + o for o in offs]].mean(-1)
    return f, snr


def process(subject, args, roi_chs):
    proc_dir = args.derivatives_root / f"sub-{subject}" / f"ses-{args.session}" / "meg"
    raw = rft._load_raw_for_hilbert(proc_dir, subject, args.session, args.task, args.proc)
    if raw is None:
        print(f"  sub-{subject}: no continuous raw -- skipping")
        return None
    raw.load_data()
    raw.filter(l_freq=0.1, h_freq=40.0, picks="meg", verbose=False)
    events, evdict = mne.events_from_annotations(raw, verbose=False)
    word_events, keepev = rft._build_word_evoked_events(events, evdict, "all")
    fix_vec = rft._fixation_baseline_vector(raw, events, evdict)
    ep = mne.Epochs(raw, events=word_events, event_id=keepev, tmin=args.reject_tmin,
                    tmax=freqtag_spec.WORD_DUR, picks="meg", preload=True, baseline=None,
                    event_repeated="drop", reject=rft.WORD_EVOKED_REJECT,
                    reject_tmin=args.reject_tmin, reject_tmax=args.reject_tmax, verbose=False)
    del raw
    if fix_vec.shape[0] != len(ep.ch_names):
        print(f"  sub-{subject}: fixation baseline has {fix_vec.shape[0]} channels, epochs "
              f"{len(ep.ch_names)} -- skipping")
        return None
    present = [k for k in keepev if k in ep.event_id and len(ep[k]) > 0]
    if len(present) < 4:
        print(f"  sub-{subject}: missing condition(s) after rejection -- skipping")
        return None
    rft._equalize_epochs(ep, present, args.n_epochs)
    counts = {k: len(ep[k]) for k in present}
    print(f"  sub-{subject} equalized: {counts}")
    if len(set(counts.values())) != 1 or min(counts.values()) < args.min_epochs:
        print(f"  sub-{subject}: unequal or < {args.min_epochs} epochs per condition -- skipping")
        return None
    ch = ep.ch_names
    idx = {r: [ch.index(c) for c in cs if c in ch] for r, cs in roi_chs.items()}
    tw = (ep.times >= 0) & (ep.times < freqtag_spec.WORD_DUR)
    crop = (ep.times >= CROP[0]) & (ep.times <= CROP[1])
    out = {"n": min(counts.values()), "times": ep.times[crop]}
    for cond in present:
        key = cond.replace("/", "_")
        avg = ep[cond].average().data
        f, snr = snr_spectrum(avg[:, tw], ep.info["sfreq"])
        out["freqs"] = f
        out[f"snr_{key}"] = np.stack([np.nanmean(snr[idx[r]], 0) for r in roi_chs])
        d = avg - fix_vec[:, None]  # constant per-channel offset: same as subtracting it from every epoch
        out[f"evk_{key}"] = np.stack([np.abs(d[idx[r]][:, crop]).mean(0) for r in roi_chs])
    return out


def main():
    p = ArgumentParser(description=__doc__)
    p.add_argument("--bids-root", type=Path, required=True)
    p.add_argument("--subjects", nargs="+", required=True)
    p.add_argument("--results-dir", type=Path, required=True)
    p.add_argument("--session", default="01")
    p.add_argument("--task", default="syntaxIM")
    p.add_argument("--proc", default="clean")
    p.add_argument("--n-epochs", type=int, default=20)
    p.add_argument("--min-epochs", type=int, default=8)
    p.add_argument("--reject-tmin", type=float, default=-0.2)
    p.add_argument("--reject-tmax", type=float, default=freqtag_spec.WORD_DUR,
                   help="end of the PTP-rejection window (default: whole word, as p3/p4)")
    args = p.parse_args()
    args.derivatives_root = args.bids_root / "derivatives" / "mne-bids-pipeline"
    out_dir = args.results_dir / "sub-all" / "sensor_tag_vs_evoked"
    out_dir.mkdir(parents=True, exist_ok=True)
    roi_chs = {r: [c.replace(" ", "") for s in sels for c in mne.read_vectorview_selection(s) if c[-1] in "23"]
               for r, sels in ROIS.items()}
    print(f"sensor ROIs: { {k: len(v) for k, v in roi_chs.items()} } gradiometers")
    res, subs = [], []
    for s in args.subjects:
        try:
            r = process(s, args, roi_chs)
        except Exception as e:  # keep going; report which subject failed and why
            print(f"  sub-{s}: FAILED -- {type(e).__name__}: {e}")
            r = None
        if r is not None:
            res.append(r)
            subs.append(s)
    print(f"\nN = {len(subs)} usable: {' '.join(subs)}; dropped: "
          f"{' '.join(s for s in args.subjects if s not in subs) or 'none'}")
    if len(res) < 2:
        raise SystemExit("fewer than 2 usable subjects")
    f, t = res[0]["freqs"], res[0]["times"]
    assert all(np.allclose(r["freqs"], f) for r in res), "frequency axes differ (sampling rate?)"
    assert all(np.allclose(r["times"], t) for r in res), "time axes differ (sampling rate?)"
    npz = {"freqs": f, "times": t, "subjects": np.array(subs), "rois": np.array(list(ROIS)),
           "sens_rois": np.array(list(ROIS)), "n_epochs": np.array([r["n"] for r in res])}
    for c in CONDS:
        npz[f"snr_{c}"] = np.stack([r[f"snr_{c}"] for r in res])
        npz[f"sens_evoked_{c}"] = np.stack([r[f"evk_{c}"] for r in res])
    np.savez(out_dir / "sensor_tag_vs_evoked.npz", **npz)
    print(f"saved {out_dir / 'sensor_tag_vs_evoked.npz'}")


if __name__ == "__main__":
    main()
