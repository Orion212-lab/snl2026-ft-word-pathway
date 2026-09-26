"""Analysis 4: block-2 sensor effect -- GFP noise floor and robustness to sub-20/24.

Input: sensor_evoked_topomap_sequence.npz (per-subject gradiometer field strength = RMS of each planar
pair, WORD and NONWORD, F1+F2 pooled, per-word epochs, fixation baseline, 40 Hz low-pass; N = 26).
(a) GFP of the group Word - Non-word difference vs its sign-flip null (5000 flips): per-sample 95th
    percentile and a max-statistic threshold over time (FWER), pre-stimulus included.
(b) Spatio-temporal cluster test (1-sample on W - NW, t threshold p < .05 two-sided, 5000 perm.,
    neuromag mag-layout adjacency for the 102 pair positions), whole epoch -0.2-1.0 s:
    N = 26 (as poster), N = 24 without sub-20/24, and a 20%-trimmed-mean sign-flip check on the ROI of the
    N = 26 cluster.
"""
import sys
import numpy as np
import mne
from scipy import stats

mne.set_log_level("ERROR")
z = np.load(sys.argv[1], allow_pickle=True)
t = z["times"]
subs = list(z["subjects"])
D = z["word_grad"] - z["nonword_grad"]  # sub x sensor x time (T/m)
D = D * 1e13  # fT/cm
rng = np.random.default_rng(0)
print(f"N = {len(subs)}: {' '.join(subs)}; times {t[0]:.3f}..{t[-1]:.3f} s, {len(t)} samples")


def gfp(x):  # x: sensor x time
    return np.sqrt((x ** 2).mean(0))


# (a) GFP noise floor
obs = gfp(D.mean(0))
null = np.empty((5000, len(t)))
for i in range(5000):
    s = rng.choice([-1, 1], size=len(subs))[:, None, None]
    null[i] = gfp((D * s).mean(0))
p95 = np.percentile(null, 95, axis=0)
maxthr = np.percentile(null.max(1), 95)
above = obs > maxthr
print(f"\n(a) GFP of W - NW: pre-stim mean {obs[t < 0].mean():.2f} fT/cm; sign-flip floor (mean 95th pct) "
      f"{p95.mean():.2f}; FWER (max-over-time) threshold {maxthr:.2f}")
if above.any():
    segs = np.flatnonzero(np.diff(np.r_[0, above.astype(int), 0]))
    for a, b in zip(segs[::2], segs[1::2]):
        print(f"    GFP above FWER floor: {t[a]*1000:.0f}-{t[b-1]*1000:.0f} ms (peak {obs[a:b].max():.2f})")
else:
    print("    GFP never exceeds the FWER floor")
print(f"    fraction of pre-stim samples above per-sample 95th pct: {(obs[t < 0] > p95[t < 0]).mean():.2f}")

# (b) cluster tests
adj, names = mne.channels.read_ch_adjacency("neuromag306mag")
keys = list(z["keys"])
order = [names.index(k + "1") for k in keys]
adj = adj[order][:, order]
thr = stats.t.ppf(0.975, len(subs) - 1)


def cluster(Dsub, label):
    X = Dsub.transpose(0, 2, 1)  # sub x time x sensor
    tv, clu, pv, _ = mne.stats.spatio_temporal_cluster_1samp_test(
        X, threshold=stats.t.ppf(0.975, len(X) - 1), n_permutations=5000, adjacency=adj, tail=0,
        seed=0, out_type="mask")
    print(f"\n(b) {label}: N = {len(X)}")
    best = None
    for c, p in sorted(zip(clu, pv), key=lambda x: x[1])[:4]:
        tt, ss = np.where(c)
        sign = "W>NW" if tv[c].mean() > 0 else "NW>W"
        print(f"    p = {p:.4f}  {sign}  {t[tt.min()]*1000:.0f}-{t[tt.max()]*1000:.0f} ms, {len(np.unique(ss))} sensors")
        if best is None:
            best = c
    return best


c26 = cluster(D, "poster cohort")
keep = [i for i, s in enumerate(subs) if s not in ("20", "24")]
c24 = cluster(D[keep], "without sub-20/24")

# trimmed-mean robustness on the N = 26 cluster ROI (sensors x time of the best cluster)
roi = D[:, c26.T]  # sub x (cluster samples)  -- c26 is time x sensor
m = roi.mean(1)
obs_tm = stats.trim_mean(m, 0.2)
null_tm = np.array([stats.trim_mean(m * rng.choice([-1, 1], len(m)), 0.2) for _ in range(10000)])
print(f"\n    N = 26 cluster ROI mean W - NW: mean {m.mean():.2f}, 20%-trimmed {obs_tm:.2f} fT/cm, "
      f"sign-flip p = {(np.abs(null_tm) >= abs(obs_tm)).mean():.4f}; "
      f"{(m > 0).sum()}/{len(m)} subjects positive")
print("    per-subject cluster-ROI means (fT/cm): " + " ".join(f"{s}:{v:+.1f}" for s, v in zip(subs, m)))
