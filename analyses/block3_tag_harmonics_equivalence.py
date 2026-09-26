"""Analysis 3: block-1 tag response with harmonics + equivalence tests.

Input: p4 per-word phase-locked SNR spectra (2.83 s FFT, df = 0.353 Hz; left-temporal and occipital
gradiometers; SNR = power / mean of 3 neighbour bins per side). Each condition is read at its own
tag: WORD/NONWORD_F1 at k x 6.00 Hz (bins 17k), WORD/NONWORD_F2 at k x 7.06 Hz (bins 20k).
Metrics: fundamental SNR; harmonic sum S = sum_k (SNR_k - 1), k = 1..4 (<= 28.2 Hz), a proxy for the
baseline-corrected amplitude sum (Retter & Rossion 2016) that is exact up to the noise level, which is
condition-matched here. Tests: paired t Word vs Non-word per tag and F1/F2-averaged; TOST with
bounds dz = +/-0.5 and +/-0.8; JZS Bayes factor BF01 (Cauchy prior r = 0.707, Rouder et al. 2009).
"""
import sys
import numpy as np
from scipy import stats, integrate

z = np.load(sys.argv[1])
f = z["freqs"]
df = f[1] - f[0]
rois = list(z["rois"])
subs = list(z["subjects"])
restrict = sys.argv[2].split(",") if len(sys.argv) > 2 else None
sel = [i for i, s in enumerate(subs) if restrict is None or s in restrict]
print(f"df = {df:.4f} Hz; N = {len(sel)} ({'restricted' if restrict else 'all p4'}); bins F1 = {6.0/df:.2f}, F2 = {(240/34)/df:.2f}")
B1, B2 = int(round(6.0 / df)), int(round((240 / 34) / df))
H = range(1, 5)


def bf01(t, n, r=0.707):
    """JZS BF01 for a one-sample / paired t-test (Rouder et al., 2009)."""
    v = n - 1
    num = (1 + t ** 2 / v) ** (-(v + 1) / 2)
    g = lambda g_: ((1 + n * g_ * r ** 2) ** -0.5 * (1 + t ** 2 / ((1 + n * g_ * r ** 2) * v)) ** (-(v + 1) / 2)
                    * (2 * np.pi) ** -0.5 * g_ ** -1.5 * np.exp(-1 / (2 * g_)))
    den = integrate.quad(g, 0, np.inf)[0]
    return num / den


def tost(d, bound_dz):
    n = len(d); sd = d.std(ddof=1); se = sd / np.sqrt(n); b = bound_dz * sd
    p_lo = stats.t.sf((d.mean() + b) / se, n - 1)
    p_hi = stats.t.cdf((d.mean() - b) / se, n - 1)
    return max(p_lo, p_hi)


def metric(cond, bin0, kind, r):
    S = z[f"snr_{cond}"][sel, r]
    if kind == "fund":
        return S[:, bin0]
    return sum(S[:, k * bin0] - 1 for k in H)


for r, roi in enumerate(rois):
    print(f"\n=== {roi}")
    for kind in ("fund", "harm1-4"):
        w1, n1 = metric("WORD_F1", B1, kind, r), metric("NONWORD_F1", B1, kind, r)
        w2, n2 = metric("WORD_F2", B2, kind, r), metric("NONWORD_F2", B2, kind, r)
        det = 0.25 * (w1 + n1 + w2 + n2) - (1 if kind == "fund" else 0)
        td = stats.ttest_1samp(det, 0)
        print(f"  {kind:8s} detection (> {'1' if kind == 'fund' else '0'}): mean {det.mean() + (1 if kind == 'fund' else 0):.2f}, "
              f"t({len(det)-1}) = {td.statistic:.2f}, p = {td.pvalue:.2g}")
        for lab, d in (("F1", w1 - n1), ("F2", w2 - n2), ("F1+F2", 0.5 * ((w1 - n1) + (w2 - n2)))):
            tt = stats.ttest_1samp(d, 0); n = len(d)
            print(f"    W-NW {lab:6s}: dz = {d.mean()/d.std(ddof=1):+.2f}, t({n-1}) = {tt.statistic:+.2f}, p = {tt.pvalue:.3f}; "
                  f"TOST p(|dz|<.5) = {tost(d, .5):.3f}, p(|dz|<.8) = {tost(d, .8):.3f}; BF01 = {bf01(tt.statistic, n):.2f}")
n = len(sel)
print(f"\nSensitivity: N = {n}, 80% power, alpha .05 two-sided -> MDE dz = "
      f"{(stats.t.ppf(.975, n-1) + stats.t.ppf(.8, n-1)) / np.sqrt(n):.2f}")
