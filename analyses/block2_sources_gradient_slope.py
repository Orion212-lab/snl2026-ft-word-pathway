"""Is the Word > Non-word source effect graded posterior -> anterior? (block-3 data, N=22, log space)
Per subject: whole-epoch log(W/NW) per ROI; (1) slope over pathway rank (1..10), one-sample t on slopes;
(2) Spearman rho per subject, Wilcoxon vs 0; (3) paired anterior-temporal vs ventral-stream mean."""
import sys
import numpy as np
from scipy import stats
z = np.load(sys.argv[1])
rois = list(z["roi_names"])
ORDER = ["vwfa_v1", "vwfa_pit", "vwfa_ffc", "vwfa_vwfa", "vwfa_pstg", "vwfa_pmtg", "vwfa_astg", "vwfa_amtg",
         "L_TGd_ROI-lh", "L_TGv_ROI-lh"]
L = np.log(np.stack([z[c] for c in ("WORD_F1", "WORD_F2", "NONWORD_F1", "NONWORD_F2")]))
lex = (0.5 * ((L[0] - L[2]) + (L[1] - L[3]))).mean(-1)[:, [rois.index(r) for r in ORDER]]  # sub x 10
n = lex.shape[0]
rank = np.arange(1, 11)
slopes = np.array([np.polyfit(rank, s, 1)[0] for s in lex])
r = stats.ttest_1samp(slopes, 0)
print(f"N = {n}")
print(f"(1) slope per ROI step: mean {slopes.mean():+.4f}, dz = {slopes.mean()/slopes.std(ddof=1):.2f}, "
      f"t({n-1}) = {r.statistic:.2f}, p = {r.pvalue:.4f}; {int((slopes > 0).sum())}/{n} subjects positive")
rho = np.array([stats.spearmanr(rank, s)[0] for s in lex])
w = stats.wilcoxon(rho)
print(f"(2) Spearman rho per subject: median {np.median(rho):+.2f}, Wilcoxon p = {w.pvalue:.4f}")
vent, post, ant = lex[:, 0:4].mean(1), lex[:, 4:6].mean(1), lex[:, 6:10].mean(1)
for a, b, lab in ((ant, vent, "anterior temporal - ventral"), (post, vent, "posterior temporal - ventral"),
                  (ant, post, "anterior - posterior temporal")):
    d = a - b; t = stats.ttest_rel(a, b)
    print(f"(3) {lab:30s}: diff {d.mean():+.3f}, dz = {d.mean()/d.std(ddof=1):.2f}, t({n-1}) = {t.statistic:.2f}, p = {t.pvalue:.4f}")
