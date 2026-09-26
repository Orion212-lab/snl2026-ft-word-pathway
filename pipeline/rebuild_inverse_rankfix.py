"""Rebuild inverse operators whose noise-covariance rank is overestimated.

Diagnosis (2026-09-26): for sub-06/07/21 the task-syntaxIM inverse keeps one numerically-zero
noise-covariance eigenvalue (~1e-38, an SSS null-space direction) inside its rank. The whitener then
scales that direction by ~1e19, the dSPM kernel gain is ~1e6 x normal, and the source maps are
distorted (not just rescaled).

For every requested subject this script:
  1. reads <sub>_ses-01_task-syntaxIM_inv.fif and flags it as broken when the smallest retained
     eigenvalue is < 1e-8 x the median retained eigenvalue;
  2. if broken: rebuilds the operator from the same forward model and the same noise covariance
     (taken from the inverse itself), with rank = mne.compute_rank(cov, tol='auto') and the original
     loose / depth (read back from the operator's orientation / depth priors);
  3. keeps the original as <name>_inv.fif.bak-rankbug and writes the fixed operator in place, so every
     downstream script picks it up unchanged;
  4. prints the dSPM kernel gain (median peak |dSPM| for 200 unit dipoles) before and after.

Usage: python rebuild_inverse_rankfix.py --bids-root <root> --subjects 06 07 21 [--dry-run] [--out-dir DIR]
"""
import argparse
import glob
import shutil
from pathlib import Path

import mne
import numpy as np

mne.set_log_level("ERROR")


def kernel_gain(inv, fwd_fixed, cols, info):
    ev = mne.EvokedArray(fwd_fixed["sol"]["data"][:, cols], mne.pick_info(
        info, [info["ch_names"].index(c) for c in fwd_fixed["sol"]["row_names"]]))
    stc = mne.minimum_norm.apply_inverse(ev, inv, lambda2=1 / 9, method="dSPM", pick_ori=None)
    return float(np.median(stc.data.max(0)))


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--bids-root", type=Path, required=True)
    p.add_argument("--subjects", nargs="+", required=True)
    p.add_argument("--session", default="01")
    p.add_argument("--task", default="syntaxIM")
    p.add_argument("--dry-run", action="store_true")
    p.add_argument("--out-dir", type=Path, default=None,
                   help="write fixed operators here instead of in place (testing)")
    a = p.parse_args()
    for sub in a.subjects:
        meg = a.bids_root / "derivatives" / "mne-bids-pipeline" / f"sub-{sub}" / f"ses-{a.session}" / "meg"
        stem = f"sub-{sub}_ses-{a.session}_task-{a.task}"
        inv_path = meg / f"{stem}_inv.fif"
        inv = mne.minimum_norm.read_inverse_operator(inv_path)
        eig = inv["noise_cov"]["eig"]
        kept = eig[eig > 0]
        ratio = kept.min() / np.median(kept)
        broken = ratio < 1e-8
        print(f"sub-{sub}: retained rank {kept.size}, min/median eigenvalue {ratio:.1e} -> "
              f"{'BROKEN' if broken else 'ok'}", flush=True)
        if not broken or a.dry_run:
            continue

        info_files = sorted(glob.glob(str(meg / f"{stem}_proc-clean*_epo.fif"))) or \
            sorted(glob.glob(str(meg / f"{stem}_ave.fif")))
        info = mne.io.read_info(info_files[0])
        fwd = mne.read_forward_solution(meg / f"{stem}_fwd.fif")
        fwd = mne.pick_channels_forward(fwd, inv["info"]["ch_names"], ordered=True)
        nc = inv["noise_cov"]
        cov = mne.Covariance(nc["data"], nc["names"], nc["bads"], nc["projs"], nc["nfree"])
        info = mne.pick_info(info, [info["ch_names"].index(c) for c in nc["names"]])
        loose = float(np.round(np.min(inv["orient_prior"]["data"]), 3)) if inv["orient_prior"] is not None else 1.0
        dp = inv["depth_prior"]["data"] if inv["depth_prior"] is not None else None
        depth = None if dp is None or np.allclose(dp, dp[0]) else 0.8
        if depth is not None:
            raise RuntimeError(f"sub-{sub}: non-flat depth prior, set --depth explicitly before rebuilding")
        rank = mne.compute_rank(cov, rank=None, info=info, tol="auto")
        new = mne.minimum_norm.make_inverse_operator(info, fwd, cov, loose=loose, depth=depth, rank=rank)

        ff = mne.convert_forward_solution(fwd, surf_ori=True, force_fixed=True, use_cps=True)
        cols = np.linspace(0, ff["sol"]["data"].shape[1] - 1, 200).astype(int)
        g0, g1 = kernel_gain(inv, ff, cols, info), kernel_gain(new, ff, cols, info)
        print(f"    rebuilt with rank {rank}, loose {loose}, depth {depth}: kernel gain {g0:.3g} -> {g1:.3g}",
              flush=True)
        if a.out_dir is not None:
            a.out_dir.mkdir(parents=True, exist_ok=True)
            mne.minimum_norm.write_inverse_operator(a.out_dir / inv_path.name, new, overwrite=True)
            print(f"    wrote {a.out_dir / inv_path.name} (test mode, original untouched)", flush=True)
            continue
        bak = Path(str(inv_path) + ".bak-rankbug")
        if not bak.exists():
            shutil.copy2(inv_path, bak)
        mne.minimum_norm.write_inverse_operator(inv_path, new, overwrite=True)
        print(f"    wrote {inv_path.name} (original kept as {bak.name})", flush=True)


if __name__ == "__main__":
    main()
