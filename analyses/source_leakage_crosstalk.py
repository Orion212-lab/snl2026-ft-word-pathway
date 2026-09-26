"""Analysis 1b: ROI-to-ROI leakage (point-spread) of the block-3 dSPM pipeline.

For each subject: every native source vertex inside ROI j gets a unit normal-oriented dipole
(one at a time, via its forward column); the dSPM inverse used for block 3 (task-syntaxIM_inv.fif,
lambda2 = 1/9, free-orientation norm) is applied, the estimate is morphed to fsaverage and the
mean |dSPM| is read in each of the 10 fsaverage ROIs, exactly as in the pipeline.
Leak[i, j] = mean estimate in ROI i for sources in ROI j, divided by the estimate in ROI j itself.

Usage: BIDS_ROOT=<root> python source_leakage_crosstalk.py <out.npz> sub [sub ...]
"""
import os
import sys
import time
from pathlib import Path

import mne
import numpy as np

mne.set_log_level("ERROR")
BIDS = Path(os.environ["BIDS_ROOT"])  # MNE-BIDS root with derivatives/mne-bids-pipeline and derivatives/freesurfer
DER = BIDS / "derivatives" / "mne-bids-pipeline"
SUBJECTS_DIR = BIDS / "derivatives" / "freesurfer"
LAMBDA2 = 1.0 / 9.0
ROIS = [("V1", ["L_V1_ROI-lh"]), ("Post. IT", ["L_PIT_ROI-lh"]), ("FFC", ["L_FFC_ROI-lh"]),
        ("VWFA", ["VWFA_MNI-lh"]), ("Post. STG", ["L_STSdp_ROI-lh"]), ("Post. MTG", ["L_TPOJ2_ROI-lh"]),
        ("Ant. STG", ["L_STGa_ROI-lh"]), ("Ant. MTG", ["L_STSva_ROI-lh"]), ("TP dorsal", ["L_TGd_ROI-lh"]),
        ("TP ventral", ["L_TGv_ROI-lh"])]


def fsaverage_labels():
    hcp = {l.name: l for l in mne.read_labels_from_annot("fsaverage", parc="HCPMMP1", hemi="lh",
                                                         subjects_dir=SUBJECTS_DIR)}
    coords, _ = mne.surface.read_surface(str(SUBJECTS_DIR / "fsaverage" / "surf" / "lh.white"))
    v = np.arange(coords.shape[0])
    mni = mne.vertex_to_mni(v, hemis=0, subject="fsaverage", subjects_dir=SUBJECTS_DIR)
    hcp["VWFA_MNI-lh"] = mne.Label(v[np.linalg.norm(mni - (-43.0, -54.0, -12.0), axis=1) <= 8.0],
                                   hemi="lh", subject="fsaverage", name="VWFA_MNI-lh")
    out = []
    for _, names in ROIS:
        lab = hcp[names[0]]
        for n in names[1:]:
            lab = lab + hcp[n]
        out.append(lab)
    return out


def run_subject(sub, fs_labels):
    t0 = time.time()
    meg = DER / f"sub-{sub}" / "ses-01" / "meg"
    inv = mne.minimum_norm.read_inverse_operator(meg / f"sub-{sub}_ses-01_task-syntaxIM_inv.fif")
    fwd = mne.read_forward_solution(meg / f"sub-{sub}_ses-01_task-syntaxIM_fwd.fif")
    fwd = mne.convert_forward_solution(fwd, surf_ori=True, force_fixed=True, use_cps=True)
    fwd = mne.pick_channels_forward(fwd, inv["info"]["ch_names"], ordered=True)
    fs_sub = inv["src"][0].get("subject_his_id") or f"sub-{sub}"
    fs_orig = fs_sub
    if not (SUBJECTS_DIR / fs_sub / "surf" / "lh.white").exists():
        print(f"sub-{sub}: inverse src names FreeSurfer subject '{fs_sub}' (missing) -> using 'sub-{sub}'", flush=True)
        fs_sub = f"sub-{sub}"
    lh_vert = fwd["src"][0]["vertno"]
    native = mne.morph_labels(fs_labels, subject_to=fs_sub, subject_from="fsaverage",
                              subjects_dir=SUBJECTS_DIR)
    cols, owner = [], []
    for j, lab in enumerate(native):
        idx = np.where(np.isin(lh_vert, lab.vertices))[0]  # lh columns come first in fwd
        cols.append(idx)
        owner += [j] * idx.size
    allcols = np.concatenate(cols)
    names = fwd["sol"]["row_names"]
    types = [mne.channel_type(inv["info"], inv["info"]["ch_names"].index(c)) for c in names]
    info = mne.create_info(names, 1000.0, types)
    ev = mne.EvokedArray(fwd["sol"]["data"][:, allcols], info, tmin=0.0)
    stc = mne.minimum_norm.apply_inverse(ev, inv, lambda2=LAMBDA2, method="dSPM", pick_ori=None)
    cache = SUBJECTS_DIR / "morph-maps" / f"fsaverage-{fs_sub}-surf-ico5"
    stc.subject = fs_sub
    morph = None
    cache0 = SUBJECTS_DIR / "morph-maps" / f"fsaverage-{fs_orig}-surf-ico5"
    for c in (cache, Path(str(cache) + "-morph.h5"), Path(str(cache0) + "-morph.h5")):
        if c.exists():
            m = mne.read_source_morph(c)
            if all(np.array_equal(a, b) for a, b in zip(m.vertices_from, stc.vertices)):
                morph = m
                break
    try:
        stc_fs = morph.apply(stc)
    except Exception:  # missing or stale cache (built for another source space)
        morph = mne.compute_source_morph(stc, subject_from=fs_sub, subject_to="fsaverage",
                                         subjects_dir=SUBJECTS_DIR)
        stc_fs = morph.apply(stc)
    lh_data = stc_fs.data[:len(stc_fs.vertices[0])]
    est = np.stack([lh_data[np.isin(stc_fs.vertices[0], l.vertices)].mean(0) for l in fs_labels])  # roi x src
    owner = np.array(owner)
    raw = np.stack([est[:, owner == j].mean(1) for j in range(len(ROIS))], axis=1)  # target i x source j
    leak = raw / np.diag(raw)[None, :]
    print(f"sub-{sub}: {allcols.size} source vertices, {time.time() - t0:.0f} s", flush=True)
    return raw, leak


if __name__ == "__main__":
    out, subs = sys.argv[1], sys.argv[2:]
    fs_labels = fsaverage_labels()
    R, L, done = [], [], []
    for s in subs:
        try:
            r, l = run_subject(s, fs_labels)
        except Exception as e:  # keep going; report which subjects failed
            print(f"sub-{s}: FAILED {type(e).__name__}: {e}", flush=True)
            continue
        R.append(r); L.append(l); done.append(s)
        np.savez(out, raw=np.array(R), leak=np.array(L), subjects=np.array(done),
                 rois=np.array([n for n, _ in ROIS]))
    Lm = np.array(L).mean(0)
    names = [n for n, _ in ROIS]
    print("\nmean leakage (rows = where it shows up, cols = true source), N =", len(done))
    print(" " * 11 + "".join(f"{n[:8]:>9s}" for n in names))
    for i, n in enumerate(names):
        print(f"{n:11s}" + "".join(f"{Lm[i, j]:9.2f}" for j in range(len(names))))
