# Characterizing word processing pathway using frequency tagging

**SNL 2026 · Poster F52 · Session F, Friday 2 October, 2:45–4:45 pm**
Kishen Senziani, Berk Gercek, Nina Kazanina — University of Geneva, Faculty of Medicine, Department of Basic Neurosciences

This repository holds the analysis code, supplementary figures and group-level results behind the poster.
The MEG / MRI data are not shared here.

## Take-home

A 6–7 Hz flicker tag tracks the **visual input**, not wordness. The **word-locked response** (≥ 250 ms) separates words from consonant strings. The difference is largest in anterior temporal cortex, where it is sustained across words.

| | Tag (frequency tagging) | Word-locked (evoked) response |
|---|---|---|
| Left-temporal sensors | Words ≈ consonant strings (harmonics 1–4: equivalence \|dz\| < .5, TOST p = .03, BF01 = 4.2) | Word > Non-word, 250–600 ms (dz = 1.04) |
| Occipital sensors | Consonant strings > words (dz = −.88, p = .001), matching their ~10 % more ink and 20 % greater height | n.s. |
| Direct test (same epochs) | — | Evoked − tag Δdz = 1.02 [.29, 1.92], p = .01 |
| Source (dSPM, N = 22) | tag at zero along the pathway | Word > Non-word in all 6 temporal ROIs (p_FDR < .05), maximal anteriorly (slope t(21) = 4.81) |

**Update relative to the submitted abstract.** Two changes reversed its conclusions: a fixation baseline (instead of the previous word's tail) and equal trial counts per condition.
- The tag effect is now null.
- The source effect is now Word > Non-word.

## Aims and the two readouts

- **Aim 1** — does MEG (sensors, then sources) recover the V1 → vOT → temporal sequence? Poster blocks 1–2.
- **Aim 2** — does the frequency-tagged response differ between words and consonant strings? Poster blocks 3–4.

The same recordings give two measures:

| | Tag response (FT) | Word-locked response (evoked) |
|---|---|---|
| What | Power at the flicker rate (6 or 7.06 Hz, plus harmonics) relative to neighbouring frequencies (SNR) | MEG field averaged across words, time-locked to each word onset |
| Follows | the flicker, cycle after cycle | the one-off response to each new word (every 2.83 s) |
| Word vs Non-word | SNR difference | amplitude difference, 250–600 ms |

## Design (one-word task)
- **Stimuli:** 60 words (e.g. *wary*) and 60 consonant strings (e.g. *hkwj*). Each string is shown for 2.83 s and flickers at 6.00 or 7.06 Hz on a 240 Hz projector. The photodiode measured 5.965 and 7.024 Hz.
- **Blocks:** mini-blocks of 10 strings from the same condition (28.33 s), each preceded by a 2 s fixation, with a catch query after each block.
- **Recordings:** 306-channel MEG (MaxFilter with movement compensation, no ICA), eye tracker, individual T1 MRI. Source estimates use dSPM on HCP-MMP1 ROIs of the left hemisphere.

## Repository

| Folder | Content |
|---|---|
| `analyses/` | Group-level statistics run on derived per-subject files (spectra, ROI time courses, sensor field strength) |
| `pipeline/` | Scripts that need the MEG derivatives: inverse-operator rank fix, eye-tracker control |
| `figures/` | Scripts for the poster panels and the supplementary figures |
| `results/` | Text / JSON outputs of the analyses (group level) |
| `supplementary/` | Figures S1–S4 |
| `cohort.md` | Subjects in each panel and exclusion reasons |

Each script's docstring lists its inputs. Paths are passed as arguments or environment variables (`BIDS_ROOT`, `SUBJECTS_DIR`, `POSTER_OUT`). The per-subject inputs come from the lab's MNE-BIDS-Pipeline derivatives and are not included.

## Supplementary figures

- **S1 · Leakage.** ROI-to-ROI leakage of the dSPM pipeline (unit dipoles, median of 19 subjects).
  - The four anterior temporal ROIs are not separable (cross-talk ≈ .7), and neither are FFC and VWFA.
  - One anterior temporal source reproduces most of the block-2 (sources) profile.
- **S2 · Stimuli.** Low-level properties of the stimuli. Consonant strings are always 4 letters, have more ink and are taller, with 3–4× more descender letters.
- **S3 · Cascade windows.** Tag vs evoked Word > Non-word in the a-priori cascade windows (80–130, 150–225, 250–600 ms).
- **S4 · Block-2 (sources) gradient.**
  - It is not a ratio artifact: it is the same on raw dSPM differences.
  - Its anterior part is already present before word onset.
  - At 250–600 ms the effect is broad.

## Pitfalls we hit (and fixes)
1. **Tags off by one FFT bin.** The display ran 0.6 % slow, so SNR is read at the measured bin.
2. **Unequal trial counts faked an effect.** Counts are now equalized within subject.
3. **No gap between words.** The pre-stimulus window holds the previous word, so the fixation baseline is used instead.
4. **Noise-covariance rank.** After SSS, 3 subjects kept a ~1e-38 eigenvalue: the dSPM gain was ×10⁶ and the maps were distorted. The fix is `mne.compute_rank(cov, tol='auto')` (`pipeline/rebuild_inverse_rankfix.py`).
5. **Match the ink.** The occipital tag followed the low-level differences between stimuli.

## Limitations
- **Stimulus contrast.** Words vs consonant strings confounds lexicality with orthographic legality, pronounceability and ink; there are no pseudowords.
- **Eye movements.** Words drew more blinks (dz = .94) and saccades. The sensor effect holds in PTP-clean, blink- and saccade-free epochs (dz = 1.31, 20/21 subjects; `results/eye_control_group.txt`).
- **Sustained anterior temporal effect.** It is present before word onset, so it may reflect a block-level state (mini-blocks are condition-pure) rather than word-locked processing.
- **Ventral stream.** No whole-epoch difference in FFC/VWFA. A word-locked rise at 250–600 ms is not robust to excluding subjects with trigger anomalies.
- **Latency.** The latency ordering of the cascade is not significant.
- **Not yet analysed.** Catch-query accuracy per condition has not been scored, and the language localizer was not used for these ROIs.

## Software
MNE-Python 1.10.2, MNE-BIDS-Pipeline 1.10.1, FreeSurfer 7.1.2; see `requirements.txt` for the analysis scripts.

## Contact
Kishen Senziani — University of Geneva (NeuroLingo lab).
