# Cohort per poster panel

**Recorded:** 30 participants (sub-01 … sub-30). All were native English speakers and right-handed.

## Excluded from every analysis

| Subject | Reason |
|---|---|
| sub-01 | No flicker on the photodiode during the one-word task |
| sub-09 | Malformed blocks, no usable one-word triggers |
| sub-17 | Unreliable photodiode match |

## Panels

| Panel | N | Subjects | Notes |
|---|---|---|---|
| Block 3 · tag spectra, harmonic sums, evoked violins (sensors) | 20 | 04 06 07 08 11 12 13 14 16 18 19 21 22 23 25 26 27 28 29 30 | Same per-word epochs for both measures (PTP rejection over the whole word, −0.2–2.83 s, ≥ 8 equalized epochs/condition). Not retained from the 27-subject sensor roster: 02, 03, 05, 10, 15, 20, 24 |
| Block 1 · sensor cluster test, panels a–b | 26 | 03 04 05 06 07 08 10 11 12 13 14 15 16 18 19 20 21 22 23 24 25 26 27 28 29 30 | sub-02: every per-word epoch PTP-rejected |
| Block 1 · topomaps (grand averages) | 25 | as above, without 10 | |
| Block 1 · eye-tracker control | 25 | as above, without 07 | sub-07: eye-tracker signal lost 96 % of the time |
| Block 2 · source ROIs | 22 | 03 04 05 06 07 08 12 13 14 15 16 18 19 21 22 23 25 26 27 28 29 30 | No MRI: 10, 11. Source artifact: 20, 24. Inverse rank corrected for 06, 07, 21 |
| Block 4 · tag vs evoked (sources + sensors) | 17 | 04 06 07 08 12 13 14 18 19 21 22 23 25 26 28 29 30 | Source run of the same pipeline; 11 has no MRI; 16 and 27 were not retained by the source run |

## Trigger anomalies

These subjects do not have the expected 120 one-word markers:

| Subject | Markers |
|---|---|
| sub-10 | 60 |
| sub-11 | 444 |
| sub-12 | 165 |
| sub-13 | 60 |
| sub-14 | 53 |

**Robustness without them:**
- Block 1, sensors (N = 21): cluster p = .0002.
- Block 2, sources (N = 19): slope t = 4.92, all 6 temporal ROIs p_FDR < .05.
- The late ventral (FFC/VWFA) word-locked effect is not robust to this exclusion.
