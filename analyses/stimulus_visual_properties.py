"""Point 3: low-level visual properties of the one-word stimuli, Word vs Non-word.

The task font is "Cousine Nerd Font Mono", bold (freqtag_spec). Cousine is metric-compatible with
Courier New, so Courier New Bold (courbd.ttf) is used as a proxy for rendering. Per string: letters,
ascender/descender letters, letter area (lit pixels of the rendered string), horizontal extent, vertical extent.
"""
import csv
import sys
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from scipy import stats

STIM = sys.argv[1].rstrip("/") + "/"  # folder with {even,odd}_one_word_stimuli.csv
font = ImageFont.truetype(sys.argv[2], 200)  # path to Courier New Bold (courbd.ttf), metric twin of Cousine
ASC, DESC = set("bdfhklt"), set("gjpqy")


def props(w):
    im = Image.new("L", (1400, 400), 0)
    ImageDraw.Draw(im).text((50, 50), w, font=font, fill=255)
    a = np.asarray(im) > 127
    ys, xs = np.nonzero(a)
    return dict(letters=len(w), asc=sum(c in ASC for c in w), desc=sum(c in DESC for c in w),
                letter_area_px=int(a.sum()), width=int(xs.max() - xs.min() + 1), height=int(ys.max() - ys.min() + 1))


items = []
for lst in ("even", "odd"):
    rows = list(csv.DictReader(open(STIM + f"{lst}_one_word_stimuli.csv")))
    P = {c: [props(r["w1"]) for r in rows if r["condition"] == c] for c in ("word", "non-word")}
    items += [dict(list=lst, string=r["w1"], condition=r["condition"], **props(r["w1"])) for r in rows]
    print(f"\n=== {lst} list: {len(P['word'])} words, {len(P['non-word'])} non-words")
    for k in ("letters", "asc", "desc", "letter_area_px", "width", "height"):
        w = np.array([p[k] for p in P["word"]]); n = np.array([p[k] for p in P["non-word"]])
        u = stats.mannwhitneyu(w, n)
        d = (n.mean() - w.mean()) / np.sqrt((w.var(ddof=1) + n.var(ddof=1)) / 2)
        print(f"  {k:8s} word {w.mean():8.1f}  non-word {n.mean():8.1f}  NW/W {n.mean()/w.mean():.2f}  "
              f"d(NW-W) = {d:+.2f}  Mann-Whitney p = {u.pvalue:.2g}")
    print("  3-letter words:", sum(p['letters'] == 3 for p in P['word']), "/", len(P['word']))

if len(sys.argv) > 3:  # optional per-item table
    with open(sys.argv[3], "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(items[0]))
        w.writeheader()
        w.writerows(items)
