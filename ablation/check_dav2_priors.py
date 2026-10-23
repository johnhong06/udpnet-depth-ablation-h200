"""
check_dav2_priors.py -- compare freshly generated DAv2 priors with reference statistics of the priors used for the
local dav2 run (same script, RTX PRO 4500). Detects a different model / preprocessing / orientation, not 1-level noise.

  python check_dav2_priors.py --make <depth_hazy/dav2 dir> --out dav2_reference_stats.json   # once, on the reference machine
  python check_dav2_priors.py --check <depth_hazy/dav2 dir> --ref dav2_reference_stats.json  # after generating elsewhere
Per image: mean, std, 5/50/95th percentile and an 8x8 grid of block means (uint8 levels).
"""
import argparse
import json
import os
import random
import sys

import numpy as np
from PIL import Image


def stats(path):
    a = np.asarray(Image.open(path), np.float32)
    h, w = a.shape
    g = a[: h - h % 8, : w - w % 8].reshape(8, (h - h % 8) // 8, 8, (w - w % 8) // 8).mean((1, 3))
    return dict(shape=[h, w], mean=round(float(a.mean()), 2), std=round(float(a.std()), 2),
                pct=[round(float(x), 1) for x in np.percentile(a, [5, 50, 95])], grid=[int(round(x)) for x in g.ravel()])


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--make"); ap.add_argument("--out")
    ap.add_argument("--check"); ap.add_argument("--ref")
    ap.add_argument("--n", type=int, default=200)
    ap.add_argument("--tol", type=float, default=2.0, help="max allowed mean |grid difference| per image (levels)")
    a = ap.parse_args()
    if a.make:
        names = sorted(os.listdir(a.make)); random.Random(0).shuffle(names)
        ref = {n: stats(os.path.join(a.make, n)) for n in sorted(names[: a.n])}
        json.dump(ref, open(a.out, "w"), separators=(",", ":"))
        print(f"wrote {len(ref)} reference images to {a.out}")
        return
    ref = json.load(open(a.ref))
    worst, bad, missing = [], 0, 0
    for n, r in ref.items():
        p = os.path.join(a.check, n)
        if not os.path.exists(p):
            missing += 1; continue
        s = stats(p)
        if s["shape"] != r["shape"]:
            bad += 1; worst.append((999.0, n)); continue
        d = float(np.abs(np.array(s["grid"], np.float32) - np.array(r["grid"], np.float32)).mean())
        worst.append((d, n)); bad += d > a.tol
    worst.sort(reverse=True)
    if not worst:
        print(f"[dav2 check] none of the {len(ref)} reference images found in {a.check}")
        print("[dav2 check] MISMATCH: priors differ from the local dav2 run")
        sys.exit(1)
    print(f"[dav2 check] {len(ref) - missing}/{len(ref)} reference images, mean |grid diff| "
          f"median {np.median([d for d, _ in worst]):.2f}, max {worst[0][0]:.2f} levels ({worst[0][1]}); "
          f"over tolerance {a.tol}: {bad}, missing {missing}")
    ok = bad == 0 and missing == 0
    print("[dav2 check] " + ("OK: priors match the local dav2 run" if ok else "MISMATCH: priors differ from the local dav2 run"))
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
