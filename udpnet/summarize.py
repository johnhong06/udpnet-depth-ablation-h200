"""Collect every test.py --json_out result into one table.

    python summarize.py results/summary
"""
import glob
import json
import os
import sys

d = sys.argv[1] if len(sys.argv) > 1 else "results/summary"
rows = []
for p in sorted(glob.glob(os.path.join(d, "*.json"))):
    r = json.load(open(p))
    model, _, data = os.path.basename(p)[:-5].partition("__")
    rows.append((model, data, r))

if not rows:
    sys.exit(f"no result json in {d}")

print(f"\n{'trained on':16s} {'tested on':18s} {'n':>5s} | {'input PSNR':>10s} "
      f"{'UDPNet PSNR':>11s} {'gain':>7s} | {'input SSIM':>10s} {'UDPNet SSIM':>11s}")
print("-" * 98)
for model, data, r in rows:
    print(f"{model:16s} {data:18s} {r['n']:5d} | {r['in_psnr']:10.3f} {r['psnr']:11.3f} "
          f"{r['gain_psnr']:+7.3f} | {r['in_ssim']:10.4f} {r['ssim']:11.4f}")
print("\npaper (FSNet+UDP, Haze4K test): PSNR 35.31, SSIM 0.99")
print("read 'gain' first: it is what dehazing added over leaving the image hazy.")
