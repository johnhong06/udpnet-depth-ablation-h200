"""
link_layout.py -- step 3 (Linux training machine). Build one UDPNet data folder per depth variant out of
symlinks (no image is copied):

    <root>/udpnet_<variant>/<split>/hazy/<id>_b<beta>[_r<k>].png  -> <haze>/<src>/hazy/<tag>_b<beta>.png
    <root>/udpnet_<variant>/<split>/gt/<id>.png                    -> <haze>/<src>/gt/<tag>.png
    <root>/udpnet_<variant>/<split>/depth2l/<same stem as hazy>    -> depth prior of that variant
        dav2   <root>/depth_hazy/dav2/<id>_b<beta>.png      (make_dav2_depth.py; per hazy image)
        gt     <root>/depth_frames/gt/<id>.png              (make_layout.py;     per frame)
        const  <root>/depth_frames/const/<W>x<H>.png        (written here, 128 everywhere)

Oblique train frames appear `repeat` times (suffix _r1, _r2 ...): same files, sampled more often.

  python link_layout.py --haze /data/haze_asm_v2 --root /data/udpnet_haze_asm_v2 [--variants dav2 gt const]
                        [--mode symlink|hardlink|copy]
--mode: symlink (default, no extra space); hardlink (same file system, for storage without symlinks);
        copy (any file system; the gt variant then takes ~75 GB instead of ~55 GB because of the repeats).
"""
import argparse
import json
import os
import shutil

import numpy as np
from PIL import Image


MODE = "symlink"


def sym(src, dst):
    if not os.path.exists(src):
        raise FileNotFoundError(src)
    if os.path.lexists(dst):
        os.remove(dst)
    if MODE == "symlink":
        os.symlink(os.path.abspath(src), dst)
    elif MODE == "hardlink":
        os.link(os.path.realpath(src), dst)
    else:
        shutil.copyfile(src, dst)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--haze", required=True)
    ap.add_argument("--root", required=True)
    ap.add_argument("--variants", nargs="+", default=["dav2", "gt", "const"])
    ap.add_argument("--mode", choices=["symlink", "hardlink", "copy"], default="symlink")
    a = ap.parse_args()
    global MODE
    MODE = a.mode
    S = json.load(open(os.path.join(a.root, "splits.json")))
    cdir = os.path.join(a.root, "depth_frames", "const")
    os.makedirs(cdir, exist_ok=True)

    for v in a.variants:
        for split, entries in S["splits"].items():
            base = os.path.join(a.root, f"udpnet_{v}", split)
            for sub in ("hazy", "gt", "depth2l"):
                os.makedirs(os.path.join(base, sub), exist_ok=True)
            n = 0
            for e in entries:
                i, tag, src = e["id"], e["tag"], e["src"]
                gt = os.path.join(a.haze, src, "gt", f"{tag}.png")
                sym(gt, os.path.join(base, "gt", f"{i}.png"))
                if v == "gt":
                    prior = os.path.join(a.root, "depth_frames", "gt", f"{i}.png")
                elif v == "const":
                    W, H = Image.open(gt).size
                    prior = os.path.join(cdir, f"{W}x{H}.png")
                    if not os.path.exists(prior):
                        Image.fromarray(np.full((H, W), 128, np.uint8)).save(prior)
                for b in S["betas"]:
                    if v == "dav2":
                        prior = os.path.join(a.root, "depth_hazy", "dav2", f"{i}_b{b}.png")
                    for r in range(e["repeat"]):
                        stem = f"{i}_b{b}" + (f"_r{r}" if r else "")
                        sym(os.path.join(a.haze, src, "hazy", f"{tag}_b{b}.png"),
                            os.path.join(base, "hazy", f"{stem}.png"))
                        sym(prior, os.path.join(base, "depth2l", f"{stem}.png"))
                        n += 1
            print(f"udpnet_{v}/{split}: {len(entries)} frames, {n} hazy images", flush=True)


if __name__ == "__main__":
    main()
