"""
make_layout.py -- step 1 (any machine with the external drive; CPU).
Split haze_asm_v2 for the depth-prior ablation and write the GT depth prior once per frame.

The external drive is exFAT (no hard/symbolic links), so nothing is duplicated here. The UDPNet folders
(one per depth variant, with oblique repetition and one depth file per hazy name) are built later with
symlinks on the Linux training machine by link_layout.py.

Output (<out>, default <AerialMetric>/udpnet_haze_asm_v2):
    depth_frames/gt/<id>.png    GT depth prior, one per frame, aligned with haze_asm_v2/<src>/gt/<tag>.png
    splits.json                 per split: list of {id, tag, src, repeat}

  splits
    train         haze_asm_v2/train minus val frames; oblique frames (pitch < 70) get repeat = --oblique-repeat
    val           --val-frac of the train FRAMES (all 4 betas), stable md5 hash -> checkpoint selection
    test          haze_asm_v2/test, all 770 frames (official Oblique test = CSV "val" + 16 "NAN")
    test_heldout  test frames from places with NO training frames: Artsci, caliterra-output, polytech

  <id>  = <scene>--<frame>   (no "_": UDPNet finds the gt as hazy_name.split("_")[0])
  <tag> = <scene>__<frame>   (haze_asm_v2 file names)

  GT depth prior = same format as UDPNet's DAv2 prior (depth2l): inverse depth 1/d, per-image min-max
  to 0..255 over pixels with valid depth, near = bright; pixels without depth (sky / holes) = 0 (far).
  The depth is re-loaded and put through exactly the generator's resize + 90-degree "upright" rotation,
  and every frame is checked by comparing the re-derived RGB with the saved gt PNG (mean abs diff <= 1/255).

  python make_layout.py                       # all frames, 6 worker processes (~15 min)
  python make_layout.py --limit 20 --out <AerialMetric>/_udp_smoke
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from multiprocessing import Pool

import numpy as np
from PIL import Image

AM = os.environ.get("AERIALMETRIC_ROOT", "AerialMetric")   # root of the AerialMetric download
HAZE = AM + "/haze_asm_v2"
sys.path.insert(0, AM + "/haze_asm/code")
import aerialmetric_asm_cam_v2 as G          # noqa: E402
import aerialmetric_asm_cam_v2_train as T    # noqa: E402

HELDOUT = {"Artsci", "caliterra-output", "polytech"}
_META = None


def frame_id(scene, stem):
    i = f"{scene}--{stem}"
    assert "_" not in i, i
    return i


def aligned(rec, src):
    """RGB, depth (m) and valid mask in the frame of the saved gt PNG."""
    scene, stem = rec["scene"], rec["stem"]
    if src == "train":
        fdir = f"{AM}/Train-Oblique-dense/TrainingData_Final_MoGe-All/{scene}/{stem}"
        rgb = G.load_rgb(fdir + "/image.jpg")
        H, W = rgb.shape[:2]
        d, v = G.load_depth(fdir + "/depth.npy", W, H)
        rgb, d, v, *_ = T.make_upright_fxfy(rgb, d, v, 1.0, 1.0, W / 2, H / 2, rec["roll_fit_stored_deg"])
    else:
        rd, dd = f"{AM}/Oblique/{scene}/rgbs", f"{AM}/Oblique/{scene}/depth"
        f = next(x for x in os.listdir(rd) if os.path.splitext(x)[0] == stem)
        rgb = G.load_rgb(os.path.join(rd, f))
        H, W = rgb.shape[:2]
        d, v = G.load_depth(G.find_depth(dd, stem), W, H)
        _, cx, cy, _, roll = G.camera_params(_META.get(scene, stem), W, H)
        rgb, d, v, _ = G.resize_pair(rgb, d, v, 1920)
        rgb, d, v, *_ = G.make_upright(rgb, d, v, cx, cy, roll)
    return rgb, d, v


def gt_prior(d, v):
    out = np.zeros(d.shape, np.uint8)
    if v.any():
        inv = 1.0 / np.maximum(d[v], 1e-3)
        lo, hi = inv.min(), inv.max()
        out[v] = np.round((inv - lo) / max(hi - lo, 1e-12) * 255.0).astype(np.uint8)
    return out


def work(job):
    rec, src, out = job
    i, tag = frame_id(rec["scene"], rec["stem"]), f'{rec["scene"]}__{rec["stem"]}'
    dst = f"{out}/depth_frames/gt/{i}.png"
    try:
        if os.path.exists(dst):
            return i, None
        rgb, d, v = aligned(rec, src)
        saved = np.asarray(Image.open(f"{HAZE}/{src}/gt/{tag}.png"), np.float32) / 255.0
        if saved.shape != rgb.shape:
            return i, f"shape mismatch {saved.shape} vs {rgb.shape}"
        mad = float(np.abs(saved - rgb).mean())
        if mad > 1.0 / 255:
            return i, f"alignment check failed (mean abs diff {mad * 255:.2f}/255)"
        Image.fromarray(gt_prior(d, v)).save(dst)
        return i, None
    except Exception as e:  # noqa: BLE001
        return i, f"{type(e).__name__}: {e}"


def init_worker():
    global _META
    _META = G.Meta(AM + "/final_merged.csv")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", default=AM + "/udpnet_haze_asm_v2")
    ap.add_argument("--val-frac", type=float, default=0.05)
    ap.add_argument("--oblique-repeat", type=int, default=3,
                    help="oblique (pitch < 70) train frames are 25%% of train vs 58%% of test; 3 -> ~50%%. 1 = off")
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--limit", type=int, default=0, help="frames per source split (smoke test)")
    a = ap.parse_args()

    tr = json.load(open(HAZE + "/train/index.json"))
    te = json.load(open(HAZE + "/test/index.json"))
    if a.limit:
        tr, te = tr[:: max(1, len(tr) // a.limit)][: a.limit], te[:: max(1, len(te) // a.limit)][: a.limit]

    def is_val(r):
        return int(hashlib.md5(frame_id(r["scene"], r["stem"]).encode()).hexdigest(), 16) % 10000 < a.val_frac * 1e4

    splits = {"train": [], "val": [], "test": [], "test_heldout": []}
    for r in tr:
        e = dict(id=frame_id(r["scene"], r["stem"]), tag=f'{r["scene"]}__{r["stem"]}', src="train",
                 pitch=r["pitch_deg"], repeat=1)
        if is_val(r):
            splits["val"].append(e)
        else:
            e["repeat"] = a.oblique_repeat if r["pitch_deg"] < 70 else 1
            splits["train"].append(e)
    for r in te:
        e = dict(id=frame_id(r["scene"], r["stem"]), tag=f'{r["scene"]}__{r["stem"]}', src="test",
                 pitch=r["pitch_deg"], repeat=1)
        splits["test"].append(e)
        if r["scene"] in HELDOUT:
            splits["test_heldout"].append(e)

    os.makedirs(f"{a.out}/depth_frames/gt", exist_ok=True)
    jobs = [(r, "train", a.out) for r in tr] + [(r, "test", a.out) for r in te]
    print(f"{len(jobs)} frames, {a.workers} workers -> {a.out}", flush=True)
    failed = {}
    with Pool(a.workers, initializer=init_worker) as pool:
        for k, (i, err) in enumerate(pool.imap_unordered(work, jobs, chunksize=8), 1):
            if err:
                failed[i] = err
                print(f"  FAIL {i}: {err}", flush=True)
            if k % 500 == 0:
                print(f"{k}/{len(jobs)}", flush=True)

    for s in splits:
        splits[s] = [e for e in splits[s] if e["id"] not in failed]
    summary = {s: dict(frames=len(v), hazy_images=4 * sum(e["repeat"] for e in v),
                       oblique_frames=sum(e["pitch"] < 70 for e in v)) for s, v in splits.items()}
    json.dump(dict(summary=summary, betas=["1", "2.5", "6", "15"], val_frac=a.val_frac,
                   oblique_repeat=a.oblique_repeat, heldout_scenes=sorted(HELDOUT), failed=failed,
                   splits=splits), open(f"{a.out}/splits.json", "w"), indent=1)
    print(json.dumps(summary, indent=1))
    print(f"failed frames: {len(failed)}")


if __name__ == "__main__":
    main()
