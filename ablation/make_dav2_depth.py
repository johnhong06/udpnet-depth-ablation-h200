"""
make_dav2_depth.py -- step 2 (GPU machine). DAv2-Large prior for every HAZY image of the ablation splits,
written exactly like the HazyDet priors (the HazyDet prior script): HF Depth-Anything-V2-Large in fp16,
prediction resized bicubically to the full image size, per-image min-max, uint8, near = bright.

DAv2 sees the hazy image (what UDPNet gets at test time), so each beta level has its own prior.

    <root>/depth_hazy/dav2/<id>_b<beta>.png

  python make_dav2_depth.py --haze /data/haze_asm_v2 --root /data/udpnet_haze_asm_v2
Resumable: existing outputs are skipped. ~0.1 s/image on an RTX 5090 -> ~1 h for ~35k images.
"""
import argparse
import json
import os
import time

import numpy as np
import torch
from PIL import Image
from transformers import AutoImageProcessor, AutoModelForDepthEstimation

MODEL_ID = "depth-anything/Depth-Anything-V2-Large-hf"


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--haze", required=True, help="copy of <AerialMetric>/haze_asm_v2")
    ap.add_argument("--root", required=True, help="copy of <AerialMetric>/udpnet_haze_asm_v2")
    a = ap.parse_args()

    S = json.load(open(os.path.join(a.root, "splits.json")))
    frames = {e["id"]: e for v in S["splits"].values() for e in v}            # unique frames
    jobs = [(os.path.join(a.haze, e["src"], "hazy", f'{e["tag"]}_b{b}.png'),
             os.path.join(a.root, "depth_hazy", "dav2", f"{i}_b{b}.png"))
            for i, e in sorted(frames.items()) for b in S["betas"]]
    todo = [j for j in jobs if not os.path.exists(j[1])]
    os.makedirs(os.path.join(a.root, "depth_hazy", "dav2"), exist_ok=True)
    print(f"{len(jobs)} hazy images, {len(todo)} to do", flush=True)

    dev = "cuda"
    proc = AutoImageProcessor.from_pretrained(MODEL_ID)
    model = AutoModelForDepthEstimation.from_pretrained(MODEL_ID, dtype=torch.float16).to(dev).eval()
    t0 = time.time()
    for k, (src, dst) in enumerate(todo, 1):
        img = Image.open(src).convert("RGB")
        x = proc(images=img, return_tensors="pt").to(dev)
        x["pixel_values"] = x["pixel_values"].half()
        with torch.no_grad():
            p = model(**x).predicted_depth
        d = torch.nn.functional.interpolate(p[:, None].float(), size=(img.height, img.width),
                                            mode="bicubic", align_corners=False)[0, 0]
        d = (d - d.min()) / (d.max() - d.min() + 1e-8) * 255.0
        Image.fromarray(d.clamp(0, 255).cpu().numpy().astype(np.uint8), mode="L").save(dst)
        if k % 500 == 0:
            el = time.time() - t0
            print(f"{k}/{len(todo)}  {el / k:.3f}s/img  eta {(len(todo) - k) * el / k / 60:.0f} min", flush=True)
    print(f"done: {len(todo)} written, {time.time() - t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
