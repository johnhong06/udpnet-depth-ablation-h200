"""
Evaluate a UDPNet checkpoint on a split, and -- in the same loop, with the same
metric code -- the do-nothing baseline (hazy input vs GT). A dehazing PSNR can
only be read against that baseline.

  python test.py --data_dir /data/haze4k_udp  --test_model results/x/checkpoints/last.ckpt
  python test.py --data_dir /data/hazydet_udp --test_model ... --split test --json_out r.json

Metrics are the upstream ones: PSNR = 10 log10(1/MSE) on the clamped [0,1]
output; SSIM (pytorch_msssim) after average-pooling to ~256 on the short side.
"""
import argparse
import json
import os
import time

import torch
import torch.nn.functional as f
from pytorch_msssim import ssim
from torch.backends import cudnn
from torchvision.transforms import functional as F

from data import test_dataloader
from models.FSNet_UDPNet import build_net


def load_weights(model, path):
    """Accept a Lightning .ckpt ('state_dict', keys 'model.x') or a raw state
    dict / {'model': state_dict} as released weights often are."""
    obj = torch.load(path, map_location='cpu', weights_only=False)
    sd = obj
    if isinstance(obj, dict):
        for k in ('state_dict', 'model', 'params', 'net'):
            if k in obj and isinstance(obj[k], dict):
                sd = obj[k]
                break
    sd = {(k[6:] if k.startswith('model.') else k): v for k, v in sd.items()}
    sd = {(k[7:] if k.startswith('module.') else k): v for k, v in sd.items()}
    missing, unexpected = model.load_state_dict(sd, strict=False)
    if missing or unexpected:
        print(f"[load] missing {len(missing)}  unexpected {len(unexpected)}")
        print(f"       first missing: {missing[:5]}\n       first unexpected: {unexpected[:5]}")
        if len(missing) > 0.1 * len(model.state_dict()):
            raise RuntimeError("checkpoint does not match FSNet_UDPNet (ITS variant) -- "
                               "wrong architecture or wrong file")


def metrics(pred, gt):
    """Upstream PSNR/SSIM on (1,3,h,w) tensors in [0,1]."""
    h, w = gt.shape[2], gt.shape[3]
    psnr = 10 * torch.log10(1 / f.mse_loss(pred, gt)).item()
    r = max(1, round(min(h, w) / 256))
    s = ssim(f.adaptive_avg_pool2d(pred, (int(h / r), int(w / r))),
             f.adaptive_avg_pool2d(gt, (int(h / r), int(w / r))),
             data_range=1, size_average=False).item()
    return psnr, s


@torch.inference_mode()
def evaluate(model, args, device):
    loader = test_dataloader(args.data_dir, batch_size=1, num_workers=4,
                             center_crop=args.test_crop, split=args.split)
    factor = 8
    sums = dict(psnr=0.0, ssim=0.0, in_psnr=0.0, in_ssim=0.0)
    n, t_total = 0, 0.0
    for i, (inp, gt, name) in enumerate(loader):
        inp, gt = inp.to(device), gt.to(device)
        h, w = inp.shape[2], inp.shape[3]
        H, W = ((h + factor) // factor) * factor, ((w + factor) // factor) * factor
        padh = H - h if h % factor else 0
        padw = W - w if w % factor else 0
        x = f.pad(inp, (0, padw, 0, padh), 'reflect')

        if device.type == 'cuda':
            torch.cuda.synchronize()
        t0 = time.time()
        pred = model(x)[2][:, :, :h, :w].clamp(0, 1)
        if device.type == 'cuda':
            torch.cuda.synchronize()
        t_total += time.time() - t0

        p, s = metrics(pred, gt)
        ip, is_ = metrics(inp[:, :3].clamp(0, 1), gt)      # do-nothing baseline
        sums['psnr'] += p; sums['ssim'] += s
        sums['in_psnr'] += ip; sums['in_ssim'] += is_
        n += 1
        if args.verbose or (i + 1) % 200 == 0:
            print(f"{i+1:5d}  {name[0]:28s} PSNR {p:6.2f}  (input {ip:6.2f})  SSIM {s:.4f}")

        if args.save_image:
            out = pred.squeeze(0).cpu() + 0.5 / 255
            F.to_pil_image(out.clamp(0, 1), 'RGB').save(os.path.join(args.result_dir, name[0]))

    res = {k: v / max(n, 1) for k, v in sums.items()}
    res.update(n=n, split=args.split, data_dir=args.data_dir,
               checkpoint=args.test_model, test_crop=args.test_crop,
               sec_per_image=t_total / max(n, 1))
    res['gain_psnr'] = res['psnr'] - res['in_psnr']
    print("=" * 66)
    print(f" {args.data_dir}  [{args.split}]  n={n}")
    print(f"   hazy input (no dehazing) : PSNR {res['in_psnr']:6.3f}  SSIM {res['in_ssim']:.4f}")
    print(f"   UDPNet                   : PSNR {res['psnr']:6.3f}  SSIM {res['ssim']:.4f}")
    print(f"   gain                     : {res['gain_psnr']:+.3f} dB   "
          f"({res['sec_per_image']*1000:.1f} ms/img)")
    print("=" * 66)
    if args.json_out:
        os.makedirs(os.path.dirname(os.path.abspath(args.json_out)), exist_ok=True)
        with open(args.json_out, 'w') as fh:
            json.dump(res, fh, indent=1)
        print("wrote", args.json_out)
    return res


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--data_dir', required=True)
    ap.add_argument('--test_model', required=True)
    ap.add_argument('--split', default='test')
    ap.add_argument('--model_name', default='udpnet_eval')
    ap.add_argument('--test_crop', type=int, default=None,
                    help='centre-crop (None = full resolution, the paper protocol)')
    ap.add_argument('--json_out', default='')
    ap.add_argument('--save_image', action='store_true')
    ap.add_argument('--verbose', action='store_true')
    args = ap.parse_args()
    args.result_dir = os.path.join('results', args.model_name, args.split)
    os.makedirs(args.result_dir, exist_ok=True)

    cudnn.benchmark = False            # varying test sizes: benchmark would re-tune each image
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = build_net()
    load_weights(model, args.test_model)
    model.to(device).eval()
    evaluate(model, args, device)
