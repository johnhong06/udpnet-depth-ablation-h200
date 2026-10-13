"""
UDPNet (FSNet + UDP) training entry point for Haze4K / HazyDet.

Model is the ITS-folder FSNet_UDPNet (1 attention head, InstanceNorm in the depth
refinement) -- the variant the released Haze4K code path uses, so released
Haze4K weights load into it. Note the upstream ITS/main.py imports the depth-free
baseline `models.FSNet` by mistake; this one imports FSNet_UDPNet explicitly.

Paper recipe for Haze4K (UDPNet sec. 5.2): 1000 epochs, batch 6, 256x256 patches,
Adam(0.9, 0.999), lr 4e-4 -> 1e-6 cosine, 2x RTX 3090.
"""
import os
import shutil

import argparse
import torch
from torch.backends import cudnn

from models.FSNet_UDPNet import build_net
from trainlightning import _train


def main(args):
    cudnn.benchmark = True
    os.makedirs(args.model_save_dir, exist_ok=True)
    os.makedirs(args.result_dir, exist_ok=True)

    model = build_net()
    n = sum(p.numel() for p in model.parameters())
    print(f"[model] FSNet_UDPNet (ITS variant)  {n/1e6:.2f} M params")
    if torch.cuda.is_available():
        model.cuda()
    if args.mode == 'train':
        _train(model, args)


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--model_name', default='udpnet_haze4k', type=str)
    p.add_argument('--data_dir', type=str, required=True,
                   help='root containing train/ and test/ (and val/), each with '
                        'hazy/ gt/ depth2l/')
    p.add_argument('--mode', default='train', choices=['train', 'test'], type=str)

    # recipe (defaults = paper, Haze4K)
    p.add_argument('--batch_size', type=int, default=6)
    p.add_argument('--patch_size', type=int, default=256)
    p.add_argument('--learning_rate', type=float, default=4e-4)
    p.add_argument('--weight_decay', type=float, default=0)
    p.add_argument('--num_epoch', type=int, default=1000)
    p.add_argument('--num_worker', type=int, default=8)
    p.add_argument('--precision', type=str, default='32-true',
                   choices=['32-true', 'bf16-mixed', '16-mixed'],
                   help='32-true = paper. bf16-mixed ~1.5-2x faster on RTX 40/50 '
                        'series; verify PSNR against a 32-true run before relying on it')
    p.add_argument('--compile', action='store_true',
                   help='[mod] torch.compile the training forward (RTX PRO 4500: '
                        'fp32 472 -> 347 ms/step). Validation stays eager')

    # validation / checkpoints
    p.add_argument('--val_split', type=str, default='test',
                   help="'test' = paper protocol (Haze4K has no val split). Use "
                        "'val' for HazyDet so test stays untouched by model selection")
    p.add_argument('--valid_freq', type=int, default=10,
                   help='validate every N epochs. Every epoch adds ~27%% to Haze4K '
                        'wall time (1000 test images at 400x400)')
    p.add_argument('--val_crop', type=int, default=None,
                   help='centre-crop validation images (None = full resolution)')
    p.add_argument('--save_freq', type=int, default=50)
    # [mod 10-08]
    p.add_argument('--accum', type=int, default=1,
                   help='accumulate_grad_batches. Upstream trains with DDP on 2 GPUs (batch 6 each = effective '
                        '12); on one GPU use --batch_size 6 --accum 2 for the same effective batch')
    p.add_argument('--max_steps', type=int, default=-1,
                   help='stop after this many optimizer steps (-1 = run all epochs; LR schedule still follows --num_epoch)')
    p.add_argument('--monitor_freq', type=int, default=0,
                   help='every N optimizer steps log max |activation| of Encoder.2 / Decoder / outputs on the current batch (0 = off)')
    p.add_argument('--probe_batch', type=str, default='',
                   help="file with {'x': tensor} fed every --probe_freq steps, max |activation| per sample")
    p.add_argument('--probe_freq', type=int, default=0)
    p.add_argument('--resume', type=str, default='')
    p.add_argument('--logs', type=str, default='udpnet')

    # kept for compatibility with trainlightning / test
    p.add_argument('--print_freq', type=int, default=100)
    p.add_argument('--test_model', type=str, default='')
    p.add_argument('--save_image', type=bool, default=False)

    args = p.parse_args()
    args.model_save_dir = os.path.join('results', args.model_name, 'checkpoints')
    args.result_dir = os.path.join('results', args.model_name, 'test')

    # snapshot the code that produced this run next to its checkpoints
    os.makedirs(args.model_save_dir, exist_ok=True)
    for f in ('main.py', 'trainlightning.py', 'models/FSNet_UDPNet.py', 'data/data_load.py'):
        if os.path.exists(f):
            shutil.copy(f, os.path.join(args.model_save_dir, os.path.basename(f)))
    print(vars(args))
    main(args)
