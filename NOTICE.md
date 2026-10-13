# NOTICE

## Upstream code

`udpnet/` is derived from the official implementation of

> Zengyuan Zuo, Junjun Jiang, Gang Wu, Xianming Liu. *UDPNet: Unleashing Depth-based Priors for Robust Image
> Dehazing.* arXiv:2601.06909. AIIA Lab, Harbin Institute of Technology.

licensed under the **Apache License 2.0** (copy in `LICENSE`). It was taken from the `Dehazing/ITS` folder of the
upstream archive (`UDPNet-main`, obtained 2026-09). TODO: add the exact repository URL / commit of the archive.
UDPNet builds on FSNet / ConvIR by Yuning Cui et al. (https://github.com/c-yn/FSNet,
https://github.com/c-yn/ConvIR), as acknowledged in the upstream README.

### Files and changes

The modified upstream files are kept unchanged in `upstream_orig/*.orig` (`diff upstream_orig/main.py.orig udpnet/main.py`).

| file | status | changes |
|---|---|---|
| `udpnet/models/FSNet_UDPNet.py` | unchanged | – |
| `udpnet/data/__init__.py`, `data_augment.py` | unchanged | – |
| `udpnet/models/fsnetlayers.py` | modified | inference only: OCAB attention and the `dynamic_filter` unfold are computed in chunks for large inputs (outputs bit-identical on 400², 640×480, 1920×1080; training path untouched); intermediates freed early |
| `udpnet/main.py` | modified | imports `FSNet_UDPNet` explicitly (upstream ITS `main.py` imports the depth-free `FSNet`); recipe defaults; new options `--precision`, `--compile`, `--val_split`, `--valid_freq`, `--val_crop`, `--accum`, `--max_steps`, `--monitor_freq`, `--probe_batch`, `--probe_freq`; snapshot of the code into the checkpoint folder |
| `udpnet/trainlightning.py` | modified | accelerator chosen from the hardware, PyTorch Lightning 2.6 compatibility; per-epoch LR as a closed-form `LambdaLR` (equal to upstream warmup + cosine in a fresh run, but correct after resume); `weights_only=False` on resume; `--valid_freq 0` mode with an atomic per-epoch `last.ckpt`; optional `torch.compile` of the training step; FFT loss in fp32; gradient accumulation; guards against loss spikes / non-finite gradients; activation monitoring. Gradient clipping is 0.01 (upstream ITS: 0.001, OTS: 0.01) |
| `udpnet/data/data_load.py` | modified | gt name rule strips the extension first; `drop_last`, persistent workers; `val` split and centre crop for validation; skips macOS `._*` files |
| `udpnet/test.py` | modified | also reports the hazy-input baseline (PSNR/SSIM) and the gain; JSON output; `--split`, `--test_crop` |
| `udpnet/summarize.py` | new | table of `test.py` JSON results |

Comments in the modified files are marked `[mod]`.

## Own work (this repository)

`run_gt.sh`, `ablation/` (`make_layout.py`, `make_dav2_depth.py`, `link_layout.py`, `run_depth_ablation.sh`),
`README.md`, `docs/`. Released under the same Apache License 2.0.

## Data and models (not redistributed here)

- **AerialMetric** (ECCV 2026, arXiv:2606.29716, Hugging Face `Kuiee/AerialMetric-ECCV2026`): RGB frames and
  metric depth. The data archive used by `run_gt.sh` is derived from it (synthetic haze, GT depth priors) and is
  **not** part of this repository; its use is subject to the AerialMetric terms — check them before sharing the archive.
- **Depth Anything V2 Large** (`depth-anything/Depth-Anything-V2-Large-hf`; Yang et al., NeurIPS 2024): used only
  for the `dav2` prior (`ablation/make_dav2_depth.py`), not by the `gt` run. Check the model card for its license
  (the Large checkpoint is distributed under a non-commercial license).

## Citation

```bibtex
@article{zuo2026udpnet,
  title   = {UDPNet: Unleashing Depth-based Priors for Robust Image Dehazing},
  author  = {Zuo, Zengyuan and Jiang, Junjun and Wu, Gang and Liu, Xianming},
  journal = {arXiv preprint arXiv:2601.06909},
  year    = {2026}
}
```
