# UDPNet depth-prior ablation — GT-depth run

UDPNet ([Zuo et al., arXiv 2601.06909](https://arxiv.org/abs/2601.06909)) dehazes an image with the help of a
depth prior (a 4th input channel). This repository trains **the same UDPNet three times on synthetic aerial haze,
changing only the depth prior**, and is set up to run one of them — the **GT-depth variant** — on a separate GPU
(e.g. an H200 cluster). The other two variants run elsewhere with identical code and recipe.

| variant | depth channel | what it measures |
|---|---|---|
| `dav2` | Depth Anything V2 Large on the *hazy* image, per-image min-max (what UDPNet normally uses) | the estimated prior |
| **`gt`** | ground-truth metric depth → 1/d, per-image min-max (same format) | a perfect prior |
| `const` | 128 everywhere | no depth information |

Read the result as **gain** = PSNR(UDPNet output) − PSNR(hazy input), on `test` (official Oblique test, 770 frames)
and `test_heldout` (frames from 3 places with no training frames). `gt − dav2` is what DAv2's depth errors cost on
aerial views; `dav2 − const` and `gt − const` are what each prior adds.

> The haze was synthesised from the same GT depth (atmospheric scattering model), so `gt` is an optimistic upper bound.

## Running on the Aerodrone H200 service

Jenkins clones this repository into a container and runs one command; the console output (max 65,000 characters)
comes back as an issue comment. `/app/data` is read-only input, `/app/output` is the result volume, `/app/scratch`
is work space; neither `/app/output` nor `/app/scratch` carries over to the next job.

| request field | value |
|---|---|
| GPU quota | **`7`** (one full H200). `1` (a 16 GB slice) is too small for batch 6 |
| image | `pytorch/pytorch:latest` (deployed: `kau/pytorch-master`, torch 2.11, Python 3.10). `run_gt.sh` installs `requirements.txt` at start and, if torch < 2.6, torch 2.11 + torchvision 0.26 (cu128) |
| command | `bash run_gt.sh check` → `bash run_gt.sh smoke` → `bash run_gt.sh full` (`… full const` for the const variant) |

1. Give the admin the 27 data parts + `SHA256SUMS_parts` (see Data) for `/app/data/<any folder>`. They take 57 GB
   of the space that `/app/data` and `/app/output` share (~103 GB free on 2026-09-28) — ask for older packs to be removed if needed.
2. `check` (~1 min): paths, free space, GPU via torch, packages, where the parts are.
3. `smoke` (~15 min): unpack, 300 training steps, quick eval, results zip; prints ms/step and the projected hours.
4. `full`: unpack → train → eval on `test` and `test_heldout` (full resolution) → `/app/output/udpnet_gt/results_am_v2_lr1e-4_gt.zip`.
   Ask the admin for this zip.
5. The **const** variant uses the same data (its 128-everywhere priors are generated): `bash run_gt.sh smoke const`, then
   `bash run_gt.sh full const` → `/app/output/udpnet_const/results_am_v2_lr1e-4_const.zip`. A resume checkpoint for it is
   named `am_v2_lr1e-4_const_last.ckpt`.

What `run_gt.sh` takes care of:

- **Data**: finds `udpnet_gt_data.tar.part_*` anywhere under `/app/data` (or a zip holding the parts, as a Drive
  folder download arrives, or a single `udpnet_gt_data.zip`), checks SHA256, streams it into `/app/scratch` — never into
  `/app/output`. The training folder is built in the work space, so `/app/data` may stay read-only.
- **Watchdog**: if the training log gets no new line for `STALL_MIN` (30) minutes, the run is killed and restarted from
  `last.ckpt` (up to `MAX_RETRY` = 10 times). Out of memory stops without retrying; a failing `torch.compile` is retried
  without compile.
- **Time limit**: after `TIME_LIMIT_H` (34) hours minus `EVAL_RESERVE_MIN` (45) the training is stopped, the best
  checkpoint so far is evaluated and zipped (a job was once killed at 35 h). To continue in a new job, ask the admin
  to put `results/am_v2_lr1e-4_gt/checkpoints/last.ckpt` from the zip into `/app/data/<folder>/am_v2_lr1e-4_gt_last.ckpt`
  and request `full` again; it resumes from there.
- **Console**: one status line every `HB_MIN` (30) minutes (step, ms/step, ETA, activations, skipped batches, best val PSNR),
  the first 20 guard lines and errors. Everything else goes to files in `/app/output/udpnet_gt/`.
- **GPU memory** is read with torch (`nvidia-smi` may be unavailable in the container). No `setsid`, `zip`, `unzip`
  or `nvidia-smi` is needed; a missing C compiler turns `torch.compile` off.
- **Evaluation** has a timeout per split (`EVAL_TIMEOUT_MIN` = 240), so a GPU hang there cannot hold the job.

## Running elsewhere

```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu128   # match the CUDA
DATA_DIR=/path/udpnet_gt_data OUT_DIR=/path/out bash run_gt.sh smoke   # then: ... bash run_gt.sh full
```

Re-running continues: unpack and layout are skipped when done, training resumes from `last.ckpt`, a finished
training (`Final.ckpt`) is not repeated. Modes: `check | data | smoke | full | train | eval`.

| variable | default | meaning |
|---|---|---|
| `DATA_SRC` | `/app/data` | where the parts / zip / an unpacked `udpnet_gt_data` are searched |
| `DATA_DIR` | – | an unpacked `udpnet_gt_data` folder (skips the search) |
| `WORK_ROOT` | `/app/scratch/udpnet_work`, else `./work` | unpacked data and the training folder |
| `OUT_DIR` | `/app/output/udpnet_<variant>`, else `./outputs` | checkpoints, TensorBoard, logs, eval JSON, results zip |
| `VARIANT` (or 2nd argument) | `gt` | `gt` / `const` / `dav2` (`dav2` also needs `depth_hazy/dav2`, not in the gt pack) |
| `TIME_LIMIT_H`, `EVAL_RESERVE_MIN` | `34`, `45` | stop training in time for eval + zip (`0` = no limit) |
| `STALL_MIN`, `MAX_RETRY`, `HB_MIN` | `30`, `10`, `30` | watchdog, restarts, status-line interval |
| `RESUME_CKPT` | – | checkpoint to resume from (default: `*/am_v2_lr1e-4_gt_last.ckpt` under `DATA_SRC`) |
| `LINK_MODE` | `symlink` | `copy` if the file system has no symlinks (~75 GB) |
| `COMPILE` | `1` | `torch.compile` the training step (fp32 kept); turned off automatically if it fails |
| `EVAL_TIMEOUT_MIN` | `240` | per evaluation split |
| `WORKERS`, `PYTHON`, `SMOKE_STEPS` | `6`, `python` (else `python3`), `300` | |

## Data

Not in git. `udpnet_gt_data` (56.9 GB, PNG images) as an uncompressed tar split into 27 parts of 2 GiB,
`udpnet_gt_data.tar.part_00` … `part_26`, plus `SHA256SUMS_parts` (`sha256sum -c SHA256SUMS_parts`):

| path in the pack | files | size | used for |
|---|---|---|---|
| `haze_asm_v2/train/hazy` | 31,892 | 40.2 GB | train + val hazy images (4 haze densities per frame) |
| `haze_asm_v2/train/gt` | 7,973 | 10.8 GB | train + val clean images |
| `haze_asm_v2/test/hazy`, `test/gt` | 3,080 + 770 | 5.1 GB | test / test_heldout (evaluation only) |
| `udpnet_haze_asm_v2/depth_frames/gt` | 8,743 | 0.8 GB | GT depth prior, one per frame |
| `udpnet_haze_asm_v2/splits.json` | 1 | 1.3 MB | train / val / test / test_heldout lists |
| `probe/nan_batch_1008.pt` | 1 | 11 MB | the batch that blew up training at lr 4e-4 (monitoring only) |
| `MANIFEST.txt`, `haze_asm_v2/*/index.json` | | | file counts, generator metadata |

`run_gt.sh` checks that every file listed in `splits.json` exists and builds the per-variant training folder
(`udpnet_gt/{train,val,test,test_heldout}/{hazy,gt,depth2l}`) with `ablation/link_layout.py`; oblique frames
(pitch < 70°) appear 3× in train (45,664 training images).

How the data was made (for reference; not needed to run): AerialMetric frames → haze_asm_v2 haze generator
(β = 1, 2.5, 6, 15 /km) → `ablation/make_layout.py` (splits + GT depth prior, with an alignment check of every frame
against the saved clean image) → `ablation/make_dav2_depth.py` (DAv2 prior, `dav2` only).

## Recipe (identical for all three variants)

| | value | note |
|---|---|---|
| model | FSNet_UDPNet (upstream ITS variant) | architecture unchanged |
| training images | 3,000,000 (66 epochs × 45,664) | = Haze4K paper: 3,000 × 1,000 epochs |
| batch | 6 per step × 2 accumulation = **12** | upstream trains with DDP on 2 GPUs × 6 |
| optimizer | Adam(0.9, 0.999), **lr 1e-4**, gradient clip (norm) 0.01 | 1-epoch warmup (lr 0), cosine to 1e-6 per epoch |
| patch / precision | 256², fp32 (`torch.compile` on the training step) | |
| validation | `val` split (5 % of train frames), centre crop 512, every 3 epochs | checkpoint selection only — test is never used for selection |
| evaluation | best-val checkpoint on `test` and `test_heldout`, full resolution | reports input PSNR/SSIM, output PSNR/SSIM, gain |

251,130 optimizer steps. On an RTX PRO 4500 Blackwell: 730 ms/step, ~52 h per variant incl. validation.

## Stability

With the paper's lr 4e-4 the training diverged on this data: for some inputs (bright, almost uniform patches such
as dense haze or sky) the residual stream grows 2–10× per block through ~96 un-normalised residual blocks,
the loss reaches 1e16, the gradient norm overflows and all weights become NaN (at ~130–150 k training images).
An effective batch of 12 alone did not help (collapse at step 18,987). **lr 1e-4 trained 25 k steps with no event**,
so all variants use 1e-4.

Safety nets (see `udpnet/trainlightning.py`), logged as `[guard] skip: …` lines and TensorBoard `guard_skips`:

1. a micro-batch whose loss is non-finite or > 10× the median of the last 200 is skipped;
2. an optimizer step whose accumulated gradient is non-finite is skipped;
3. 50 consecutive skips stop the run.

`[monitor]` lines (every 100 steps) give the largest |value| of the Encoder.2 / Decoder / final outputs on the
current batch, `[probe]` lines (every 5,000 steps) the same for the saved blow-up batch. Healthy values are
~1–50; the lr 4e-4 failure showed 1e12–1e19.

## GPU memory and time

Batch 6 at 256² needs **~18–20 GiB** (17.5 GB process memory with compile, 19.8 GiB peak without). `run_gt.sh`
warns below 19 GiB and stops without retrying on out-of-memory. On the H200 service request GPU quota 7; a quota-1
slice (16 GB, 1/7 of the SMs) is too small and far slower. Do **not** lower the batch for this variant only:
the model's BatchNorm uses batch statistics, so the comparison with `dav2`/`const` would no longer be fair.

## Outputs (`OUT_DIR`)

```
results_am_v2_lr1e-4_gt.zip             what to bring back: summary JSON, logs, TensorBoard, best + last (+ Final) checkpoints
results/am_v2_lr1e-4_gt/checkpoints/   last.ckpt, epoch_epoch=XX_psnr_val_psnr=YY.ckpt (top 3), epoch_XX.ckpt, Final.ckpt
logs/am_v2_lr1e-4_gt/                   TensorBoard (train loss, lr, val_psnr, guard_skips, monitor/*, probe/*)
train_am_v2_lr1e-4_gt.log               full training log; run_<mode>_<name>.log = the console lines
summary_am_v2_lr1e-4_gt/gt__test.json, gt__test_heldout.json    -> summary table printed at the end
```

`results_am_v2_lr1e-4_gt.zip` holds everything needed to compare with the other variants.

## Layout

```
run_gt.sh          entry point (setup -> data -> layout -> train with watchdog/time limit -> eval -> results zip)
udpnet/            UDPNet training / evaluation code (modified from upstream, see NOTICE.md)
upstream_orig/     the upstream files that were modified, unchanged, for diff
ablation/          layout and prior scripts of the ablation; run_depth_ablation.sh = the local 3-variant driver
docs/              Korean run guide
```

## License and credits

Apache-2.0 (from upstream UDPNet). Upstream code, modifications, data and model sources: see [NOTICE.md](../NOTICE.md).
