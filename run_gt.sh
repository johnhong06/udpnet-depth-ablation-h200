#!/usr/bin/env bash
# UDPNet depth-prior ablation: train + evaluate one variant (default: the GT depth prior) on one GPU, with exactly the
# recipe of the other variants (README "Recipe"). Written for the Aerodrone H200 service (Jenkins runs one command in a
# container; /app/data read-only input, /app/output results, /app/scratch work space; nothing survives the job except
# what the admin keeps from /app/output) and also runs on any Linux GPU machine.
#
# One-line commands (H200 request: GPU quota 7 = one full H200; quota 1 = 16 GB slice is too small):
#   bash run_gt.sh check   # paths, free space, GPU (via torch), packages, where the data is              (~1 min)
#   bash run_gt.sh smoke   # data + 300 training steps + quick eval; prints ms/step and projected hours     (~15 min)
#   bash run_gt.sh full    # data -> train (watchdog, auto-restart) -> eval -> results zip
#   bash run_gt.sh eval    # eval + results zip only (checkpoints in OUT_DIR, or RESUME_CKPT)
#   bash run_gt.sh data    # find, verify and unpack the data only
#   bash run_gt.sh full const   # 2nd argument = variant (gt default; const needs no extra data)
# Elsewhere: DATA_DIR=<unpacked udpnet_gt_data> OUT_DIR=<out> bash run_gt.sh full
set -euo pipefail
REPO=$(cd "$(dirname "$0")" && pwd)
MODE=${1:-${MODE:-full}}

# ------------------------------------------------------------------ settings (env overrides)
VARIANT=${2:-${VARIANT:-gt}}            # gt | const | dav2 (dav2 also needs depth_hazy/dav2, not in the gt pack)
case $VARIANT in gt|const|dav2) ;; *) echo "unknown variant '$VARIANT' (gt | const | dav2)"; exit 1;; esac
PYTHON=${PYTHON:-$(command -v python >/dev/null 2>&1 && echo python || echo python3)}
PACK=${PACK:-udpnet_gt_data}            # data pack: $PACK.tar.part_NN + SHA256SUMS_parts (or $PACK.zip)
DATA_SRC=${DATA_SRC:-/app/data}         # where the pack / an unpacked copy is searched (read-only is fine)
TIME_LIMIT_H=${TIME_LIMIT_H:-0}         # optional: stop training after this many hours of the job, then eval + zip (0 = no limit; decimals ok)
EVAL_RESERVE_MIN=${EVAL_RESERVE_MIN:-45}  # time kept for eval + zip before TIME_LIMIT_H
STALL_MIN=${STALL_MIN:-30}              # no new training log output for this long -> kill and restart from last.ckpt
HB_MIN=${HB_MIN:-30}                    # one status line every HB_MIN minutes
MAX_RETRY=${MAX_RETRY:-10}              # restarts after a crash / stall, each from last.ckpt
LINK_MODE=${LINK_MODE:-symlink}         # symlink | copy (hardlink only if data and work dir share a file system)
COMPILE=${COMPILE:-1}                   # torch.compile the training step (fp32 kept); auto-off if it fails
NPROC=$(nproc 2>/dev/null || echo 8)
WORKERS=${WORKERS:-$(( NPROC >= 48 ? 16 : (NPROC >= 24 ? 8 : 6) ))}   # data loading only; no effect on the results
MONITOR_FREQ=${MONITOR_FREQ:-100}
PROBE_FREQ=${PROBE_FREQ:-5000}
SMOKE_STEPS=${SMOKE_STEPS:-300}
GROW_SEC=${GROW_SEC:-60}                # check: a part whose size changes within this many seconds is still being copied
EVAL_TIMEOUT_MIN=${EVAL_TIMEOUT_MIN:-240}  # per split; a hung GPU during eval must not hold the job
export TQDM_MININTERVAL=${TQDM_MININTERVAL:-30}   # progress bar redraw every 30 s: keeps the log small (it also feeds the stall check)

# recipe -- identical for every variant, do not change one alone
SAMPLES=${SAMPLES:-3000000}   # training images seen (66 epochs of 45,664)
BATCH=6; ACCUM=2              # effective 12 = upstream DDP on 2 GPUs x 6
LR=1e-4                       # 4e-4 (paper) diverged on this data (README "Stability")
PATCH=256; PRECISION=32-true; VALID_FREQ=3; SAVE_FREQ=10; VAL_CROP=512

alive () { [[ -e /proc/$1 ]] && ! grep -q '^State:[[:space:]]*Z' "/proc/$1/status" 2>/dev/null; }   # running, not a zombie
writable () { [ -d "$1" ] || return 1; touch "$1/.w_test" 2>/dev/null || return 1; rm -f "$1/.w_test"; }
freegb () { df -Pk "$1" 2>/dev/null | awk 'NR==2{printf "%d", int($4/1048576)}' || true; }
if [[ -z ${OUT_DIR:-} ]]; then
  if writable /app/output; then OUT_DIR=/app/output/udpnet_$VARIANT; else OUT_DIR=$REPO/outputs
    [[ -d /app/output ]] && echo "!!! /app/output exists but is not writable -> results go to $OUT_DIR and are lost with the container"; fi
fi
if [[ -z ${WORK_ROOT:-} ]]; then
  if writable /app/scratch; then WORK_ROOT=/app/scratch/udpnet_work; else WORK_ROOT=$REPO/work; fi
fi
case $WORK_ROOT in /app/output*) echo "!!! WORK_ROOT must not be in /app/output (results only)"; exit 1;; esac
mkdir -p "$OUT_DIR" "$WORK_ROOT"
if [[ $MODE == smoke ]]; then NAME=smoke_$VARIANT; else NAME=${NAME:-am_v2_lr1e-4_$VARIANT}; fi
LOG=$OUT_DIR/train_$NAME.log
JOBLOG=$OUT_DIR/run_${MODE}_$NAME.log
say () { echo "[run_gt $(date '+%m-%d %H:%M:%S')] $*" | tee -a "$JOBLOG"; }
die () { say "!!! $*"; exit 1; }

# ------------------------------------------------------------------ setup
stage_setup () {
  say "mode $MODE | variant $VARIANT | repo $REPO | out $OUT_DIR | work $WORK_ROOT | data source $DATA_SRC"
  if ! "$PYTHON" -c "import torch, torchvision, sys; v = tuple(int(x) for x in torch.__version__.split('+')[0].split('.')[:2]); sys.exit(0 if v >= (2, 6) else 1)" 2>/dev/null; then
    say "torch < 2.6 or torchvision missing -> installing torch 2.11.0 + torchvision 0.26.0 (cu128, ~3 GB)"
    "$PYTHON" -m pip uninstall -y -q torchaudio torchtext torchdata >/dev/null 2>&1 || true
    "$PYTHON" -m pip install -q "torch==2.11.0" "torchvision==0.26.0" --index-url https://download.pytorch.org/whl/cu128 2>&1 | tail -2 || true
    "$PYTHON" -c "import torch, torchvision" || die "torch / torchvision not importable"
  fi
  if ! "$PYTHON" -c "import pytorch_lightning, torchmetrics, einops, pytorch_msssim, tensorboard" 2>/dev/null; then
    say "installing requirements.txt"
    "$PYTHON" -m pip install -q -r "$REPO/requirements.txt" 2>&1 | tail -3 || true
    "$PYTHON" -c "import pytorch_lightning, torchmetrics, einops, pytorch_msssim, tensorboard" || die "requirements not importable"
  fi
  "$PYTHON" - <<'EOF' | tee -a "$JOBLOG"
import sys, torch, torchvision, pytorch_lightning as pl
print(f"[env] python {sys.version.split()[0]} | torch {torch.__version__} | torchvision {torchvision.__version__} | lightning {pl.__version__}")
if torch.cuda.is_available():
    p = torch.cuda.get_device_properties(0); gib = p.total_memory / 2**30
    print(f"[env] GPU {p.name} | {gib:.1f} GiB | {p.multi_processor_count} SMs")
    if gib < 19:
        print("[env] WARNING: batch 6 needs ~18-20 GiB (17.5 GB with compile, 19.8 GiB peak without) -> likely out of memory. "
              "Request GPU quota 7. Do not lower the batch for one variant only (BatchNorm uses batch statistics).")
else:
    print("[env] WARNING: no CUDA device")
EOF
  if [[ $COMPILE == 1 ]] && ! command -v gcc >/dev/null 2>&1 && ! command -v cc >/dev/null 2>&1; then
    COMPILE=0; say "no C compiler in the container -> torch.compile off (COMPILE=0: same fp32 math, slower)"
  fi
  say "CPU cores $NPROC -> data workers $WORKERS | /dev/shm $(df -h /dev/shm 2>/dev/null | awk 'NR==2{print $2}')"
  say "recipe: $SAMPLES images, batch $BATCH x accum $ACCUM, lr $LR, patch $PATCH, $PRECISION, compile $COMPILE," \
      "val every $VALID_FREQ ep (crop $VAL_CROP) | time limit $( [[ $TIME_LIMIT_H == 0 ]] && echo none || echo "${TIME_LIMIT_H} h"), stall ${STALL_MIN} min"
}

# ------------------------------------------------------------------ data: find -> verify -> unpack into WORK_ROOT
data_ok () { [[ -f $1/udpnet_haze_asm_v2/splits.json && -d $1/haze_asm_v2/train/hazy && -d $1/udpnet_haze_asm_v2/depth_frames/gt ]]; }

unpack_zips_of_parts () {   # Google Drive folder downloads arrive as zip(s) holding the tar parts: stream them into tar
  "$PYTHON" - "$PACK" "$WORK_ROOT" "$@" <<'EOF'
import hashlib, os, re, shutil, subprocess, sys, zipfile
pack, work, zips = sys.argv[1], sys.argv[2], sys.argv[3:]
members, sums = {}, {}
for z in zips:
    zf = zipfile.ZipFile(z)
    for n in zf.namelist():
        b = os.path.basename(n); m = re.match(re.escape(pack) + r"\.tar\.part_(\d+)$", b)
        if m: members[int(m.group(1))] = (zf, n)
        elif b == "SHA256SUMS_parts":
            for line in zf.read(n).decode().split("\n"):
                if line.strip(): h, f = line.split(); sums[f.lstrip("*")] = h
idx = sorted(members)
if not idx or idx != list(range(len(idx))): sys.exit(f"[data] parts in zips incomplete: {idx}")
print(f"[data] {len(idx)} parts inside {len(zips)} zip(s), {len(sums)} checksums", flush=True)
tmp = os.path.join(work, ".unpack_tmp"); shutil.rmtree(tmp, ignore_errors=True); os.makedirs(tmp)
tar = subprocess.Popen(["tar", "-xf", "-", "-C", tmp], stdin=subprocess.PIPE); bad = []
for i in idx:
    zf, n = members[i]; h = hashlib.sha256()
    with zf.open(n) as f:
        while chunk := f.read(1 << 24):
            h.update(chunk); tar.stdin.write(chunk)
    name = f"{pack}.tar.part_{i:02d}"
    if name in sums and sums[name] != h.hexdigest(): bad.append(name)
tar.stdin.close(); rc = tar.wait()
if bad or rc: shutil.rmtree(tmp, ignore_errors=True); sys.exit(f"[data] SHA256 mismatch {bad} / tar exit {rc}")
os.rename(os.path.join(tmp, pack), os.path.join(work, pack)); shutil.rmtree(tmp, ignore_errors=True)
EOF
}

stage_data () {
  local d pd zips
  if [[ -n ${DATA_DIR:-} ]]; then data_ok "$DATA_DIR" || die "DATA_DIR=$DATA_DIR is not an unpacked $PACK"; say "data: $DATA_DIR (given)"
  else
    for d in "$WORK_ROOT/$PACK" $(find "$DATA_SRC" -maxdepth 4 -type d -name "$PACK" 2>/dev/null || true); do
      if data_ok "$d"; then DATA_DIR=$d; say "data: already unpacked at $d"; break; fi
    done
  fi
  if [[ -z ${DATA_DIR:-} ]]; then
    pd=$(find "$DATA_SRC" -maxdepth 4 -name "$PACK.tar.part_00" -printf "%h\n" 2>/dev/null | head -1 || true)
    zips=$(find "$DATA_SRC" -maxdepth 4 -name "*.zip" 2>/dev/null | grep -v "/$PACK.zip$" | tr '\n' ' ' || true)
    local srcs fg need   # space check against the size of what will be unpacked (full pack: 57 GB)
    if [[ -n $pd ]]; then srcs=$(ls "$pd"/$PACK.tar.part_*); elif [[ -n $zips ]]; then srcs=$zips
    else srcs=$(find "$DATA_SRC" -maxdepth 4 -name "$PACK.zip" 2>/dev/null | head -1 || true); fi
    need=0; [[ -n $srcs ]] && need=$(du -skL $srcs 2>/dev/null | awk '{s += $1} END {print int(s / 1048576) + 3}' || echo 0)
    fg=$(freegb "$WORK_ROOT"); [[ -z $fg || $fg -ge $need ]] || die "$WORK_ROOT has $fg GB free, unpacking needs $need GB"
    if [[ -n $pd ]]; then
      say "data: $(ls "$pd"/$PACK.tar.part_* | wc -l) tar parts in $pd -> verify + unpack into $WORK_ROOT"
      [[ -f $pd/SHA256SUMS_parts ]] || die "no SHA256SUMS_parts next to the parts: upload incomplete? ('bash run_gt.sh check' shows the parts)"
      (cd "$pd" && sha256sum -c --quiet SHA256SUMS_parts) || die "part checksum mismatch: upload incomplete or broken ('bash run_gt.sh check' shows the parts)"
      say "data: SHA256 of all parts OK"
      rm -rf "$WORK_ROOT/.unpack_tmp"; mkdir -p "$WORK_ROOT/.unpack_tmp"
      cat "$pd"/$PACK.tar.part_* | tar -xf - -C "$WORK_ROOT/.unpack_tmp" || die "tar failed"
      mv "$WORK_ROOT/.unpack_tmp/$PACK" "$WORK_ROOT/$PACK"; rm -rf "$WORK_ROOT/.unpack_tmp"
    elif [[ -n $zips ]] && unpack_zips_of_parts $zips; then :
    elif d=$(find "$DATA_SRC" -maxdepth 4 -name "$PACK.zip" 2>/dev/null | head -1) && [[ -n $d ]]; then
      say "data: $d -> unzip into $WORK_ROOT/$PACK"; rm -rf "$WORK_ROOT/.unpack_tmp"
      "$PYTHON" -c "import sys, zipfile; zipfile.ZipFile(sys.argv[1]).extractall(sys.argv[2])" "$d" "$WORK_ROOT/.unpack_tmp" || die "unzip failed"
      if [[ -d $WORK_ROOT/.unpack_tmp/$PACK ]]; then mv "$WORK_ROOT/.unpack_tmp/$PACK" "$WORK_ROOT/$PACK"; rm -rf "$WORK_ROOT/.unpack_tmp"
      else mv "$WORK_ROOT/.unpack_tmp" "$WORK_ROOT/$PACK"; fi
    else die "no data: put $PACK.tar.part_* (+ SHA256SUMS_parts) anywhere under $DATA_SRC, or set DATA_DIR"; fi
    DATA_DIR=$WORK_ROOT/$PACK; data_ok "$DATA_DIR" || die "unpacked data incomplete"
    say "data: unpacked at $DATA_DIR"
  fi
  "$PYTHON" - "$DATA_DIR" "$VARIANT" <<'EOF' | tee -a "$JOBLOG"
import json, os, sys
d, variant = sys.argv[1:3]
S = json.load(open(os.path.join(d, "udpnet_haze_asm_v2", "splits.json")))
need = {"train": set(), "test": set()}; ids = set()
for entries in S["splits"].values():
    for e in entries: need[e["src"]].add(e["tag"]); ids.add(e["id"])
ok = True
for src, tags in need.items():
    hz = set(os.listdir(os.path.join(d, "haze_asm_v2", src, "hazy"))); gt = set(os.listdir(os.path.join(d, "haze_asm_v2", src, "gt")))
    miss = sum(f"{t}_b{b}.png" not in hz for t in tags for b in S["betas"]) + sum(f"{t}.png" not in gt for t in tags)
    print(f"[data] {src}: {len(tags)} frames, hazy {len(hz)}, gt {len(gt)}, missing {miss}"); ok &= miss == 0
dg = set(os.listdir(os.path.join(d, "udpnet_haze_asm_v2", "depth_frames", "gt")))
miss = sum(f"{i}.png" not in dg for i in ids); print(f"[data] depth_frames/gt: {len(dg)}, missing {miss}"); ok &= miss == 0 or variant != "gt"
sys.exit(0 if ok else "[data] ERROR: files missing")
EOF
}

# ------------------------------------------------------------------ layout (always in WORK_ROOT: /app/data may be read-only)
stage_prep () {
  HAZE=$DATA_DIR/haze_asm_v2; LROOT=$WORK_ROOT/layout; VDIR=$LROOT/udpnet_$VARIANT
  mkdir -p "$LROOT/depth_frames"
  cp -f "$DATA_DIR/udpnet_haze_asm_v2/splits.json" "$LROOT/splits.json"
  ln -sfn "$DATA_DIR/udpnet_haze_asm_v2/depth_frames/gt" "$LROOT/depth_frames/gt"
  [[ -d $DATA_DIR/udpnet_haze_asm_v2/depth_hazy ]] && ln -sfn "$DATA_DIR/udpnet_haze_asm_v2/depth_hazy" "$LROOT/depth_hazy"
  local need; need=$("$PYTHON" -c "import json,sys; print(json.load(open(sys.argv[1]))['summary']['train']['hazy_images'])" "$LROOT/splits.json")
  if [[ -f $VDIR/.complete && $(ls "$VDIR/train/hazy" | wc -l) == "$need" ]]; then say "layout: $VDIR exists ($need training images)"; return; fi
  say "layout: building $VDIR ($LINK_MODE)"
  rm -rf "$VDIR"
  "$PYTHON" "$REPO/ablation/link_layout.py" --haze "$HAZE" --root "$LROOT" --variants "$VARIANT" --mode "$LINK_MODE" | tee -a "$JOBLOG"
  touch "$VDIR/.complete"
}

udp_cd () {   # main.py / test.py write to ./results and ./logs -> point them into OUT_DIR
  mkdir -p "$OUT_DIR/results" "$OUT_DIR/logs"; cd "$REPO/udpnet"
  local d; for d in results logs; do
    [[ -e $d && ! -L $d ]] && die "$REPO/udpnet/$d is a real directory; move it into $OUT_DIR"
    ln -sfn "$OUT_DIR/$d" "$d"; done
}

# ------------------------------------------------------------------ status helpers (read the full log)
# tqdm redraws with \r and our print() lines can land right after a bar on the same line -> split before the tags
norm () { tr '\r' '\n' | sed -E 's/\[(monitor|guard|probe)\]/\n[\1]/g'; }
gpu_util () { nvidia-smi --query-gpu=utilization.gpu --format=csv,noheader,nounits 2>/dev/null | head -1 | tr -dc 0-9 || true; }
last_step () { norm < "$LOG" 2>/dev/null | grep -a '^\[monitor\] step' | tail -1 | awk '{print $3}' || true; }
heartbeat () {   # one line: progress, speed, ETA, activations, skips, best val
  local step=$1 rate=$2 total=$3 mon skips best
  mon=$(norm < "$LOG" | grep -a '^\[monitor\] step' | tail -1 | awk '{printf "enc2 %.1f dec0 %.1f out2 %.2f", $5, $7, $(NF)}' || true)
  skips=$(norm < "$LOG" | grep -ac '^\[guard\]' || true)
  best=$(ls "$OUT_DIR/results/$NAME/checkpoints"/epoch_epoch=*_psnr_val_psnr=*.ckpt 2>/dev/null \
         | sed -E 's/.*epoch=([0-9]+)_psnr_val_psnr=([0-9.]+)\.ckpt/\2 (ep \1)/' | sort -rn | head -1 || true)
  local eta=""; [[ -n $rate && $rate != 0 ]] && eta=$(awk -v r="$rate" -v s="$step" -v t="$total" 'BEGIN{printf "%.0f ms/step, ETA %.1f h", r*1000, (t-s)*r/3600}')
  local gu; gu=$(awk '{s += $1; n++} END {if (n) printf "%.0f%%", s/n}' "$WORK_ROOT/.gpu_util" 2>/dev/null); : > "$WORK_ROOT/.gpu_util"
  say "[hb] step $step/$total ($(awk -v s="$step" -v t="$total" 'BEGIN{printf "%.1f", 100*s/t}')%) $eta | GPU ${gu:-?} | $mon | skips ${skips:-0} | best val ${best:--}"
}

# ------------------------------------------------------------------ train: watchdog + time limit + restarts
stage_train () {
  udp_cd
  local ck=$OUT_DIR/results/$NAME/checkpoints n ep total try rc reason shown=0 errs=0 extra smoke src
  if [[ -f $ck/Final.ckpt ]]; then say "train: Final.ckpt exists -> skipped"; return 0; fi
  if [[ ! -f $ck/last.ckpt ]]; then   # resume from a previous job's last.ckpt that the admin put under /app/data
    src=${RESUME_CKPT:-$(find "$DATA_SRC" -maxdepth 4 -name "${NAME}_last.ckpt" 2>/dev/null | head -1 || true)}
    if [[ -n $src && -f $src ]]; then mkdir -p "$ck"; cp "$src" "$ck/last.ckpt"; say "train: resume checkpoint $src"; fi
  fi
  n=$(ls "$VDIR/train/hazy" | wc -l); ep=$(( (SAMPLES + n - 1) / n )); total=$(( ep * (n / BATCH / ACCUM) ))
  smoke=(); [[ $MODE == smoke ]] && { smoke=(--max_steps "$SMOKE_STEPS"); total=$SMOKE_STEPS; }
  local deadline; deadline=$(awk -v h="$TIME_LIMIT_H" -v r="$EVAL_RESERVE_MIN" 'BEGIN{d = h*3600 - r*60; if (d < 60) d = 60; printf "%d", (h > 0 ? d : 0)}')
  say "train $NAME: $n images -> $ep epochs, $total optimizer steps | full log $LOG"; rm -f "$WORK_ROOT/.gpu_util_all"
  for ((try = 0; try <= MAX_RETRY; try++)); do
    local resume=(); [[ -f $ck/last.ckpt ]] && resume=(--resume "$ck/last.ckpt")
    extra=(); [[ $COMPILE == 1 ]] && extra=(--compile)
    local off off0; off=$(stat -c %s "$LOG" 2>/dev/null || echo 0); off0=$off
    say "train: start (try $try${resume:+, resume from last.ckpt}, compile $COMPILE, workers $WORKERS)"
    echo "[run_gt $(date '+%m-%d %H:%M:%S')] start try $try" >> "$LOG"   # fresh mtime: the stall clock starts now
    local probe=(); [[ -f $DATA_DIR/probe/nan_batch_1008.pt ]] && probe=(--probe_batch "$DATA_DIR/probe/nan_batch_1008.pt" --probe_freq "$PROBE_FREQ")
    "$PYTHON" -c 'import os, sys; os.setsid(); os.execvp(sys.argv[1], sys.argv[1:])' "$PYTHON" main.py --model_name "$NAME" --data_dir "$VDIR" \
      --num_epoch "$ep" --batch_size "$BATCH" --patch_size "$PATCH" --learning_rate "$LR" \
      --precision "$PRECISION" --num_worker "$WORKERS" \
      --val_split val --val_crop "$VAL_CROP" --valid_freq "$VALID_FREQ" --save_freq "$SAVE_FREQ" \
      --monitor_freq "$MONITOR_FREQ" "${probe[@]}" \
      --accum "$ACCUM" --logs "$NAME" "${extra[@]}" "${smoke[@]}" "${resume[@]}" >> "$LOG" 2>&1 < /dev/null &
    local bg=$! pg=$!   # setsid + exec: the PID is the process-group id of main.py and its data workers
    reason=""; local t_hb=$SECONDS s_hb; s_hb=$(last_step); s_hb=${s_hb:-0}; : > "$WORK_ROOT/.gpu_util"
    while alive "$bg"; do
      sleep 10
      local u st; u=$(gpu_util); st=$(last_step); st=${st:-0}
      if [[ -n $u ]] && (( st >= 100 )); then   # skip compile / warm-up
        echo "$u" >> "$WORK_ROOT/.gpu_util"; echo "$u" >> "$WORK_ROOT/.gpu_util_all"; fi
      # surface guard lines (first 20 over the whole run) and errors
      local new; new=$(tail -c +$((off + 1)) "$LOG" 2>/dev/null | norm | grep -aE '^\[guard\]|Traceback|Error|error:' || true)
      off=$(stat -c %s "$LOG")
      if [[ -n $new ]]; then
        while IFS= read -r l; do
          if [[ $l == "[guard]"* ]]; then (( shown < 20 )) && echo "  $l" | tee -a "$JOBLOG"; shown=$((shown + 1))
          else (( errs < 40 )) && echo "  $l" | tee -a "$JOBLOG"; errs=$((errs + 1)); fi
        done <<< "$new"
      fi
      local age=$(( $(date +%s) - $(stat -c %Y "$LOG") ))
      if (( age > STALL_MIN * 60 )); then reason=stall; say "train: no log output for $((age / 60)) min -> kill and restart"; break; fi
      if (( deadline > 0 && SECONDS > deadline )); then reason=timelimit; say "train: time limit (${TIME_LIMIT_H} h - ${EVAL_RESERVE_MIN} min) -> stop, eval the best checkpoint so far"; break; fi
      if (( SECONDS - t_hb >= HB_MIN * 60 )); then
        local s; s=$(last_step); s=${s:-0}
        heartbeat "$s" "$(awk -v a="$s" -v b="$s_hb" -v t=$((SECONDS - t_hb)) 'BEGIN{if (a > b) printf "%.4f", t/(a-b)}')" "$total"
        t_hb=$SECONDS; s_hb=$s
      fi
    done
    if [[ -n $reason ]]; then kill -TERM -- "-$pg" 2>/dev/null || true; sleep 20; kill -KILL -- "-$pg" 2>/dev/null || true; fi
    rc=0; wait "$bg" 2>/dev/null || rc=$?
    [[ -n $reason ]] && rc=killed
    if [[ $rc == 0 ]]; then say "train: done ($(last_step) steps logged)"; return 0; fi
    [[ $reason == timelimit ]] && { TRAIN_INCOMPLETE=1; return 0; }
    local seg tailtxt; seg=$(tail -c +$((off0 + 1)) "$LOG" | norm || true); tailtxt=$(grep -av '^\s*$' <<< "$seg" | tail -400 || true)
    if grep -aqE 'OutOfMemoryError|CUDA out of memory' <<< "$tailtxt"; then die "train: out of memory -> not retrying (README 'GPU memory')"; fi
    if (( WORKERS > 6 )) && grep -aqiE 'shared memory|/dev/shm|Bus error|DataLoader worker .* (killed|exited unexpectedly)' <<< "$tailtxt"; then
      WORKERS=6; say "train: data-loader / shared-memory failure -> retry with WORKERS=6"; continue; fi
    if [[ $COMPILE == 1 ]] && ! grep -aq '^\[monitor\]' <<< "$seg" \
       && grep -aqE 'BackendCompilerFailed|InductorError|CompilationError|torch\._dynamo\.exc|torch\._inductor\.exc' <<< "$tailtxt"; then
      COMPILE=0; say "train: torch.compile failed -> retry with COMPILE=0 (same fp32 math, slower)"; continue; fi
    say "train: FAILED (exit $rc${reason:+, $reason}, try $try)"
    grep -aE 'Error|error:' <<< "$tailtxt" | tail -5 | sed 's/^/  /' | tee -a "$JOBLOG" || true
    (( try < MAX_RETRY )) && { say "train: restart in 60 s from last.ckpt"; sleep 60; }
  done
  die "train: gave up after $MAX_RETRY restarts"
}

best_ckpt () {   # highest val_psnr checkpoint, else last.ckpt, else Final.ckpt
  local d=$OUT_DIR/results/$NAME/checkpoints b
  b=$(ls "$d"/epoch_epoch=*_psnr_val_psnr=*.ckpt 2>/dev/null | sed -E 's/.*val_psnr=([0-9.]+)\.ckpt/\1 &/' | sort -rn | head -1 | cut -d' ' -f2 || true)
  if [[ -n $b ]]; then echo "$b"; elif [[ -f $d/last.ckpt ]]; then echo "$d/last.ckpt"; else echo "$d/Final.ckpt"; fi
}

stage_eval () {
  udp_cd
  local ck splits=(test test_heldout) crop=() s
  ck=$(best_ckpt); [[ -f $ck ]] || die "eval: no checkpoint in $OUT_DIR/results/$NAME/checkpoints"
  [[ $MODE == smoke ]] && { splits=(test_heldout); crop=(--test_crop 256); }
  mkdir -p "$OUT_DIR/summary_$NAME"
  say "eval: $ck"
  for s in "${splits[@]}"; do
    timeout "${EVAL_TIMEOUT_MIN}m" "$PYTHON" test.py --data_dir "$VDIR" --split "$s" --test_model "$ck" "${crop[@]}" --model_name "eval_${NAME}_$s" \
      --json_out "$OUT_DIR/summary_$NAME/${VARIANT}__${s}.json" >> "$OUT_DIR/eval_$NAME.log" 2>&1 \
      || { tail -5 "$OUT_DIR/eval_$NAME.log"; die "eval on $s failed or timed out (${EVAL_TIMEOUT_MIN} min)"; }
    grep -aE "hazy input|UDPNet  |gain" "$OUT_DIR/eval_$NAME.log" | tail -3 | sed "s/^/  [$s] /" | tee -a "$JOBLOG"
  done
  "$PYTHON" summarize.py "$OUT_DIR/summary_$NAME" | tee -a "$JOBLOG"
}

stage_package () {   # one zip for the admin to hand back: results, logs, TensorBoard, best + last checkpoints
  cd "$OUT_DIR"
  local z=results_$NAME.zip b f items=()
  b=$(best_ckpt); b=${b#"$OUT_DIR"/}
  for f in "summary_$NAME" "train_$NAME.log" "eval_$NAME.log" "run_${MODE}_$NAME.log" "logs/$NAME" "$b" \
           "results/$NAME/checkpoints/last.ckpt" "results/$NAME/checkpoints/Final.ckpt"; do
    [[ -e $f && " ${items[*]} " != *" $f "* ]] && items+=("$f"); done
  rm -f "$z"
  "$PYTHON" - "$z" "${items[@]}" <<'EOF2' || say "!!! could not write $z"
import os, sys, zipfile
with zipfile.ZipFile(sys.argv[1], "w", zipfile.ZIP_STORED, allowZip64=True) as z:
    for it in sys.argv[2:]:
        for root, _, files in (os.walk(it) if os.path.isdir(it) else [(os.path.dirname(it), [], [os.path.basename(it)])]):
            for f in files: z.write(os.path.join(root, f))
EOF2
  say "results: $OUT_DIR/$z ($(du -h "$z" 2>/dev/null | cut -f1)): ${items[*]}"
  if [[ ${TRAIN_INCOMPLETE:-0} == 1 ]]; then
    say "!!! training stopped at the time limit (step $(last_step)). To continue: ask the admin to put" \
        "$OUT_DIR/results/$NAME/checkpoints/last.ckpt into /app/data/<folder>/${NAME}_last.ckpt and request 'bash run_gt.sh full' again"
  fi
}

stage_check () {
  local d
  say "== paths"
  for d in /app/data /app/output /app/scratch "$WORK_ROOT" "$OUT_DIR"; do
    [[ -e $d ]] && echo "  $d: $(writable "$d" && echo writable || echo read-only), free $(freegb "$d") GB" | tee -a "$JOBLOG" || echo "  $d: missing"; done
  say "== data under $DATA_SRC"
  # only this pack (the shared /app/data holds other users' archives too)
  find "$DATA_SRC" -maxdepth 4 \( -name "$PACK.tar.part_00" -o -name "$PACK.zip" -o -type d -name "$PACK" -o -name "am_v2_lr1e-4_*_last.ckpt" \) 2>/dev/null | head -10 | sed 's/^/  /' | tee -a "$JOBLOG" || true
  local pd; pd=$(find "$DATA_SRC" -maxdepth 4 -name "$PACK.tar.part_*" -printf "%h\n" 2>/dev/null | sort -u | head -1 || true)
  if [[ -n $pd ]]; then   # upload progress: which parts are there (by name and size; the full SHA256 check runs in smoke/full)
    say "upload check: sizes now and after ${GROW_SEC} s"
    "$PYTHON" - "$pd" "$PACK" "$GROW_SEC" <<'EOF2' | tee -a "$JOBLOG"
import os, re, sys, time
d, pack, grow = sys.argv[1], sys.argv[2], int(sys.argv[3])
GIB2 = 2 * 2**30
LAST = {"udpnet_gt_data": (27, 1141204992)}   # (number of parts, exact size of the last part) of the published pack
pat = re.compile(re.escape(pack) + r"\.tar\.part_(\d+)")
def sizes():
    return {int(m.group(1)): os.path.getsize(os.path.join(d, f)) for f in os.listdir(d) if (m := pat.fullmatch(f))}
a = sizes(); time.sleep(grow); have = sizes()          # two snapshots: a file that grows is still being copied
sums = os.path.join(d, "SHA256SUMS_parts")
want = sorted(int(re.search(r"part_(\d+)", l).group(1)) for l in open(sums) if "part_" in l) if os.path.exists(sums) else None
n = len(want) if want else (max(have) + 1 if have else 0)
miss = [i for i in range(n) if i not in have]
growing = [i for i in have if a.get(i) != have[i]]
short = [i for i in have if i < n - 1 and have[i] < GIB2]
if pack in LAST and n == LAST[pack][0] and (n - 1) in have and have[n - 1] != LAST[pack][1]:
    short.append(n - 1)
print(f"  parts in {d}: {len(have)}/{n}" + ("" if want else " (SHA256SUMS_parts missing: total unknown)")
      + (f", missing {miss}" if miss else "") + (f", still growing {sorted(growing)}" if growing else "")
      + (f", incomplete size {sorted(set(short))}" if short else ""))
temp = sorted(f for f in os.listdir(d) if (pack + ".tar.part_") in f and not pat.fullmatch(f))   # .part / .crdownload / rclone .partial ...
if temp:
    print(f"  temporary / partial files (a download may still be running): {temp[:10]}"
          + (f" (+{len(temp) - 10} more)" if len(temp) > 10 else ""))
ok = want and not miss and not growing and not short
print("  data upload: " + ("COMPLETE (checksums are verified in smoke/full)" if ok else "INCOMPLETE -> wait before smoke/full"))
EOF2
  fi
  say "== CPU cores $(nproc 2>/dev/null || echo ?), RAM $(awk '/MemTotal/{printf "%.0f GB", $2/1048576}' /proc/meminfo 2>/dev/null) (data loading needs ~1 core per worker; WORKERS=$WORKERS)"
  say "== tools: gcc $(command -v gcc >/dev/null && echo yes || echo no), nvidia-smi $(command -v nvidia-smi >/dev/null && echo yes || echo no), zip $(command -v zip >/dev/null && echo yes || echo no)"
}

TRAIN_INCOMPLETE=0
case $MODE in
  check) stage_setup; stage_check ;;
  data)  stage_setup; stage_data; stage_prep ;;
  smoke|full) stage_setup; stage_data; stage_prep; stage_train; stage_eval; stage_package ;;
  train) stage_setup; stage_data; stage_prep; stage_train; stage_package ;;
  eval)  stage_setup; stage_data; stage_prep; stage_eval; stage_package ;;
  *) echo "usage: bash run_gt.sh check|smoke|full|train|eval|data"; exit 1 ;;
esac
if [[ $MODE == smoke ]]; then   # projected full run from the measured speed (steps 100..300, compile warm-up excluded)
  "$PYTHON" - "$OUT_DIR/logs/$NAME" "$WORK_ROOT/.gpu_util_all" <<'EOF' | tee -a "$JOBLOG" || true
import glob, os, sys
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
d = sorted(glob.glob(os.path.join(sys.argv[1], "version_*")), key=os.path.getmtime)[-1]
ea = EventAccumulator(d, size_guidance={"scalars": 0}); ea.Reload()
s = [x for x in ea.Scalars("train_loss") if x.step >= 100]
if len(s) >= 2:
    r = (s[-1].wall_time - s[0].wall_time) / (s[-1].step - s[0].step)
    print(f"[smoke] {r * 1000:.0f} ms/optimizer step -> full run 251,130 steps ~ {251130 * r / 3600:.1f} h (+ ~22 validations)")
u = [int(x) for x in open(sys.argv[2]).read().split()] if os.path.exists(sys.argv[2]) else []
if u:
    print(f"[smoke] GPU utilisation during training: mean {sum(u) / len(u):.0f}% over {len(u)} samples"
          + (" -> GPU is not kept busy (data loading or kernel launches limit the speed)" if sum(u) / len(u) < 85 else ""))
EOF
fi
say "DONE $MODE"
