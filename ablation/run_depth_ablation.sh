#!/usr/bin/env bash
# Depth-prior ablation on haze_asm_v2: the same UDPNet trained three times, only the depth input differs.
#   dav2   DAv2-Large on the hazy image, min-max (what UDPNet normally uses)
#   gt     GT metric depth -> 1/d, min-max (same format, better depth)
#   const  128 everywhere (no depth information)
#
#   bash run_depth_ablation.sh prep      # DAv2 priors + symlinked data folders (GPU, ~1 h)
#   bash run_depth_ablation.sh train     # 3 trainings (PARALLEL=1: all at once on one GPU)
#   bash run_depth_ablation.sh eval      # best val checkpoint -> test and test_heldout, full resolution
#   bash run_depth_ablation.sh summary
#   bash run_depth_ablation.sh all
# Smoke test first:  SAMPLES=3000 VARIANTS=const MODEL_PREFIX=smoke bash run_depth_ablation.sh train
set -euo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"

# ------------------------------------------------------------------ paths (edit or export)
HAZE=${HAZE:-/data/haze_asm_v2}                 # copy of <AerialMetric>/haze_asm_v2 (hazy/ gt/ index.json)
ROOT=${ROOT:-/data/udpnet_haze_asm_v2}          # copy of <AerialMetric>/udpnet_haze_asm_v2
UDP=${UDP:-$HOME/UDPNet-main/Dehazing/Haze4K}   # the UDPNet folder with main.py / test.py / summarize.py

# ------------------------------------------------------------------ recipe (same for all variants)
VARIANTS=${VARIANTS:-"dav2 gt const"}
SAMPLES=${SAMPLES:-3000000}       # training images seen per run (= Haze4K paper: 3000 x 1000 epochs)
PRECISION=${PRECISION:-32-true}
WORKERS=${WORKERS:-6}
VALID_FREQ=${VALID_FREQ:-3}
SAVE_FREQ=${SAVE_FREQ:-10}
VAL_CROP=${VAL_CROP:-512}         # centre crop for checkpoint selection only (3 runs share one GPU); test is full-res
PARALLEL=${PARALLEL:-1}           # RTX PRO 4500 (32 GB): one run already takes 23 GB -> use PARALLEL=0 there
COMPILE=${COMPILE:-0}             # 1 = --compile (torch.compile on the training step, still fp32)
ACCUM=${ACCUM:-2}                 # accumulate_grad_batches: batch 6 x 2 = effective 12 = upstream DDP on 2 GPUs (paper)
LR=${LR:-1e-4}                    # 4e-4 (paper) blew up -> collapsed at ~19k steps on haze_asm_v2 (NOTES 10-08); 1e-4 passed 25k
MONITOR_FREQ=${MONITOR_FREQ:-100} # max |activation| of Encoder.2 / Decoder / outputs every N optimizer steps (0 = off)
PROBE_FREQ=${PROBE_FREQ:-5000}    # ... and of the saved blow-up batch every N steps
PROBE_BATCH=${PROBE_BATCH:-$UDP/results/probe/nan_batch_1008.pt}
MAX_RETRY=${MAX_RETRY:-10}        # restarts per run after a crash (e.g. Xid 8 GPU hang), from last.ckpt
MODEL_PREFIX=${MODEL_PREFIX:-am_v2}
RES=$UDP/results/summary_${MODEL_PREFIX}

best_ckpt () {   # highest val_psnr checkpoint of a run, else last.ckpt
  local d="$UDP/results/$1/checkpoints" b
  b=$(ls "$d"/epoch_epoch=*_psnr_val_psnr=*.ckpt 2>/dev/null \
      | sed -E 's/.*val_psnr=([0-9.]+)\.ckpt/\1 &/' | sort -rn | head -1 | cut -d' ' -f2 || true)
  if [[ -n "$b" ]]; then echo "$b"; else echo "$d/last.ckpt"; fi
}

stage_prep () {
  python "$HERE/make_dav2_depth.py" --haze "$HAZE" --root "$ROOT"
  python "$HERE/link_layout.py" --haze "$HAZE" --root "$ROOT" --variants $VARIANTS
}

train_one () {
  local v=$1 n ep try
  local ck="$UDP/results/${MODEL_PREFIX}_$v/checkpoints"
  if [[ -f $ck/Final.ckpt ]]; then echo "[$v] Final.ckpt exists -> skip"; return; fi
  n=$(ls "$ROOT/udpnet_$v/train/hazy" | wc -l)
  ep=$(( (SAMPLES + n - 1) / n ))
  echo "[$v] $n training images -> $ep epochs ($SAMPLES samples)"
  cd "$UDP"
  local extra=()
  [[ "$COMPILE" == 1 ]] && extra+=(--compile)
  for ((try = 0; try <= MAX_RETRY; try++)); do
    local resume=()
    [[ -f $ck/last.ckpt ]] && resume=(--resume "$ck/last.ckpt")
    echo "[$v] start $(date '+%m-%d %H:%M') (try $try${resume:+, resume from last.ckpt})"
    if python main.py --model_name "${MODEL_PREFIX}_$v" --data_dir "$ROOT/udpnet_$v" \
      --num_epoch "$ep" --batch_size 6 --patch_size 256 --learning_rate "$LR" \
      --precision "$PRECISION" --num_worker "$WORKERS" \
      --val_split val --val_crop "$VAL_CROP" --valid_freq "$VALID_FREQ" --save_freq "$SAVE_FREQ" \
      --monitor_freq "$MONITOR_FREQ" --probe_batch "$PROBE_BATCH" --probe_freq "$PROBE_FREQ" \
      --accum "$ACCUM" --logs "${MODEL_PREFIX}_$v" "${extra[@]}" "${resume[@]}" >> "$UDP/train_${MODEL_PREFIX}_$v.log" 2>&1; then
      echo "[$v] done $(date '+%m-%d %H:%M')"; return
    fi
    echo "[$v] FAILED $(date '+%m-%d %H:%M') (try $try) -> restart in 60 s"; sleep 60
  done
  echo "[$v] gave up after $MAX_RETRY restarts"; return 1
}

stage_train () {
  if [[ "$PARALLEL" == 1 ]]; then
    for v in $VARIANTS; do train_one "$v" & done
    wait
  else
    for v in $VARIANTS; do train_one "$v"; done
  fi
}

stage_eval () {
  mkdir -p "$RES"
  cd "$UDP"
  for v in $VARIANTS; do
    local ck; ck=$(best_ckpt "${MODEL_PREFIX}_$v")
    echo "[$v] checkpoint: $ck"
    for s in test test_heldout; do
      python test.py --data_dir "$ROOT/udpnet_$v" --split "$s" --test_model "$ck" \
        --model_name "eval_${MODEL_PREFIX}_${v}_$s" --json_out "$RES/${v}__${s}.json"
    done
  done
}

stage_summary () { cd "$UDP" && python summarize.py "$RES"; }

case "${1:-all}" in
  prep)    stage_prep ;;
  train)   stage_train ;;
  eval)    stage_eval ;;
  summary) stage_summary ;;
  all)     stage_prep; stage_train; stage_eval; stage_summary ;;
  *) echo "unknown stage $1"; exit 1 ;;
esac
