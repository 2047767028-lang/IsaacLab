#!/usr/bin/env bash
# Resume an interrupted pi05 LoRA run from its latest checkpoint, then evaluate it closed-loop.
#
# Written for the lab server (every path below is that machine's) and kept in this repo because
# ~/dev/mimic_enhance there has no version control. First use: the mixed-dataset seed-777 run died
# silently at step ~8520 of 10000 on 2026-08-29 (no traceback, the process just vanished), leaving
# only its step-5000 checkpoint, and the original mixed_s777_pipeline.sh exited on the missing
# step-9999 directory. Resuming costs the remaining ~5000 steps (~4.6 h) instead of a ~9 h retrain.
#
# usage: pi05_resume_and_eval.sh <tag> <config_name> <repo_id> <train_seed> <train_gpu> <sim_gpu> [steps]
#   e.g. pi05_resume_and_eval.sh mixed pi05_mimicgen_mixed_half pi05_lerobot_mixed_half 777 0 2
#
# Training runs as a child of this script, so nothing here waits on `pgrep -f` (see
# docs/arc_sweep_diagnosis/EXPERIMENT_LEDGER.md for how such queues hang). The one thing verified
# before letting it run on is that the resume actually took: the first progress line must start at
# or after the restored step. Without that check, a fresh run from step 0 would look exactly like a
# successful resume four hours later.
set -u
TAG=${1:?tag}; CONFIG=${2:?config_name}; REPO=${3:?repo_id}; SEED=${4:?train_seed}
TGPU=${5:?train_gpu}; SGPU=${6:?sim_gpu}; STEPS=${7:-10000}

MIMIC=/home/surf2/dev/mimic_enhance
LAB=$MIMIC/IsaacLab_perturbation
OPENPI=/home/surf2/dev/geniesim/genie_sim/openpi
LOGS=$MIMIC/isaac_downloads/logs
RESULTS=$MIMIC/eval_results_v2
EXP=lora_1gpu_b32_${STEPS}_anneal_s${SEED}
CKPT=$MIMIC/openpi_smoke_checkpoints/$CONFIG/$EXP
OUT=$RESULTS/${TAG}_s${SEED}.json
TRAIN_LOG=$LOGS/train_${TAG}_s${SEED}_resume.log
SERVE_LOG=$LOGS/serve_${TAG}_s${SEED}.log
EVAL_LOG=$LOGS/eval_${TAG}_s${SEED}.log
PORT=8601
FINAL=$((STEPS-1))

say () { echo "[$(date '+%F %T')] $*"; }

[ -f "$OUT" ] && { say "已有结果 $OUT, 跳过"; exit 0; }
LAST=$(ls -d "$CKPT"/[0-9]* 2>/dev/null | xargs -rn1 basename | sort -n | tail -1)
[ -n "$LAST" ] || { say "!!! $CKPT 下没有任何 checkpoint, 无从续训"; exit 1; }
[ -d "$CKPT/$LAST/train_state" ] || { say "!!! checkpoint $LAST 缺 train_state, 不能续训"; exit 1; }
say "从 step $LAST 续训到 $STEPS (GPU $TGPU), 训练日志 $TRAIN_LOG"

cd "$OPENPI" || exit 1
HOME=/tmp CUDA_VISIBLE_DEVICES="$TGPU" \
PI05_RESUME=1 \
PI05_SMOKE_FSDP_DEVICES=1 PI05_SMOKE_BATCH_SIZE=32 \
PI05_SMOKE_NUM_STEPS="$STEPS" PI05_SMOKE_SAVE_INTERVAL=$((STEPS/2)) \
PI05_SMOKE_EXP_NAME="$EXP" \
PI05_SMOKE_REPO_ID="$REPO" PI05_SMOKE_CONFIG_NAME="$CONFIG" \
PI05_WARMUP_STEPS=500 PI05_TRAIN_SEED="$SEED" \
PYTHONPATH=$OPENPI/src:$OPENPI JAX_PLATFORMS=cuda \
  "$OPENPI/.venv/bin/python" "$MIMIC/train_pi05_lora.py" > "$TRAIN_LOG" 2>&1 < /dev/null &
TPID=$!

# Did the resume take? Wait for the first progress line and check where the counter starts.
w=0
while ! grep -q "Progress on:" "$TRAIN_LOG" 2>/dev/null; do
  kill -0 "$TPID" 2>/dev/null || { say "!!! 训练进程在第一条进度行之前就退出了, 见 $TRAIN_LOG"; exit 1; }
  sleep 15; w=$((w+15))
  [ $w -gt 1800 ] && { say "!!! 30 分钟没有进度行, 杀掉"; kill "$TPID"; exit 1; }
done
FIRST=$(grep -m1 -oE "Progress on: [0-9.]+k?it" "$TRAIN_LOG" | grep -oE "[0-9.]+k?it")
case "$FIRST" in
  *kit) FIRST_N=$(awk "BEGIN{print int(${FIRST%kit}*1000)}") ;;
  *)    FIRST_N=${FIRST%it} ;;
esac
say "第一条进度行: $FIRST (= step $FIRST_N, 期望 >= $LAST)"
if [ "$FIRST_N" -lt "$LAST" ]; then
  say "!!! 续训没有生效 (从 step $FIRST_N 开始), 杀掉以免覆盖已有 checkpoint"; kill "$TPID"; exit 1
fi
grep -q "Found 1 checkpoint steps" "$TRAIN_LOG" && say "orbax 找到 1 个已有 checkpoint, 续训确认"

wait "$TPID"; RC=$?
say "训练结束 rc=$RC"
[ -d "$CKPT/$FINAL" ] || { say "!!! 训练结束但 $CKPT/$FINAL 不存在, 退出"; exit 1; }

# The checkpoint writes assets/norm_stats.json flat; the loader looks for assets/<repo_id>/norm_stats.json.
for d in "$CKPT"/[0-9]*; do
  [ -f "$d/assets/norm_stats.json" ] || continue
  mkdir -p "$d/assets/$REPO"; cp -n "$d/assets/norm_stats.json" "$d/assets/$REPO/norm_stats.json"
done

say "启动 policy server (GPU $TGPU, port $PORT, step $FINAL)"
PYTHONPATH=$OPENPI/src:$OPENPI PI05_SERVE_CKPT_STEP=$FINAL PI05_SERVE_PORT=$PORT \
PI05_SMOKE_REPO_ID=$REPO PI05_SMOKE_CONFIG_NAME=$CONFIG PI05_SMOKE_EXP_NAME=$EXP \
CUDA_VISIBLE_DEVICES="$TGPU" \
  "$OPENPI/.venv/bin/python" "$MIMIC/serve_pi05_checkpoint.py" > "$SERVE_LOG" 2>&1 < /dev/null &
SPID=$!
w=0
while ! grep -q "server listening" "$SERVE_LOG" 2>/dev/null; do
  grep -qE "Traceback|Error" "$SERVE_LOG" 2>/dev/null && { say "!!! server 失败, 见 $SERVE_LOG"; kill "$SPID"; exit 1; }
  kill -0 "$SPID" 2>/dev/null || { say "!!! server 进程退出, 见 $SERVE_LOG"; exit 1; }
  sleep 5; w=$((w+5)); [ $w -gt 900 ] && { say "!!! server 超时"; kill "$SPID"; exit 1; }
done
say "server 就绪, 开跑 300 次 (Isaac Sim 用 GPU $SGPU), 评测日志 $EVAL_LOG"

cd "$LAB" || { kill "$SPID"; exit 1; }
env PATH="$MIMIC/conda_envs/isaaclab_py311/bin:$PATH" \
    LD_LIBRARY_PATH="$MIMIC/conda_envs/isaaclab_py311/lib:${LD_LIBRARY_PATH:-}" \
    CUDA_VISIBLE_DEVICES="$SGPU" OMNI_KIT_ACCEPT_EULA=YES PYTHONUNBUFFERED=1 \
  ./isaaclab.sh -p scripts/imitation_learning/openpi/play_policy_server.py \
    --task Isaac-Stack-Cube-Franka-IK-Rel-Visuomotor-v0 \
    --policy_host localhost --policy_port "$PORT" --prompt "stack the cubes" \
    --num_rollouts 300 --horizon 400 --seed 101 \
    --reset_joint_std 0.0 --results_file "$OUT" \
    --enable_cameras --headless --device cuda:0 \
    > "$EVAL_LOG" 2>&1
say "eval 结束 rc=$?"
kill "$SPID" 2>/dev/null; sleep 15; kill -9 "$SPID" 2>/dev/null
if [ -f "$OUT" ]; then
  printf 'resumed from step %s (training interrupted 2026-08-29); data iterator restarted on resume, see train_pi05_lora.py PI05_RESUME\n' "$LAST" > "${OUT%.json}.RESUMED.txt"
fi
say "########## $TAG seed=$SEED 完成 ##########"
