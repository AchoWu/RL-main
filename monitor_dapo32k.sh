#!/usr/bin/env bash
# DAPO-32k 4 组串行训练的看护脚本：判断训练是否卡死/退出，必要时自动重启。
#
# 判定逻辑（保守，宁可漏报也不误杀）：
#   1. launcher 进程活着 + 日志在 STALL_SEC 内有更新        -> OK
#   2. launcher 活着但日志超过 STALL_SEC 没动 + GPU 空闲    -> 卡死，重启
#   3. launcher 不在了 + 日志尾部有 "all requested groups"  -> 正常跑完，不动
#   4. launcher 不在了 且未跑完                             -> 异常退出，重启
#
# 重启时复用原始 OPD_DATESTR，让 checkpoint_dir 指回同一目录，从最近断点续跑
# （distillation.py 的 get_latest_checkpoint_path 会自动加载）。
#
# 用法: bash monitor_dapo32k.sh
set -uo pipefail

REPO=/group/40092/howu/RL-main
REPORT="$REPO/logs/dapo32k-smoke-summary.txt"
STATE="$REPO/logs/.dapo32k-monitor-state"
# 训练的原始启动日期，重启后须保持一致，否则会另起一个空 checkpoint 目录
ORIG_DATESTR=260824
# 单步约 230s，val 约 630s，worker 初始化最长约 800s；取 90 分钟做卡死阈值
STALL_SEC=5400
MAX_RESTARTS=5

cd "$REPO"

log() { echo "[$(date '+%F %T')] $*" >> "$REPORT"; }

latest_log() { ls -t "$REPO"/logs/dapo32k-4runs-*.log 2>/dev/null | head -1; }

restarts=0
[[ -f "$STATE" ]] && restarts=$(cat "$STATE" 2>/dev/null || echo 0)

L="$(latest_log)"
if [[ -z "$L" ]]; then
  log "CHECK: 找不到 dapo32k-4runs 日志，跳过（可能尚未启动）"
  exit 0
fi

# NOTE: `pgrep -f <pat>` also matches this script's own command line (the
# pattern string appears in it), so it never reports 0. Scan `ps` directly and
# exclude our own PID plus anything that is a descendant of this script.
count_procs() {
  local pat="$1"
  ps -eo pid=,command= 2>/dev/null \
    | grep -E "$pat" \
    | grep -v grep \
    | awk -v self="$$" -v parent="$PPID" '$1 != self && $1 != parent' \
    | wc -l | tr -d ' \n'
}

alive=$(count_procs "bash .*run_opd_dapo32k_sweep\.sh")
train_alive=$(count_procs "python .*run_distillation_math\.py")
age=$(( $(date +%s) - $(stat -c %Y "$L") ))
gpu_mem=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits 2>/dev/null | paste -sd+ | bc 2>/dev/null | tr -d '\n' || echo 0)
gpu_util=$(nvidia-smi --query-gpu=utilization.gpu --format=csv,noheader,nounits 2>/dev/null | sort -rn | head -1 | tr -d '\n' || echo 0)
cur_group=$(grep -aoE "group=[a-z-]+" "$L" | tail -1 | sed 's/group=//')
done_groups=$(grep -ac "^\[done\] group=" "$L" 2>/dev/null | tr -d ' \n')
done_groups=${done_groups:-0}
finished=$(grep -qa "all requested groups finished" "$L" && echo yes || echo no)

status="group=${cur_group:-?} done=${done_groups}/4 log_age=${age}s gpu=${gpu_mem}MiB util=${gpu_util}% sweep=${alive} train=${train_alive}"

# --- 情况 3: 全部正常跑完 ---
if [[ "$finished" == "yes" ]]; then
  log "CHECK: 全部 4 组已完成 ($status) — 无需干预"
  rm -f "$STATE"
  exit 0
fi

# --- 情况 1: 一切正常 ---
if [[ "$alive" -gt 0 && "$age" -lt "$STALL_SEC" ]]; then
  log "OK: 训练正常 ($status)"
  exit 0
fi

# --- 需要重启的两种情况 ---
reason=""
if [[ "$alive" -eq 0 ]]; then
  reason="launcher 已退出但未跑完 (done=${done_groups}/4)"
elif [[ "$age" -ge "$STALL_SEC" && "$gpu_util" -lt 5 ]]; then
  reason="日志 ${age}s 无更新且 GPU 空闲(util=${gpu_util}%)，判定卡死"
else
  # 日志停了但 GPU 还在算 —— 可能只是长时间生成，不动它
  log "WARN: 日志 ${age}s 未更新，但 GPU util=${gpu_util}% 仍在计算，暂不干预 ($status)"
  exit 0
fi

if [[ "$restarts" -ge "$MAX_RESTARTS" ]]; then
  log "FAIL: $reason；已重启 ${restarts} 次达上限，停止自动重启，请人工介入"
  exit 1
fi

log "RESTART: $reason ($status)"

# 清理残留：卡死的 launcher / 训练进程 / Ray 集群，否则新的起不来。
# 按 PID 精确 kill 并排除自身 —— `pkill -f run_opd_dapo32k_sweep.sh` 会匹配到
# 这个监控脚本自己的命令行，导致监控进程自杀（实测 exit 144）。
kill_matching() {
  local pat="$1" pid
  for pid in $(ps -eo pid=,command= 2>/dev/null | grep -E "$pat" | grep -v grep \
      | awk -v self="$$" -v parent="$PPID" '$1 != self && $1 != parent {print $1}'); do
    kill "$pid" 2>/dev/null
  done
}
kill_matching "bash .*run_opd_dapo32k_sweep\.sh"
kill_matching "python .*run_distillation_math\.py"
sleep 5
env -u TMPDIR /data/miniconda3/envs/opd/bin/ray stop --force >/dev/null 2>&1
sleep 5

# 已完成的组不再重跑：把它们从待跑列表里去掉
ALL=(nomask-reverse seqmean tokmean-refkl seqmean-refkl)
REMAIN=()
for g in "${ALL[@]}"; do
  if grep -qa "^\[done\] group=${g}$" "$L"; then
    log "  skip 已完成: $g"
  else
    REMAIN+=("$g")
  fi
done

if [[ ${#REMAIN[@]} -eq 0 ]]; then
  log "  没有待跑的组，退出"
  rm -f "$STATE"
  exit 0
fi

TS=$(date +%Y%m%d-%H%M%S)
NEW_LOG="$REPO/logs/dapo32k-4runs-${TS}.log"
log "  重启剩余组: ${REMAIN[*]}  -> $(basename "$NEW_LOG")  (OPD_DATESTR=$ORIG_DATESTR 续跑)"

nohup env -u TMPDIR -u TMPPREFIX OPD_DATESTR="$ORIG_DATESTR" \
  bash "$REPO/run_opd_dapo32k_sweep.sh" "${REMAIN[@]}" > "$NEW_LOG" 2>&1 &
echo $! > "${NEW_LOG}.pid"

echo $(( restarts + 1 )) > "$STATE"
log "  已启动 PID=$(cat "${NEW_LOG}.pid") (第 $(( restarts + 1 )) 次重启)"
