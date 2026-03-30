#!/bin/bash
# ============================================================
#  batch_train.sh v6  —  并行批量跑比赛，训练数据按场次隔离
#  每场比赛有独立的match_N目录，训练CSV不会互相覆盖
# ============================================================

set -uo pipefail

# ---------- 默认参数 ----------
TEAM_A_NAME="HELIOS_base"
TEAM_B_NAME="HELIOS_base"
PARALLEL=4
TOTAL_MATCHES=100
BASE_PORT=6000
PORT_SPACING=100
MATCH_TIMEOUT=600
USE_XVFB=true
SERVER_BIN="rcssserver"
LOG_DIR="./match_logs"
TEAM_DIR="."

usage() {
    cat <<EOF
用法: $0 [选项]

选项:
  -d DIR     start.sh 所在目录 (必填)
  -a NAME    己方队名 (默认: HELIOS_base)
  -b NAME    对方队名 (默认: HELIOS_base)
  -p N       并行实例数 (默认: 4)
  -m N       总比赛场数 (默认: 100)
  --timeout SEC  单场超时秒数 (默认: 600)
  --no-xvfb  不使用 xvfb
  -s BIN     rcssserver 路径 (默认: rcssserver)
  -l DIR     根输出目录 (默认: ./match_logs)

目录结构:
  match_logs/
  ├── match_001/
  │   ├── server.log
  │   ├── incomplete.rcg  →  *.rcg (比赛完成后重命名)
  │   ├── mad_training_data/*.csv
  │   └── pass_training_data/*.csv
  ├── match_002/
  │   └── ...
  └── ...

示例:
  $0 -d ~/helios-base/build/bin -p 4 -m 800 -l ~/match_logs

EOF
}

while [[ $# -gt 0 ]]; do
    case "$1" in
        -d) TEAM_DIR="$2"; shift 2 ;;
        -a) TEAM_A_NAME="$2"; shift 2 ;;
        -b) TEAM_B_NAME="$2"; shift 2 ;;
        -p) PARALLEL="$2"; shift 2 ;;
        -m) TOTAL_MATCHES="$2"; shift 2 ;;
        --timeout) MATCH_TIMEOUT="$2"; shift 2 ;;
        --no-xvfb) USE_XVFB=false; shift ;;
        -s) SERVER_BIN="$2"; shift 2 ;;
        -l) LOG_DIR="$2"; shift 2 ;;
        -h) usage; exit 0 ;;
        *) echo "未知选项: $1"; usage; exit 1 ;;
    esac
done

# ---------- 前置检查 ----------
if [[ "$TEAM_DIR" != /* ]]; then
    TEAM_DIR=$(cd "$TEAM_DIR" 2>/dev/null && pwd || { echo "错误: 找不到目录 $TEAM_DIR"; exit 1; })
fi

START_SH="${TEAM_DIR}/start.sh"
[[ ! -f "$START_SH" ]] && { echo "错误: 找不到 $START_SH"; exit 1; }
command -v "$SERVER_BIN" &>/dev/null || { echo "错误: 找不到 $SERVER_BIN"; exit 1; }

mkdir -p "$LOG_DIR"

echo "========================================"
echo "  队伍目录:  $TEAM_DIR"
echo "  输出目录:  $LOG_DIR"
echo "  并行数:    $PARALLEL"
echo "  总场次:    $TOTAL_MATCHES"
echo "  每场超时:  ${MATCH_TIMEOUT}s"
echo "========================================"
echo ""

# ---------- 单场比赛函数 ----------
run_match() {
    local match_id=$1
    local port=$2
    local tag="M$(printf '%03d' $match_id)P$port"

    # 每场比赛独立目录：RCG + 训练数据都在这里
    local match_dir="${LOG_DIR}/match_$(printf '%03d' $match_id)"
    mkdir -p "$match_dir"

    local server_log="${match_dir}/server.log"

    # 为训练数据准备符号链接：conf/ formations/ 等
    # helios 的 start.sh 需要 CWD 下有 conf/ 目录
    for item in conf formations; do
        if [[ -e "$TEAM_DIR/$item" && ! -L "$match_dir/$item" ]]; then
            ln -sf "$TEAM_DIR/$item" "$match_dir/$item"
        fi
    done

    # 启动 server，RCG 写到 match_dir
    if [[ "$USE_XVFB" == true ]]; then
        xvfb-run -a "$SERVER_BIN" \
            "server::port=${port}" \
            "server::coach_port=$((port + 1))" \
            "server::olcoach_port=$((port + 2))" \
            "server::game_logging=on" \
            "server::game_log_dir=${match_dir}" \
            "server::auto_mode=on" \
            "server::synch_mode=off" \
            >"$server_log" 2>&1 &
    else
        "$SERVER_BIN" \
            "server::port=${port}" \
            "server::coach_port=$((port + 1))" \
            "server::olcoach_port=$((port + 2))" \
            "server::game_logging=on" \
            "server::game_log_dir=${match_dir}" \
            "server::auto_mode=on" \
            "server::synch_mode=off" \
            >"$server_log" 2>&1 &
    fi
    local server_pid=$!
    sleep 2

    if ! kill -0 "$server_pid" 2>/dev/null; then
        echo "  [$tag] Server 启动失败"
        return 1
    fi

    # 启动两支队伍，cd 到 match_dir 使训练数据写入这里
    ( cd "$match_dir" && bash "$START_SH" -p "$port" -t "$TEAM_A_NAME" -C ) >/dev/null 2>&1 &
    ( cd "$match_dir" && bash "$START_SH" -p "$port" -t "$TEAM_B_NAME" -C ) >/dev/null 2>&1 &
    sleep 3

    if ! kill -0 "$server_pid" 2>/dev/null; then
        echo "  [$tag] Server 在队伍连接阶段崩溃"
        return 1
    fi

    # 等待比赛完成
    local elapsed=0
    while (( elapsed < MATCH_TIMEOUT )); do
        if [[ ! -f "${match_dir}/incomplete.rcg" ]]; then
            break
        fi
        if ! kill -0 "$server_pid" 2>/dev/null; then
            break
        fi
        sleep 5
        ((elapsed += 5))
    done

    # 清理
    kill -9 "$server_pid" 2>/dev/null || true
    pkill -9 -f "start.sh.*port.*${port}" 2>/dev/null || true
    sleep 0.3

    # 统计本场数据
    local mad_files=$(find "$match_dir" -name "mad_*.csv" 2>/dev/null | wc -l)
    local pass_files=$(find "$match_dir" -name "pass_*.csv" 2>/dev/null | wc -l)
    local rcg_files=$(find "$match_dir" -name "*.rcg" ! -name "incomplete.rcg" 2>/dev/null | wc -l)

    echo "  [$tag] ${elapsed}s | RCG:${rcg_files} MAD:${mad_files} PASS:${pass_files}"
    return 0
}

# ---------- 主循环 ----------
COMPLETED=0
FAILED=0
NEXT=1

cleanup() {
    echo ""
    echo "清理进程..."
    pkill -9 -f rcssserver 2>/dev/null || true
    pkill -9 -f "start.sh" 2>/dev/null || true
    sleep 1
    echo "  完成: $COMPLETED  失败: $FAILED"
    exit 0
}
trap cleanup SIGINT SIGTERM

declare -a SLOT_PID=()

while (( COMPLETED + FAILED < TOTAL_MATCHES )); do
    for (( s = 0; s < PARALLEL; s++ )); do
        (( NEXT > TOTAL_MATCHES )) && break 2

        # slot 有任务在跑
        if [[ -n "${SLOT_PID[$s]:-}" ]] && kill -0 "${SLOT_PID[$s]}" 2>/dev/null; then
            continue
        fi

        # slot 空，收集结果
        if [[ -n "${SLOT_PID[$s]:-}" ]]; then
            wait "${SLOT_PID[$s]}"
            [[ $? -eq 0 ]] && ((COMPLETED++)) || ((FAILED++))
        fi

        # 启动新比赛
        match_id=$NEXT
        port=$((BASE_PORT + s * PORT_SPACING))
        ((NEXT++))

        echo -n "$(date '+%H:%M:%S') [$(printf '%d+%d' $((COMPLETED+FAILED+1)) $TOTAL_MATCHES)] Slot#$s"
        run_match "$match_id" "$port" &
        SLOT_PID[$s]=$!
    done
    sleep 1
done

# 等剩余
for (( s = 0; s < PARALLEL; s++ )); do
    [[ -n "${SLOT_PID[$s]:-}" ]] || continue
    wait "${SLOT_PID[$s]}"
    [[ $? -eq 0 ]] && ((COMPLETED++)) || ((FAILED++))
done

echo ""
echo "========================================"
echo "  完成: $COMPLETED  失败: $FAILED  总计: $TOTAL_MATCHES"
echo "  数据目录: $LOG_DIR"
echo "========================================"

# 汇总统计
total_rcg=$(find "$LOG_DIR" -name "*.rcg" ! -name "incomplete.rcg" 2>/dev/null | wc -l)
total_mad=$(find "$LOG_DIR" -name "mad_*.csv" 2>/dev/null | wc -l)
total_pass=$(find "$LOG_DIR" -name "pass_*.csv" 2>/dev/null | wc -l)
echo "  RCG文件: $total_rcg  MAD CSV: $total_mad  PASS CSV: $total_pass"
