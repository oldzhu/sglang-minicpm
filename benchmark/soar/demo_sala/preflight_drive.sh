#!/bin/bash
# SOAR CHANGE_0165 — Pre-flight runner (executes on fcloud).
#
# Runs sglang server twice (once NGRAM, once MEDUSA), sends one short prompt
# each time, and captures the verify-step state dumps to /tmp/dump_*.pkl.
#
# Caller passes mode as $1:  ngram | medusa
#
# Expects to be invoked from /root/submission_sim with prepare_env.sh,
# preflight_run.py, etc available in /root/sglang-minicpm.

# NOTE: We intentionally do NOT `set -e` here.  prepare_env.sh runs
# `uv pip install --force-reinstall pypcre -v` which fails on fcloud (clang
# missing) but is non-fatal because pypcre is already installed.  The
# regular fcloud_workflow restart-server also tolerates this.

MODE="${1:?usage: $0 ngram|medusa}"
DUMP_PATH="/tmp/dump_${MODE}.pkl"
LOG_PATH="/tmp/server_${MODE}.log"
PROMPT_LOG="/tmp/preflight_run_${MODE}.log"

cd /root/submission_sim

# Kill any existing server.
pkill -f "sglang.launch_server" 2>/dev/null || true
sleep 3

# Clean previous dump.
rm -f "$DUMP_PATH"

# Configure spec algo via env BEFORE sourcing prepare_env.sh.
case "$MODE" in
    ngram)
        export SOAR_SPEC_MEDUSA=0
        export SOAR_SPEC_NGRAM=1
        # Force ndt=2 for apples-to-apples comparison with MEDUSA (heads=1, ndt=2).
        export SGLANG_SERVER_ARGS=" --speculative-num-draft-tokens 2 "
        ;;
    medusa)
        export SOAR_SPEC_MEDUSA=1
        export SOAR_SPEC_NGRAM=0
        export SOAR_SPEC_MEDUSA_HEADS=1
        ;;
    *)
        echo "unknown mode: $MODE" >&2
        exit 2
        ;;
esac

export SOAR_PREFLIGHT_DUMP_PATH="$DUMP_PATH"
export SOAR_QUANT_MODE="${SOAR_QUANT_MODE:-gptq}"

# Source prepare_env.sh.
source ./prepare_env.sh

echo "[preflight] mode=$MODE"
echo "[preflight] SGLANG_SERVER_ARGS=$SGLANG_SERVER_ARGS"
echo "[preflight] SOAR_PREFLIGHT_DUMP_PATH=$SOAR_PREFLIGHT_DUMP_PATH"

# Launch server in background.
read -r -a EXTRA_ARGS <<< "${SGLANG_SERVER_ARGS:-}"
nohup python3 -m sglang.launch_server \
    --model-path "$MODEL_PATH" \
    --host 0.0.0.0 \
    --port 30000 \
    "${EXTRA_ARGS[@]}" \
    >"$LOG_PATH" 2>&1 &

SERVER_PID=$!
echo "[preflight] server pid=$SERVER_PID"

# Wait for server health (max 600s).
HEALTHY=0
for i in $(seq 1 120); do
    sleep 5
    if curl -sf http://127.0.0.1:30000/health_generate -o /dev/null 2>/dev/null; then
        HEALTHY=1
        echo "[preflight] server healthy after $((i*5))s"
        break
    fi
    if ! kill -0 $SERVER_PID 2>/dev/null; then
        echo "[preflight] server died early; tail log:" >&2
        tail -30 "$LOG_PATH" >&2
        exit 3
    fi
done

if [[ $HEALTHY -ne 1 ]]; then
    echo "[preflight] server failed to become healthy" >&2
    tail -30 "$LOG_PATH" >&2
    kill $SERVER_PID 2>/dev/null || true
    exit 4
fi

# Send the preflight prompt.
python3 /root/sglang-minicpm/benchmark/soar/demo_sala/preflight_run.py \
    --url http://127.0.0.1:30000 \
    --max-new-tokens 3 \
    >"$PROMPT_LOG" 2>&1 || true

echo "[preflight] prompt response:"
cat "$PROMPT_LOG"

# Give server a moment to flush the dump.
sleep 2

# Kill server.
kill $SERVER_PID 2>/dev/null || true
sleep 3
pkill -f "sglang.launch_server" 2>/dev/null || true

# Verify dump exists.
if [[ ! -s "$DUMP_PATH" ]]; then
    echo "[preflight] DUMP MISSING at $DUMP_PATH" >&2
    echo "[preflight] server log tail:" >&2
    tail -50 "$LOG_PATH" >&2
    exit 5
fi

echo "[preflight] dump: $DUMP_PATH ($(stat -c %s "$DUMP_PATH") bytes)"
echo "[preflight] DONE mode=$MODE"
