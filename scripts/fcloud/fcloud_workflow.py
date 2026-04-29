#!/usr/bin/env python3
"""fcloud_workflow.py - Automated test workflow for fcloud.

Orchestrates: git pull → copy changed files → restart sglang → run tests.

Usage:
    # Full workflow: pull, sync, restart server, run accuracy test
    python3 scripts/fcloud/fcloud_workflow.py full

    # Just sync files (git pull + copy)
    python3 scripts/fcloud/fcloud_workflow.py sync

    # Just restart the server (kill old, start new)
    python3 scripts/fcloud/fcloud_workflow.py restart-server

    # Run accuracy test only (server must be running)
    python3 scripts/fcloud/fcloud_workflow.py accuracy

    # Run speed benchmark (s1, s8, smax)
    python3 scripts/fcloud/fcloud_workflow.py speed --variant s1
    python3 scripts/fcloud/fcloud_workflow.py speed --variant s8
    python3 scripts/fcloud/fcloud_workflow.py speed --variant smax

    # Run all speed benchmarks
    python3 scripts/fcloud/fcloud_workflow.py speed --variant all

    # Wait for server to be ready
    python3 scripts/fcloud/fcloud_workflow.py wait-server

    # Show server logs (last N lines)
    python3 scripts/fcloud/fcloud_workflow.py server-logs --lines 100
"""

import argparse
import json
import os
import re
import subprocess
import sys
import time

# Add parent so we can import fcloud_exec
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import fcloud_exec

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
FCLOUD_REPO = "/root/sglang-minicpm"
FCLOUD_SIM = "/root/submission_sim"
FCLOUD_DATA = "/root/data"
MODEL_PATH = "/root/models/openbmb/MiniCPM-SALA-90-qa-cwe-mcq-sparse_qkv_w8"
FP8_MODEL_PATH = "/root/models/minicpm_fp8_blockwise"
NOQUANT_MODEL_PATH = "/root/models/openbmb/MiniCPM-SALA"
HOST = "0.0.0.0"
PORT = 30000
API_BASE = f"http://127.0.0.1:{PORT}"
LOCAL_REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))

# Server terminal name (stable, so we can reconnect)
SERVER_TERMINAL = None  # Will be set when starting server


def fcloud_run(base_url, token, cmd, timeout=300, cwd=None, background=False):
    """Execute a command on fcloud and return output."""
    name, output = fcloud_exec.exec_command(
        base_url, token, cmd, timeout=timeout, cwd=cwd, background=background
    )
    return name, output


def print_section(title):
    print(f"\n{'='*60}")
    print(f"  {title}")
    print(f"{'='*60}\n")


def get_local_changed_files(path_prefix):
    """Return local modified/tracked/untracked files under a prefix.

    This catches local uncommitted kernel edits that remote `git pull` cannot see.
    """
    changed = set()
    commands = [
        ["git", "-C", LOCAL_REPO, "diff", "--name-only", "HEAD", "--", path_prefix],
        ["git", "-C", LOCAL_REPO, "diff", "--cached", "--name-only", "--", path_prefix],
        ["git", "-C", LOCAL_REPO, "ls-files", "--others", "--exclude-standard", "--", path_prefix],
    ]
    for cmd in commands:
        try:
            out = subprocess.check_output(cmd, text=True, stderr=subprocess.DEVNULL)
        except subprocess.CalledProcessError:
            continue
        for line in out.splitlines():
            line = line.strip()
            if line:
                changed.add(line)
    return sorted(changed)


# ---------------------------------------------------------------------------
# Workflow steps
# ---------------------------------------------------------------------------
def step_sync(base_url, token):
    """Git pull and copy changed files to submission_sim.

    Captures pre-pull HEAD SHA, then diffs against post-pull HEAD so that ALL
    commits pulled (not just the last one) are reflected in the copy list.
    This is important when syncing to a fcloud instance whose repo is many
    commits behind (e.g., after a fresh setup or instance switch).
    """
    print_section("SYNC: git pull + copy files")

    # Capture pre-pull SHA so we can compute the full changed-file set after pull
    _, pre_sha = fcloud_run(
        base_url, token,
        "cd /root/sglang-minicpm && git rev-parse HEAD 2>&1",
        timeout=15,
    )
    pre_sha = pre_sha.strip().split("\n")[-1].strip()
    print(f"[pre-pull sha] {pre_sha}")

    # Git pull with auto-merge message
    _, out = fcloud_run(
        base_url, token,
        "cd /root/sglang-minicpm && git pull --no-edit 2>&1",
        timeout=120,
    )
    print(f"[git pull] {out}")

    _, post_sha = fcloud_run(
        base_url, token,
        "cd /root/sglang-minicpm && git rev-parse HEAD 2>&1",
        timeout=15,
    )
    post_sha = post_sha.strip().split("\n")[-1].strip()
    print(f"[post-pull sha] {post_sha}")

    # Diff against the pre-pull SHA so we see every file changed across ALL
    # commits that were just pulled, not only the most recent commit.
    if pre_sha and post_sha and pre_sha != post_sha and len(pre_sha) >= 7:
        diff_cmd = (
            f"cd /root/sglang-minicpm && "
            f"git diff --name-only {pre_sha} {post_sha} 2>/dev/null || echo 'DIFF_FAILED'"
        )
    else:
        # No pull advance — still compute last-commit diff as a safety net
        diff_cmd = (
            "cd /root/sglang-minicpm && "
            "git diff --name-only HEAD~1 HEAD 2>/dev/null || echo 'DIFF_FAILED'"
        )

    _, changed = fcloud_run(base_url, token, diff_cmd, timeout=30)
    print(f"[changed files]\n{changed}")

    if "DIFF_FAILED" in changed or not changed.strip():
        print("[sync] Could not determine changed files via diff; falling back to "
              "force-copy of python/ and benchmark/soar/demo_sala/ trees")
        changed_files = None  # sentinel for fallback
    else:
        changed_files = [f.strip() for f in changed.strip().split("\n") if f.strip()]

    local_sgl_kernel_changes = get_local_changed_files("sgl-kernel/")
    if local_sgl_kernel_changes:
        print("[local sgl-kernel changes]")
        for path in local_sgl_kernel_changes:
            print(path)

    # Copy files based on path mapping
    copy_cmds = []
    sgl_kernel_changed = False

    if changed_files is None:
        # Fallback: force-copy entire python/ tree and demo_sala flat files.
        # Used when the pre/post-pull diff fails (e.g., shallow clone).
        copy_cmds.append(
            f"cp -r {FCLOUD_REPO}/python/sglang {FCLOUD_SIM}/sglang/python/"
        )
        copy_cmds.append(
            f"cd {FCLOUD_REPO}/benchmark/soar/demo_sala && "
            f"cp -v gptqmodel_minicpm_sala.py preprocess_model.py prepare_env.sh "
            f"prepare_model.sh {FCLOUD_SIM}/ 2>&1 | tail -10"
        )
        # eval scripts -> /root/data
        copy_cmds.append(
            f"cp -v {FCLOUD_REPO}/benchmark/soar/demo_sala/eval_model.py "
            f"{FCLOUD_REPO}/benchmark/soar/demo_sala/eval_model_001.py "
            f"{FCLOUD_DATA}/ 2>&1 | tail -5"
        )
        # Assume sgl-kernel may have changed when we lose the diff
        sgl_kernel_changed = True
    else:
        eval_script_changed = False
        for f in changed_files:
            if not f:
                continue
            if f.startswith("benchmark/soar/demo_sala/"):
                rel = f.replace("benchmark/soar/demo_sala/", "")
                src = f"{FCLOUD_REPO}/{f}"
                # eval_model*.py live in /root/data, everything else in /root/submission_sim
                if rel.startswith("eval_model"):
                    dst = f"{FCLOUD_DATA}/{rel}"
                    eval_script_changed = True
                else:
                    dst = f"{FCLOUD_SIM}/{rel}"
                copy_cmds.append(f"cp -v {src} {dst}")
            elif f.startswith("python/"):
                # Copy to /root/submission_sim/sglang/python/...
                src = f"{FCLOUD_REPO}/{f}"
                dst = f"{FCLOUD_SIM}/sglang/{f}"
                copy_cmds.append(f"mkdir -p $(dirname {dst}) && cp -v {src} {dst}")
            elif f.startswith("sgl-kernel/"):
                sgl_kernel_changed = True
        if eval_script_changed:
            print("[sync] eval_model*.py changes detected → copying to /root/data")

    if local_sgl_kernel_changes:
        sgl_kernel_changed = True
        for rel_path in local_sgl_kernel_changes:
            local_path = os.path.join(LOCAL_REPO, rel_path)
            remote_path = f"{FCLOUD_REPO}/{rel_path}"
            ok = fcloud_exec.upload_file(base_url, token, local_path, remote_path)
            if not ok:
                raise RuntimeError(f"Failed to upload local sgl-kernel change: {rel_path}")
        print(f"[sync] Uploaded {len(local_sgl_kernel_changes)} local sgl-kernel file(s)")

    if copy_cmds:
        cmd = " && ".join(copy_cmds)
        _, out = fcloud_run(base_url, token, cmd, timeout=60)
        print(f"[copy] {out}")
    else:
        print("[sync] No files to copy (or using fallback)")

    if sgl_kernel_changed:
        print("[sync] sgl-kernel changed — building wheel...")
        _, out = fcloud_run(
            base_url, token,
            f"cd {FCLOUD_REPO}/sgl-kernel && pip wheel --no-build-isolation -w dist . 2>&1 | tail -5",
            timeout=600,
        )
        print(f"[sgl-kernel build] {out}")
        _, out = fcloud_run(
            base_url, token,
            f"cp -v {FCLOUD_REPO}/sgl-kernel/dist/sgl_kernel-*.whl {FCLOUD_SIM}/ 2>&1",
            timeout=30,
        )
        print(f"[sgl-kernel copy] {out}")

    print("[sync] Done")


def step_restart_server(base_url, token, quant_mode="gptq", model_path=None):
    """Kill existing sglang server and start a new one.

    Args:
        quant_mode: "gptq" (default) or "fp8_blockwise".
                    Controls SOAR_QUANT_MODE passed to prepare_env.sh.
        model_path: Override model path. Defaults to MODEL_PATH for gptq,
                    FP8_MODEL_PATH for fp8_blockwise.
    """
    print_section("RESTART SERVER")

    if model_path is None:
        if quant_mode == "fp8_blockwise":
            model_path = FP8_MODEL_PATH
        elif quant_mode == "noquant":
            model_path = NOQUANT_MODEL_PATH
        else:
            model_path = MODEL_PATH

    print(f"[restart-server] quant_mode={quant_mode}, model_path={model_path}")

    # Kill existing
    _, out = fcloud_run(
        base_url, token,
        'pkill -f "sglang.launch_server" 2>/dev/null; sleep 2; echo "killed"',
        timeout=15,
    )
    print(f"[kill] {out}")

    # Source prepare_env.sh and start server in background.
    # Export SOAR_QUANT_MODE before sourcing so prepare_env.sh picks up the right branch.
    server_cmd = f"""cd {FCLOUD_SIM} && export SOAR_QUANT_MODE={quant_mode} && source ./prepare_env.sh && \\
MODEL_PATH={model_path} && \\
HOST={HOST} && \\
PORT={PORT} && \\
read -r -a EXTRA_ARGS <<< "${{SGLANG_SERVER_ARGS:-}}" && \\
python3 -m sglang.launch_server \\
  --model-path "$MODEL_PATH" \\
  --host "$HOST" \\
  --port "$PORT" \\
  "${{EXTRA_ARGS[@]}}" 2>&1"""

    term_name, out = fcloud_run(
        base_url, token, server_cmd, background=True
    )
    print(f"[server] {out}")
    return term_name


def step_wait_server(base_url, token, timeout=300):
    """Wait for sglang server to be ready."""
    print_section("WAITING FOR SERVER")
    start = time.time()
    check_cmd = f'curl -s -o /dev/null -w "%{{http_code}}" {API_BASE}/health 2>/dev/null || echo 000'

    while (time.time() - start) < timeout:
        _, out = fcloud_run(base_url, token, check_cmd, timeout=10)
        code = out.strip()
        # Extract just the numeric code in case of extra output
        m = re.search(r"\b(200|000)\b", code)
        status = m.group(1) if m else code
        if status == "200":
            elapsed = int(time.time() - start)
            print(f"[server] Ready after {elapsed}s")
            return True
        print(f"[server] Not ready yet (status={status}), waiting...")
        time.sleep(10)

    print(f"[server] TIMEOUT after {timeout}s")
    return False


def _resolve_model_path(quant_mode="gptq", model_path=None):
    """Pick the eval --model_path that matches the served model.

    eval_model_001.py loads tokenizer + GenerationConfig (eos / stop
    words) from --model_path. It MUST match what the server serves,
    otherwise prompts/stop tokens may differ and generation may run
    away to max_tokens (read timeout).
    """
    if model_path is not None:
        return model_path
    if quant_mode == "fp8_blockwise":
        return FP8_MODEL_PATH
    if quant_mode == "noquant":
        return NOQUANT_MODEL_PATH
    return MODEL_PATH


def step_accuracy(base_url, token, timeout=3600, quant_mode="gptq", model_path=None):
    """Run accuracy evaluation."""
    print_section("ACCURACY TEST")
    eval_model_path = _resolve_model_path(quant_mode, model_path)
    print(f"[accuracy] quant_mode={quant_mode}, eval --model_path={eval_model_path}")
    # Kill any leftover eval processes to avoid duplicate requests
    fcloud_run(base_url, token,
               'pkill -f "eval_model" 2>/dev/null; sleep 1; echo "cleaned"',
               timeout=10)
    cmd = (
        f"cd {FCLOUD_DATA} && python3 eval_model_001.py "
        f"--api_base {API_BASE} "
        f"--model_path {eval_model_path} "
        f"--data_path {FCLOUD_DATA}/perf_public_set.jsonl "
        f"--concurrency 32 2>&1"
    )
    _, out = fcloud_run(base_url, token, cmd, timeout=timeout)
    print(out)
    return out


def step_quick_accuracy(base_url, token, task_filter=None, num_per_task=None, timeout=1200, quant_mode="gptq", model_path=None):
    """Run quick accuracy evaluation with subset of data."""
    label = "QUICK ACCURACY"
    extra = ""
    if task_filter:
        extra += f" --task_filter {task_filter}"
        label += f" (tasks={task_filter})"
    if num_per_task:
        extra += f" --num_samples_per_task {num_per_task}"
        label += f" (per_task={num_per_task})"
    if not task_filter and not num_per_task:
        # Default quick mode: MCQ only (fastest, first 30 samples)
        extra += " --task_filter mcq"
        label += " (mcq-only)"
    print_section(label)
    eval_model_path = _resolve_model_path(quant_mode, model_path)
    print(f"[quick-accuracy] quant_mode={quant_mode}, eval --model_path={eval_model_path}")
    fcloud_run(base_url, token,
               'pkill -f "eval_model" 2>/dev/null; sleep 1; echo "cleaned"',
               timeout=10)
    cmd = (
        f"cd {FCLOUD_DATA} && python3 eval_model_001.py "
        f"--api_base {API_BASE} "
        f"--model_path {eval_model_path} "
        f"--data_path {FCLOUD_DATA}/perf_public_set.jsonl "
        f"--concurrency 32{extra} 2>&1"
    )
    _, out = fcloud_run(base_url, token, cmd, timeout=timeout)
    print(out)
    return out


def step_speed(base_url, token, variant="s1", timeout=600):
    """Run speed benchmark."""
    print_section(f"SPEED TEST: {variant}")

    if variant == "all":
        results = {}
        for v in ["s1", "s8", "smax"]:
            results[v] = step_speed(base_url, token, v, timeout)
        return results

    SPEED_DATA = {
        "s1": "/root/data/speed_s1.jsonl",
        "s8": "/root/data/speed_s8.jsonl",
        "smax": "/root/data/speed_smax.jsonl",
    }

    data_file = SPEED_DATA.get(variant)
    if not data_file:
        print(f"[speed] Unknown variant: {variant}")
        return ""

    # Use bench_serving.sh with the appropriate SPEED_DATA_* env var
    env_var = f"SPEED_DATA_{variant.upper()}"
    cmd = (
        f"cd /root/data && "
        f"{env_var}={data_file} "
        f"bash bench_serving.sh {API_BASE} 2>&1"
    )
    _, out = fcloud_run(base_url, token, cmd, timeout=timeout)
    print(out)
    return out


def step_server_logs(base_url, token, lines=100):
    """Show recent server logs."""
    terms = fcloud_exec.list_terminals(base_url, token)
    if not terms:
        print("[logs] No active terminals")
        return ""

    # Try to get output from terminals
    for t in terms:
        output = fcloud_exec.tail_terminal(base_url, token, t["name"], lines=lines, wait=3)
        if output.strip():
            print(f"--- Terminal: {t['name']} ---")
            print(output)
            return output

    print("[logs] No output captured from terminals")
    return ""


def workflow_full(base_url, token, quant_mode="gptq", model_path=None):
    """Full workflow: sync → restart → wait → accuracy."""
    step_sync(base_url, token)
    server_term = step_restart_server(base_url, token, quant_mode=quant_mode, model_path=model_path)
    ready = step_wait_server(base_url, token, timeout=300)
    if not ready:
        print("[ABORT] Server did not start in time")
        step_server_logs(base_url, token, lines=50)
        return

    accuracy_out = step_accuracy(base_url, token, quant_mode=quant_mode, model_path=model_path)

    # Parse accuracy from output
    m = re.search(r"Average Score:\s*([\d.]+)%", accuracy_out)
    if m:
        score = float(m.group(1))
        print(f"\n{'='*60}")
        print(f"  ACCURACY: {score}%")
        if score >= 80:
            print(f"  STATUS: PASS (>= 80%)")
        else:
            print(f"  STATUS: FAIL (< 80%)")
        print(f"{'='*60}")


# ---------------------------------------------------------------------------
# Setup: bootstrap a clean fcloud instance
# ---------------------------------------------------------------------------
# Local workspace paths (relative to repo root)
LOCAL_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
LOCAL_DEMO_SALA = os.path.join(LOCAL_REPO_ROOT, "benchmark", "soar", "demo_sala")

# Files to sync from demo_sala → /root/submission_sim
DEMO_SALA_TO_SIM = [
    "gptqmodel_minicpm_sala.py",
    "preprocess_model.py",
    "prepare_env.sh",
    "prepare_model.sh",
    "perf_public_set.jsonl",
]

# Files to sync from demo_sala → /root/data
DEMO_SALA_TO_DATA = [
    "eval_model_001.py",
    "eval_model.py",
]


def step_setup(base_url, token, skip_existing=True):
    """Bootstrap a clean fcloud instance with all required files.

    Steps:
      1. Check existing paths — skip setup if already done
      2. Git clone sglang-minicpm repo
      3. Upload & extract submission_sim.tar → /root/submission_sim
      4. Copy python/ from repo → submission_sim/sglang/python/
      5. Sync demo_sala scripts → /root/submission_sim
      6. Upload & extract data.tar.gz → /root/data
      7. Sync eval scripts → /root/data
    """
    print_section("SETUP: Bootstrap fcloud instance")

    # Step 1: Check existing paths
    needs_repo = True
    needs_sim = True
    needs_data = True

    if skip_existing:
        needs_repo = not fcloud_exec.path_exists(base_url, token, FCLOUD_REPO)
        needs_sim = not fcloud_exec.path_exists(base_url, token, FCLOUD_SIM)
        needs_data = not fcloud_exec.path_exists(base_url, token, FCLOUD_DATA)
        print(f"[setup] Needs: repo={needs_repo}, sim={needs_sim}, data={needs_data}")
        if not needs_repo and not needs_sim and not needs_data:
            print("[setup] All paths exist — running incremental sync only")
            _setup_incremental_sync(base_url, token)
            return

    # Step 2: Upload repo tarball (faster than git clone on slow networks)
    if needs_repo:
        print("[setup] Step 2: Uploading repo tarball...")
        # Create a minimal tarball of needed files
        import subprocess, tempfile
        repo_tar = os.path.join(tempfile.gettempdir(), "sglang-minicpm-repo.tar.gz")
        print(f"  Creating tarball from {LOCAL_REPO_ROOT}...")
        subprocess.run(
            ["tar", "czf", repo_tar,
             "--exclude=*.tar", "--exclude=*.tar.gz",
             "--exclude=.git", "--exclude=test", "--exclude=docs",
             "--exclude=3rdparty", "--exclude=sgl-kernel/benchmark",
             "--exclude=sgl-kernel/tests",
             "python/", "benchmark/soar/demo_sala/", "scripts/fcloud/"],
            cwd=LOCAL_REPO_ROOT, check=True,
        )
        tar_size = os.path.getsize(repo_tar)
        print(f"  Tarball size: {tar_size / 1024 / 1024:.1f} MB")
        # Clean up any partial clone first
        fcloud_run(base_url, token, "rm -rf /root/sglang-minicpm 2>/dev/null; true", timeout=30)
        ok = fcloud_exec.upload_file(base_url, token, repo_tar, "/root/sglang-minicpm-repo.tar.gz")
        if not ok:
            print("  ERROR: Repo tarball upload failed")
            return
        print("[setup]   Extracting...")
        _, out = fcloud_run(
            base_url, token,
            "mkdir -p /root/sglang-minicpm && cd /root/sglang-minicpm && "
            "tar xzf /root/sglang-minicpm-repo.tar.gz && rm -f /root/sglang-minicpm-repo.tar.gz && echo OK",
            timeout=120,
        )
        print(f"  {out}")
        os.unlink(repo_tar)
    else:
        print("[setup] Step 2: Repo exists, syncing latest files...")
        _setup_incremental_sync(base_url, token)
        # Incremental sync handles steps 4-5, 7.
        # Jump to step 6 if data dir is still needed, otherwise done.
        if not needs_data:
            print("[setup] All up to date")
            return
        # Fall through to step 6 only
        needs_sim = False  # skip step 3 (sim already exists per step 1 check)

    # Step 3: Upload & extract submission_sim.tar
    if needs_sim:
        print("[setup] Step 3: Uploading submission_sim.tar...")
        sim_tar_local = os.path.join(LOCAL_DEMO_SALA, "submission_sim.tar")
        if not os.path.isfile(sim_tar_local):
            print(f"  ERROR: Local file not found: {sim_tar_local}")
            print("  Please place submission_sim.tar in benchmark/soar/demo_sala/")
            return
        ok = fcloud_exec.upload_file(base_url, token, sim_tar_local, "/root/submission_sim.tar")
        if not ok:
            print("  ERROR: Upload failed")
            return
        print("[setup]   Extracting...")
        _, out = fcloud_run(
            base_url, token,
            "cd /root && tar xf submission_sim.tar && rm -f submission_sim.tar && echo OK",
            timeout=120,
        )
        print(f"  {out}")
    else:
        print("[setup] Step 3: submission_sim exists, skipping upload")

    # Step 4: Copy python/ from repo to submission_sim/sglang/python/
    print("[setup] Step 4: Copying python/ → submission_sim/sglang/python/...")
    _, out = fcloud_run(
        base_url, token,
        f"mkdir -p {FCLOUD_SIM}/sglang/python && cp -a {FCLOUD_REPO}/python/* {FCLOUD_SIM}/sglang/python/ 2>&1 && echo OK",
        timeout=60,
    )
    print(f"  {out}")

    # Step 5: Sync demo_sala files → /root/submission_sim
    print("[setup] Step 5: Syncing demo_sala scripts to submission_sim...")
    for fname in DEMO_SALA_TO_SIM:
        src = f"{FCLOUD_REPO}/benchmark/soar/demo_sala/{fname}"
        dst = f"{FCLOUD_SIM}/{fname}"
        _, out = fcloud_run(base_url, token, f"cp -v {src} {dst} 2>&1", timeout=15)
        print(f"  {out}")

    # Step 6: Upload & extract data.tar.gz
    if needs_data:
        print("[setup] Step 6: Uploading data.tar.gz...")
        data_tar_local = os.path.join(LOCAL_DEMO_SALA, "data.tar.gz")
        if not os.path.isfile(data_tar_local):
            print(f"  ERROR: Local file not found: {data_tar_local}")
            print("  Please place data.tar.gz in benchmark/soar/demo_sala/")
            return
        ok = fcloud_exec.upload_file(base_url, token, data_tar_local, "/root/data.tar.gz")
        if not ok:
            print("  ERROR: Upload failed")
            return
        print("[setup]   Extracting into /root/data/...")
        _, out = fcloud_run(
            base_url, token,
            "mkdir -p /root/data && cd /root/data && tar xzf /root/data.tar.gz && rm -f /root/data.tar.gz && echo OK",
            timeout=120,
        )
        print(f"  {out}")
    else:
        print("[setup] Step 6: data dir exists, skipping upload")

    # Step 7: Sync eval scripts → /root/data
    print("[setup] Step 7: Syncing eval scripts to /root/data...")
    for fname in DEMO_SALA_TO_DATA:
        src = f"{FCLOUD_REPO}/benchmark/soar/demo_sala/{fname}"
        dst = f"{FCLOUD_DATA}/{fname}"
        _, out = fcloud_run(base_url, token, f"cp -v {src} {dst} 2>&1", timeout=15)
        print(f"  {out}")

    print_section("SETUP COMPLETE")
    _, out = fcloud_run(
        base_url, token,
        f"echo '--- Repo ---' && ls {FCLOUD_REPO}/ | head -5 && "
        f"echo '--- Submission Sim ---' && ls {FCLOUD_SIM}/ | head -10 && "
        f"echo '--- Data ---' && ls {FCLOUD_DATA}/",
        timeout=15,
    )
    print(out)


def _setup_incremental_sync(base_url, token):
    """Incremental sync: pull repo, copy python/ and demo_sala files."""
    # Pull latest
    _, out = fcloud_run(
        base_url, token,
        "cd /root/sglang-minicpm && git pull --no-edit 2>&1",
        timeout=60,
    )
    print(f"[sync] git pull: {out}")

    # Copy python/
    _, out = fcloud_run(
        base_url, token,
        f"cp -a {FCLOUD_REPO}/python/* {FCLOUD_SIM}/sglang/python/ 2>&1 && echo OK",
        timeout=60,
    )
    print(f"[sync] python/ copy: {out}")

    # Sync demo_sala → sim
    for fname in DEMO_SALA_TO_SIM:
        src = f"{FCLOUD_REPO}/benchmark/soar/demo_sala/{fname}"
        dst = f"{FCLOUD_SIM}/{fname}"
        fcloud_run(base_url, token, f"cp -v {src} {dst} 2>&1", timeout=15)

    # Sync demo_sala → data
    for fname in DEMO_SALA_TO_DATA:
        src = f"{FCLOUD_REPO}/benchmark/soar/demo_sala/{fname}"
        dst = f"{FCLOUD_DATA}/{fname}"
        fcloud_run(base_url, token, f"cp -v {src} {dst} 2>&1", timeout=15)

    print("[sync] Incremental sync complete")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description="fcloud automated test workflow")
    sub = parser.add_subparsers(dest="action", required=True)

    p_full = sub.add_parser("full", help="Full workflow: sync → restart → accuracy")
    p_full.add_argument("--quant-mode", choices=["gptq", "fp8_blockwise", "noquant"], default="gptq")
    p_full.add_argument("--model-path", type=str, default=None)
    sub.add_parser("sync", help="Git pull and copy changed files")
    p_restart = sub.add_parser("restart-server", help="Restart sglang server")
    p_restart.add_argument("--quant-mode", choices=["gptq", "fp8_blockwise", "noquant"], default="gptq",
                           help="Quantization mode (default: gptq)")
    p_restart.add_argument("--model-path", type=str, default=None,
                           help="Override model path (default: auto from quant-mode)")
    sub.add_parser("wait-server", help="Wait for server to be ready")
    p_acc = sub.add_parser("accuracy", help="Run accuracy test")
    p_acc.add_argument("--quant-mode", choices=["gptq", "fp8_blockwise", "noquant"], default="gptq",
                       help="Pick eval --model_path matching the served model (default: gptq)")
    p_acc.add_argument("--model-path", type=str, default=None,
                       help="Override eval --model_path (default: auto from quant-mode)")

    p_qacc = sub.add_parser("quick-accuracy", help="Quick accuracy test (subset)")
    p_qacc.add_argument("--tasks", type=str, default=None, help="Comma-separated task types (e.g. mcq,qa,cwe)")
    p_qacc.add_argument("--per-task", type=int, default=None, help="Max samples per task type")
    p_qacc.add_argument("--quant-mode", choices=["gptq", "fp8_blockwise", "noquant"], default="gptq")
    p_qacc.add_argument("--model-path", type=str, default=None)

    p_speed = sub.add_parser("speed", help="Run speed benchmark")
    p_speed.add_argument("--variant", choices=["s1", "s8", "smax", "all"], default="s1")

    p_logs = sub.add_parser("server-logs", help="Show server logs")
    p_logs.add_argument("--lines", type=int, default=100)

    sub.add_parser("shutdown", help="Shut down the fcloud instance")

    p_setup = sub.add_parser("setup", help="Bootstrap a clean fcloud instance")
    p_setup.add_argument("--force", action="store_true", help="Re-setup even if paths exist")

    args = parser.parse_args()
    base_url, token = fcloud_exec.load_config()

    if args.action == "full":
        workflow_full(base_url, token,
                      quant_mode=args.quant_mode,
                      model_path=args.model_path)
    elif args.action == "sync":
        step_sync(base_url, token)
    elif args.action == "restart-server":
        step_restart_server(base_url, token,
                            quant_mode=args.quant_mode,
                            model_path=args.model_path)
    elif args.action == "wait-server":
        step_wait_server(base_url, token)
    elif args.action == "accuracy":
        step_accuracy(base_url, token,
                      quant_mode=args.quant_mode,
                      model_path=args.model_path)
    elif args.action == "quick-accuracy":
        step_quick_accuracy(base_url, token,
                            task_filter=args.tasks,
                            num_per_task=args.per_task,
                            quant_mode=args.quant_mode,
                            model_path=args.model_path)
    elif args.action == "speed":
        step_speed(base_url, token, args.variant)
    elif args.action == "server-logs":
        step_server_logs(base_url, token, args.lines)
    elif args.action == "shutdown":
        print_section("SHUTDOWN")
        fcloud_exec.shutdown_server(base_url, token)
        print("[shutdown] fcloud instance shutdown initiated")
    elif args.action == "setup":
        step_setup(base_url, token, skip_existing=not args.force)


if __name__ == "__main__":
    main()
