#!/usr/bin/env python3
"""
Standalone benchmark: Fused W4A8 REAL vs Marlin on real GPTQ weights.

Loads one layer's weights from the model, repacks for Marlin, and
benchmarks both kernels on large-M shapes.
"""
import json, os, sys, time
import torch

M_VALS = [4096, 8192, 16384, 65536]
GROUP_SIZE = 128

def load_weights(model_path, layer_idx=0, which="gate_up"):
    """Load one layer's GPTQ weights from safetensors."""
    from safetensors import safe_open
    import glob

    if which == "gate_up":
        prefixes = [f"model.layers.{layer_idx}.mlp.gate_proj",
                    f"model.layers.{layer_idx}.mlp.up_proj"]
    else:
        prefixes = [f"model.layers.{layer_idx}.{which}"]

    st_files = sorted(glob.glob(os.path.join(model_path, "*.safetensors")))
    raw = {}
    for st_file in st_files:
        with safe_open(st_file, framework="pt", device="cpu") as f:
            for key in f.keys():
                for p in prefixes:
                    if key.startswith(p):
                        raw[key] = f.get_tensor(key)

    if not raw:
        raise KeyError(f"No tensors with prefixes {prefixes}")

    if which == "gate_up":
        w = {
            "qweight": torch.cat([raw[f"{p}.qweight"] for p in prefixes], dim=1).cuda(),
            "scales":  torch.cat([raw[f"{p}.scales"]  for p in prefixes], dim=-1).cuda().to(torch.bfloat16),
            "qzeros":  torch.cat([raw[f"{p}.qzeros"]  for p in prefixes], dim=-1).cuda(),
            "g_idx":   raw[f"{prefixes[0]}.g_idx"].cuda().to(torch.int32),
        }
    else:
        p = prefixes[0]
        w = {
            "qweight": raw[f"{p}.qweight"].cuda(),
            "scales":  raw[f"{p}.scales"].cuda().to(torch.bfloat16),
            "qzeros":  raw[f"{p}.qzeros"].cuda(),
            "g_idx":   raw[f"{p}.g_idx"].cuda().to(torch.int32),
        }

    N = w["scales"].shape[-1]
    K = w["qweight"].shape[0] * 8
    print(f"  N={N}, K={K}, group_size={GROUP_SIZE}")
    return w, N, K


def fused_w4a8_gemm(x, w, N, K, gs=128):
    qweight, qzeros, scales = w["qweight"], w["qzeros"], w["scales"]
    M = x.size(0)
    Mp = ((M + 127) // 128) * 128
    xp = torch.nn.functional.pad(x, (0,0,0,Mp-M)) if Mp != M else x
    xf8 = xp.to(torch.float8_e4m3fn).contiguous()
    r = torch.ops.w4a8_fused.w4a8_fp8_fused_gemm(qweight, qzeros, scales, xf8, N, K, gs)
    return r[:M].to(x.dtype)


def marlin_gemm(x, w, N, K, gs=128):
    from sgl_kernel import gptq_marlin_gemm
    from sgl_kernel.scalar_type import ScalarType
    ws = torch.zeros(16384, dtype=torch.int32, device="cuda")
    gsi = torch.argsort(w["g_idx"]).to(torch.int32)
    output = gptq_marlin_gemm(
        x, None, w["qweight_marlin"], w["scales"], None,
        w["qzeros"], w["g_idx"], gsi, ws,
        ScalarType.uint(4, False),
        size_m=x.size(0), size_n=N, size_k=K,
        is_k_full=True, use_atomic_add=False, use_fp32_reduce=True, is_zp_float=False,
    )
    return output


def bench(fn, x, w, N, K, gs, warmup=5, iters=50):
    for _ in range(warmup):
        fn(x, w, N, K, gs)
    torch.cuda.synchronize()
    t = []
    for _ in range(iters):
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        fn(x, w, N, K, gs)
        torch.cuda.synchronize()
        t.append(time.perf_counter() - t0)
    t.sort()
    med = t[len(t)//2]
    M = x.size(0)
    flops = 2 * M * N * K
    return med * 1000, flops / med / 1e12


def main():
    model_path = sys.argv[1] if len(sys.argv) > 1 else "/root/models/openbmb/MiniCPM-SALA-90-qa-cwe-mcq-sparse_qkv_w8"
    layer_idx = int(sys.argv[2]) if len(sys.argv) > 2 else 0
    which = sys.argv[3] if len(sys.argv) > 3 else "gate_up"

    print(f"Loading layer {layer_idx} {which} from {model_path}")
    w, N, K = load_weights(model_path, layer_idx, which)

    # Load fused .so
    so_path = os.environ.get("SOAR_W4A8_FUSED_SO", "/root/submission_sim/libw4a8_fused_gemm.so")
    if os.path.exists(so_path):
        torch.ops.load_library(so_path)
        print(f"  loaded fused .so")
    else:
        print("  WARNING: fused .so not found, fused benchmark will fail")

    # Repack weights for Marlin
    from sgl_kernel import gptq_marlin_repack
    g_idx_sort_indices = torch.argsort(w["g_idx"]).to(torch.int32)
    w["qweight_marlin"] = gptq_marlin_repack(
        w["qweight"].contiguous(),
        g_idx_sort_indices,
        K,
        N,
        4,  # num_bits
    )
    print(f"  repacked for Marlin: {w['qweight_marlin'].shape}")

    # --- Correctness check vs Marlin (FP8 acts won't bit-match; use cosine + rel err) ---
    print("\n[correctness] fused_w4a8 vs marlin @ M=256")
    xc = torch.randn(256, K, dtype=torch.bfloat16, device="cuda")
    ref = marlin_gemm(xc, w, N, K, GROUP_SIZE).float()
    got = fused_w4a8_gemm(xc, w, N, K, GROUP_SIZE).float()
    cos = torch.nn.functional.cosine_similarity(got.flatten(), ref.flatten(), dim=0).item()
    denom = ref.abs().mean().clamp_min(1e-6)
    rel = ((got - ref).abs().mean() / denom).item()
    nan = bool(torch.isnan(got).any() or torch.isinf(got).any())
    print(f"  cosine={cos:.5f}  mean_rel_err={rel:.4f}  nan/inf={nan}")
    ok = (cos > 0.99) and (rel < 0.10) and (not nan)
    print(f"  correctness: {'PASS' if ok else 'FAIL'} (expect cos>0.99, rel<0.10 for FP8-act path)")

    print(f"\n{'Kernel':<16} {'M':>8} {'Time(ms)':>10} {'TFLOPS':>8} {'Speedup':>8}")
    print("-" * 52)

    results = []
    for M in M_VALS:
        x = torch.randn(M, K, dtype=torch.bfloat16, device="cuda")

        # Fused W4A8
        t_fus, flops_fus = bench(fused_w4a8_gemm, x, w, N, K, GROUP_SIZE)
        results.append(("fused_w4a8", M, t_fus, flops_fus))

        # Marlin
        t_mar, flops_mar = bench(marlin_gemm, x, w, N, K, GROUP_SIZE)
        results.append(("marlin", M, t_mar, flops_mar))

        speedup = t_mar / t_fus
        print(f"{'fused_w4a8':<16} {M:>8} {t_fus:>8.2f}  {flops_fus:>6.1f}  {speedup:>7.2f}x")
        print(f"{'marlin':<16} {M:>8} {t_mar:>8.2f}  {flops_mar:>6.1f}  {'':>8}")

    out = {"layer": f"layer.{layer_idx}.{which}", "N": N, "K": K, "group_size": GROUP_SIZE, "results": results}
    with open("/tmp/bench_w4a8_vs_marlin_result.json", "w") as f:
        json.dump(out, f, indent=2)
    print(f"\nResults saved to /tmp/bench_w4a8_vs_marlin_result.json")


if __name__ == "__main__":
    main()
