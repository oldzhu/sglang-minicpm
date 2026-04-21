# PROPOSAL: Option B — FP8 Blockwise GEMM via Existing SM120 Kernel

**Date**: 2026-04-21  
**Status**: AWAITING APPROVAL  
**Priority**: High — direct path to 30-50% speedup using existing SM120 kernel  
**Baseline**: S1=121.71s, S8=44.09s, Smax=35.86s (Test 12)  
**Expected result**: S1 ~85-100s, S8 ~30-35s, Smax ~25-30s (estimated)

---

## Executive Summary

sgl-kernel already has a fully functional SM120 FP8 blockwise GEMM (`fp8_blockwise_scaled_mm` with `sm120_fp8_blockwise_dispatch_shape`). The only missing piece is weight conversion from GPTQ W4 format to FP8 blockwise format, and the model-level dispatch to use the new kernel.

**No CUDA kernel writing required.** Implementation is pure Python + PyTorch.

---

## Problem Statement

Current Marlin GPTQ W4 kernel:
- Uses SM80 `mma.sync.aligned.m16n8k16` instruction 
- Achieves ~100-140 TFLOPS effective (BF16 path)
- SM120 FP8 hardware (593 TFLOPS) is completely idle during GEMM

Target after Option B:
- Use SM120 UMMA via `fp8_blockwise_scaled_mm` → 350-450 TFLOPS FP8
- Keep accuracy > 99% normalized for C=1.0
- No new CUDA dependencies

---

## Rule Compliance

| Constraint | Status |
|------------|--------|
| ≤2GB submission | ✅ No new wheels (kernel already in sgl-kernel) |
| On-site quantization | ✅ FP8 conversion runs in preprocess_model.py (≪5h) |
| Correctness C ≥ 0.96 | ✅ FP8 8-bit vs W4 4-bit → should maintain >99% accuracy |
| No forbidden tricks | ✅ Pure quantization format change |
| Reproducible | ✅ Deterministic conversion |

---

## Architecture Overview

### Current Flow
```
GPTQ W4 weights (int32 packed, group_size=128)
         ↓
Marlin gptq_marlin.cu (SM80 mma.sync)
         ↓
BF16 output
```

### After Option B
```
FP8 E4M3 weights (K×N, col-major) + scales_b (K/128 × N/128)
         ↓
Activation quantize: BF16 → FP8 E4M3 + scales_a (M × K/128)
         ↓
fp8_blockwise_scaled_mm (SM120 UMMA tcgen05.mma.ws.sync)
         ↓
BF16 output
```

### Weight Conversion (done once in preprocess_model.py)
```
GPTQ W4 model (group_size=128)
         ↓
Per layer: Marlin dequant W4 → FP16 block (shape K×N)
         ↓
Per 128×128 block: find max_abs → FP8_scale = max_abs / 448.0
         ↓
FP8_weights = clamp(FP16_block / FP8_scale, -448, 448).to(float8_e4m3fn)
         ↓
Save FP8 weights (K×N, col-major) + scales (K/128 × N/128) as safetensors
         ↓
Updated config.json with quantization_config: {"quant_type": "fp8_blockwise", "block_size": 128}
```

---

## Exact Files to Change

### 1. `preprocess_model.py` — Add FP8 blockwise conversion step

Add new mode `fp8_blockwise` (or extend `gptq` mode with `--fp8-postprocess` flag).

**After GPTQ quantization completes**, run:
```python
def run_fp8_blockwise_postprocess(gptq_model_dir: Path, dst_dir: Path):
    """Convert GPTQ W4 model to FP8 blockwise for SM120 UMMA."""
    from safetensors.torch import load_file, save_file
    import torch
    
    config = load_json(gptq_model_dir / "config.json")
    quantize_config = load_json(gptq_model_dir / "quantize_config.json")
    group_size = quantize_config["group_size"]  # 128
    
    for shard_file in sorted(gptq_model_dir.glob("*.safetensors")):
        tensors = load_file(shard_file)
        new_tensors = {}
        
        for key, tensor in tensors.items():
            if key.endswith(".qweight"):  # GPTQ packed W4
                layer_prefix = key[:-len(".qweight")]
                scales = tensors[f"{layer_prefix}.scales"]   # (K/128, N) float16
                zeros = tensors.get(f"{layer_prefix}.qzeros")  # packed int32
                
                # Dequantize W4 → FP16
                w_fp16 = gptq_dequantize(tensor, scales, zeros, group_size)  # (K, N) fp16
                
                # Requantize to FP8 blockwise (128×128)
                w_fp8, scale_b = quantize_fp8_blockwise(w_fp16, block_size=128)
                # w_fp8: (K, N) float8_e4m3fn, col-major
                # scale_b: (K/128, N/128) float32
                
                new_tensors[f"{layer_prefix}.weight_fp8"] = w_fp8
                new_tensors[f"{layer_prefix}.weight_fp8_scale"] = scale_b
                # Skip original qweight/scales/qzeros/g_idx
            elif not any(key.endswith(s) for s in [".scales", ".qzeros", ".g_idx"]):
                new_tensors[key] = tensor  # pass through non-GPTQ tensors
        
        save_file(new_tensors, dst_dir / shard_file.name)
    
    # Update config.json
    config["quantization_config"] = {
        "quant_type": "fp8_blockwise",
        "block_size": 128,
        "fp8_dtype": "float8_e4m3fn"
    }
    save_json(config, dst_dir / "config.json")
```

**Helper: `quantize_fp8_blockwise(w_fp16, block_size=128)`**:
```python
def quantize_fp8_blockwise(w: torch.Tensor, block_size: int = 128):
    """w: (K, N) float16/bfloat16 → (K, N) float8_e4m3fn col-major + (K/128, N/128) float32 scales"""
    K, N = w.shape
    assert K % block_size == 0 and N % block_size == 0
    
    # Reshape to (K/128, 128, N/128, 128) for blockwise max
    w_blocks = w.reshape(K // block_size, block_size, N // block_size, block_size)
    max_abs = w_blocks.abs().amax(dim=(1, 3))  # (K/128, N/128)
    
    # FP8 E4M3 max value = 448.0
    FP8_MAX = 448.0
    scales = (max_abs / FP8_MAX).clamp(min=1e-12)  # (K/128, N/128) float32
    
    # Quantize
    scale_expanded = scales.unsqueeze(1).unsqueeze(3).expand_as(w_blocks)  # (K/128, 128, N/128, 128)
    w_scaled = (w_blocks / scale_expanded).clamp(-FP8_MAX, FP8_MAX)
    w_fp8 = w_scaled.reshape(K, N).to(torch.float8_e4m3fn)
    
    # Convert to col-major for kernel (transpose → contiguous)
    w_fp8_col = w_fp8.t().contiguous()  # (N, K) but viewed as col-major (K, N)
    # Actually kernel expects B in col-major = (N, K) contiguous with stride(0)=1
    
    return w_fp8_col, scales
```

**Scale format alignment check**:
- Kernel checks: `mat_b.size(0) / 128 == scales_b.size(0)` and `mat_b.size(1) / 128 == scales_b.size(1)`
- `mat_b` is `(N, K)` col-major (stored as N rows, K cols) with `stride(0) = 1`
- So `scales_b` must be `(N/128, K/128)`
- Note: the weight matrix is `(K_in, N_out)` logically, but stored as `(N_out, K_in)` col-major
- Thus `scales_b` shape = `(N_out/128, K_in/128)`

### 2. New file: `python/sglang/srt/layers/quantization/fp8_blockwise.py`

New quantization method class:
```python
class FP8BlockwiseLinearMethod(LinearMethodBase):
    """FP8 blockwise linear using SM120 UMMA (fp8_blockwise_scaled_mm)."""
    
    def create_weights(self, layer, ...):
        # Register weight_fp8 and weight_fp8_scale as layer parameters
        layer.weight_fp8 = nn.Parameter(...)      # (N, K) float8_e4m3fn
        layer.weight_fp8_scale = nn.Parameter(...)  # (N/128, K/128) float32
    
    def apply(self, layer, x: torch.Tensor, bias=None) -> torch.Tensor:
        # Quantize activation per-row per-128-K-block
        scales_a, x_fp8 = quantize_activation_fp8_blockwise(x)  # (M, K/128), (M, K) fp8
        
        # Call SM120 FP8 GEMM
        from sgl_kernel import fp8_blockwise_scaled_mm
        out = fp8_blockwise_scaled_mm(
            x_fp8,                    # (M, K) fp8, row-major
            layer.weight_fp8,         # (N, K) fp8, col-major
            scales_a,                 # (M, K/128) float32
            layer.weight_fp8_scale,   # (N/128, K/128) float32
            out_dtype=torch.bfloat16
        )
        if bias is not None:
            out = out + bias
        return out
```

**Activation quantization per-row-per-block**:
```python
def quantize_activation_fp8_blockwise(x: torch.Tensor):
    """x: (M, K) bfloat16 → x_fp8: (M, K) float8_e4m3fn + scales_a: (M, K/128) float32"""
    M, K = x.shape
    assert K % 128 == 0
    FP8_MAX = 448.0
    
    x_blocks = x.reshape(M, K // 128, 128)  # (M, K/128, 128)
    max_abs = x_blocks.abs().amax(dim=2)  # (M, K/128)
    scales_a = (max_abs / FP8_MAX).clamp(min=1e-12)  # (M, K/128)
    
    scale_expanded = scales_a.unsqueeze(2)  # (M, K/128, 1)
    x_scaled = (x_blocks / scale_expanded).clamp(-FP8_MAX, FP8_MAX)
    x_fp8 = x_scaled.reshape(M, K).to(torch.float8_e4m3fn)
    
    return scales_a, x_fp8
```

### 3. `python/sglang/srt/layers/quantization/__init__.py` — Register new method

Add `FP8BlockwiseLinearMethod` to the quantization registry.

### 4. `python/sglang/srt/models/minicpm.py` — Conditional FP8 dispatch

When `quantization_config.quant_type == "fp8_blockwise"` is detected:
- Pass `FP8BlockwiseConfig` as the quant config to linear layers
- All linear layers (MLP gate/up/down, QKV, O-proj) use `FP8BlockwiseLinearMethod`
- Lightning layers (`SimpleGLAAttnBackend`) don't use linear layers → no change needed

### 5. `benchmark/soar/demo_sala/prepare_env.sh` — No change needed

The model format change is handled by `preprocess_model.py` and detected at load time via `config.json`. No server args needed.

### 6. `benchmark/soar/demo_sala/preprocess_model.py` — Mode extension

Add mode `fp8_blockwise` or `gptq+fp8` to choices:
```python
choices=["copy", "gptq", "fp8_blockwise"],
```

And new execution path:
```python
elif mode == "fp8_blockwise":
    # Option 1: directly from FP16 model (no GPTQ calibration needed)
    run_fp8_blockwise_quantization(src, dst)
    # Option 2: from already-quantized GPTQ model (convert format)
    # run_fp8_blockwise_postprocess(src, dst)
```

**Decision point**: Start directly from FP16 model (faster, no calibration, better accuracy) vs. convert from existing GPTQ model. **Recommendation**: Start from FP16 for simplicity and accuracy.

---

## Implementation Sequence

### Day 1: Weight conversion script

1. Write `run_fp8_blockwise_quantization(src, dst)` in `preprocess_model.py`
2. Use `transformers.AutoModel.from_pretrained(src, torch_dtype=torch.float16)` to load FP16
3. Iterate all linear layers, call `quantize_fp8_blockwise(layer.weight, block_size=128)`
4. Save using `safetensors.torch.save_file` with custom naming convention
5. Write `quantize_fp8_blockwise()` and test on a single layer locally

**Test**: `python preprocess_model.py --input /root/models/openbmb/MiniCPM-SALA-Copy --output /tmp/minicpm_fp8 --mode fp8_blockwise`
- Should complete in <10 minutes (no calibration, just compute)
- Check saved file sizes: should be ~2× the FP16 model? No — FP8 is 1 byte/param vs FP16 2 bytes/param → same size as GPTQ W8, 2× smaller than FP16, larger than GPTQ W4

Wait: FP8 = 1 byte/param, FP16 = 2 bytes/param, GPTQ W4 = 0.5 bytes/param
- FP8 model would be ~2× larger than GPTQ W4 but 2× smaller than FP16
- For 9B model: FP8 ≈ 9GB, W4 GPTQ ≈ 4.5GB
- For submission (≤2GB): this is weights stored in the tarball → wait, the submission includes wheels and scripts but NOT model weights. Models are already on fcloud. Check submission format!

Actually per the submission rules: model weights are NOT in the 2GB tarball. Only code wheels + scripts. So FP8 weight size (9GB vs 4.5GB W4) is not a submission size concern. It only affects GPU memory during inference (fits in 84GB) and quantization time (should be <30 min vs GPTQ which takes hours).

### Day 2: Quantization method class

1. Create `python/sglang/srt/layers/quantization/fp8_blockwise.py`
2. Implement `FP8BlockwiseConfig` and `FP8BlockwiseLinearMethod`
3. Register in `__init__.py`
4. Local unit test: single linear layer forward pass

### Day 3: Model integration

1. Modify `minicpm.py` to detect `fp8_blockwise` quantization_config
2. Pass correct quant config to linear layers
3. Handle weight loading (new parameter names: `weight_fp8`, `weight_fp8_scale`)
4. Launch server locally with FP8 model: `python -m sglang.launch_server --model-path /tmp/minicpm_fp8`
5. Verify server starts without errors

### Day 4: Accuracy validation

1. Run `eval_model_001.py` on fcloud with FP8 model
2. Check normalized accuracy: target >99% (C=1.0)
3. If accuracy <99%: investigate which layers need W4 (e.g., keep GPTQ for QKV)

### Day 5: Speed benchmark

1. Run `fcloud_workflow.py speed --variant all`
2. Compare S1/S8/Smax vs baseline (Test 12)
3. Profile with Nsight: check actual FP8 TFLOPS utilization

---

## Risk Assessment

| Risk | Probability | Impact | Mitigation |
|------|-------------|--------|------------|
| Accuracy < 99% with pure FP8 | Medium | High | Fallback: keep GPTQ for QKV, FP8 only for MLP |
| Weight conversion bugs (wrong block layout) | Low | High | Unit test: dequantize FP8 back to FP16, compare to original |
| Scale format mismatch with kernel | Medium | High | Trace TORCH_CHECK messages in kernel; test with tiny matrix first |
| Decode regression (2× weight BW) | Medium | Medium | Profile S1 separately; if decode dominant, fallback or limit FP8 to MLP only |
| FP8 decode slower than W4 decode | High | Low-Medium | Expected; S1 score depends on prefill/decode mix in benchmark |

---

## Rollback Plan

FP8 is selected at model load via `config.json` quantization_config. To rollback:
1. Use original GPTQ model directory (still preserved)
2. No code rollback needed — kernel dispatch checks quantization_config type
3. Server restart with original model path

---

## Validation Commands

```bash
# On fcloud:

# Step 1: Convert FP16 model to FP8 blockwise
python3 /root/submission_sim/preprocess_model.py \
    --input /root/models/openbmb/MiniCPM-SALA-Copy \
    --output /root/models/minicpm_fp8_blockwise \
    --mode fp8_blockwise

# Step 2: Check weights
python3 -c "
from safetensors.torch import load_file
t = load_file('/root/models/minicpm_fp8_blockwise/model-00001-of-XXXX.safetensors')
for k,v in list(t.items())[:10]:
    print(k, v.shape, v.dtype)
"

# Step 3: Start server with FP8 model
source /root/submission_sim/prepare_env.sh
python3 -m sglang.launch_server \
    --model-path /root/models/minicpm_fp8_blockwise \
    --host $HOST --port $PORT "${SGLANG_SERVER_ARGS[@]}"

# Step 4: Quick accuracy check
python3 /root/data/eval_model_001.py \
    --model_url http://localhost:30000 \
    --data_path /root/data/perf_public_set.jsonl \
    --max_samples 100  # quick sanity check

# Step 5: Full accuracy
python3 scripts/fcloud/fcloud_workflow.py accuracy

# Step 6: Speed
python3 scripts/fcloud/fcloud_workflow.py speed --variant all
```

---

## Expected Results

| Metric | Baseline (Test 12) | After Option B | Target |
|--------|-------------------|----------------|--------|
| S1 duration | 121.71s | ~85-100s | <90s |
| S8 duration | 44.09s | ~30-35s | <32s |
| Smax duration | 35.86s | ~25-30s | <28s |
| Normalized accuracy | 99.11% | >99% | >99% |
| Correctness C | 1.0 | 1.0 | 1.0 |

**Note**: These are estimates based on TFLOPS utilization improvement. Actual decode impact from 2× weight memory may partially offset prefill gains, especially for S1 (single request, likely decode-heavy).

---

## Next Steps After B

If Option B succeeds and there's still a gap to top 5, consider:
1. **FP8 KV cache + FP8 attention**: already have FP8 KV, but attention GEMM could also use FP8
2. **Selective precision**: FP8 for MLP (larger matrices → more compute-bound), W4 for QKV (smaller, decode-heavy)
3. **Option C**: W4A8 fused dequant kernel (keeps W4 memory + gains FP8 compute)
4. **Prefill/decode split**: different precision per phase

---

## Reference Implementation Files

For implementation guidance:
- Existing SM120 FP8 kernel: `sgl-kernel/csrc/gemm/fp8_blockwise_gemm_kernel.cu`
- Python binding: `sgl-kernel/python/sgl_kernel/gemm.py` (`fp8_blockwise_scaled_mm`)
- Scale format check (lines 390-415 of fp8_blockwise_gemm_kernel.cu):
  - `scales_a`: shape `(M, K/128)`, stride_0 = 1 (M-major)
  - `scales_b`: shape `(K_dim/128, N_dim/128)` where K_dim = mat_b.size(0) (the K dimension of col-major B)
- Existing FP8 quantization utils: `python/sglang/srt/layers/quantization/fp8_utils.py`
- Existing FP8 linear usage: `python/sglang/srt/layers/quantization/fp8_utils.py` (`fp8_blockwise_scaled_mm`)

**IMPORTANT**: The kernel expects `mat_b` in column-major format: `mat_b.stride(0) == 1`, i.e., N is the fast dimension. This means the weight matrix must be stored as `(K, N)` in Fortran order OR equivalently transposed `(N, K)` in C order. Ensure weight conversion produces the correct layout.
