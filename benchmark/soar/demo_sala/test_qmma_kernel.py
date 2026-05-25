import torch, os
torch.ops.load_library("/root/submission_sim/libw4a8_fused_gemm.so")

K, N, M, g = 128, 256, 128, 128

# Pack weight=8 into [K/8, N]: each int32 = 0x88888888 (8 eights)
qw = torch.full((K//8, N), 0x88888888, dtype=torch.int32, device="cuda")
# Pack zero=0 into [K/g, N/8]
qz = torch.zeros((K//g, N//8), dtype=torch.int32, device="cuda")
# Scale = 1.0
sc = torch.ones((K//g, N), dtype=torch.bfloat16, device="cuda")
# Input all ones -> FP8
a_bf16 = torch.ones(M, K, dtype=torch.bfloat16, device="cuda")
a_fp8 = a_bf16.to(torch.float8_e4m3fn).contiguous()

r = torch.ops.w4a8_fused.w4a8_fp8_fused_gemm(qw, qz, sc, a_fp8, N, K, g)
print("Shape:", r.shape)
print("min:", r.min().item(), "max:", r.max().item(), "mean:", r.mean().item())
print("First 5:", r[0,:5].tolist())

# Expected: w4=8, z4=0+1=1, scale=1.0 -> effective=7
# Each output = sum(7 * 1.0) over K=128 = 7*128 = 896
expected = 7.0 * 128.0
print("Expected:", expected, "got mean:", r.float().mean().item())
diff = abs(r.float().mean().item() - expected)
print("Diff from expected:", diff)
print("PASS" if diff < 5.0 else "FAIL")
