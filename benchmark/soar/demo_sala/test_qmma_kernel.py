import torch, os
torch.ops.load_library("/root/submission_sim/libw4a8_fused_gemm.so")

K, N, M, g = 128, 256, 128, 128

# All weights=1 packed as 0x11111111 (8 ones, 4-bit each)
qw = torch.full((K//8, N), 0x11111111, dtype=torch.int32, device="cuda")
# All zeros=0 packed as 0x00000000
qz = torch.zeros((K//g, N//8), dtype=torch.int32, device="cuda")
# Scale = 1/16
sc = torch.full((K//g, N), 1.0/16.0, dtype=torch.bfloat16, device="cuda")
# Input all ones
a_bf16 = torch.ones(M, K, dtype=torch.bfloat16, device="cuda")
a_fp8 = a_bf16.to(torch.float8_e4m3fn).contiguous()

r = torch.ops.w4a8_fused.w4a8_fp8_fused_gemm(qw, qz, sc, a_fp8, N, K, g)
print("Shape:", r.shape)
print("min:", r.min().item(), "max:", r.max().item(), "mean:", r.mean().item())
print("First 5:", r[0,:5].tolist())
