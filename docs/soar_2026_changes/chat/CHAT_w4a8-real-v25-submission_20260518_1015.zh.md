# 聊天记录: W4A8 REAL v25 提交包准备 — 2026-05-18

## 会话信息
- **日期**: 2026-05-18
- **主题**: w4a8-real-v25-submission
- **提交**: ff8894120, 377e54197, 16d1919d7

## 1. 用户请求

1. "让我们准备一个新的tar.gz用于新的正式提交，然后转向融合优化" — 用户请求创建v25提交包。

## 2. Agent操作

### Wheel重建
- 问题: 之前重建的wheel只包含.so文件，缺少Python `.dist-info`和所有Python源文件 (`__init__.py`等)
- 根因: 旧wheel的`unzip`失败，重新打包只包含了手动复制的.so文件
- 解决方案: 在fcloud上通过`make build`从源码重建wheel (增量编译，warm ccache约5分钟)
- 结果: 完整的550MB wheel，包含所有Python + .so + .dist-info文件

### Kernel操作验证
- 源码构建后验证`torch.ops.sgl_kernel.gptq_int4_to_fp8_blockwise`
- 确认必须在torch.ops调用前`import sgl_kernel` (TORCH_LIBRARY_FRAGMENT静态初始化)

### 环境变量默认值
- 在prepare_env.sh中将`SOAR_W4A8_REAL_FP8_GEMM`默认值从`0`改为`1`
- 更新注释: "Default ON (v25). Set SOAR_W4A8_REAL_FP8_GEMM=0 to disable."

### 打包创建
- 源目录: fcloud上的`/root/submission_sim/`
- 命令: `tar --exclude='__pycache__' --exclude='*.pyc' -czf /root/minicpm_sala_submit_v25.tar.gz sgl_kernel-*.whl prepare_env.sh prepare_model.sh preprocess_model.py perf_public_set.jsonl sglang`
- 大小: 529MB压缩
- 内容验证: 1823个文件，包含wheel、sglang源码、脚本

### 打包下载
- 复制到`/workspace/` (JupyterLab根目录)
- 通过JupyterLab API下载: `GET /api/contents/minicpm_sala_submit_v25.tar.gz`
- 保存至: `benchmark/soar/demo_sala/minicpm_sala_submit_v25.tar.gz` (554MB)

### 文档
- 创建`CHANGE_W4A8_REAL_FP8_GEMM.en.md`和`.zh.md`
- 更新`TEST_RESULTS_TRACKING.md`添加v25结果

### 成本节省
- 通过`pause-instance`暂停fcloud实例 (console API)

## 3. 关键决策

1. **源码构建wheel**: 选择`make build`而非手动zipfile重新打包，确保wheel完整性
2. **环境变量默认开启**: v25默认启用W4A8 REAL，因为精度/速度均优于基线
3. **暂停实例**: 下载完成后暂停fcloud以在文档编写期间节省成本

## 4. 遇到的问题

1. **多行exec语法错误**: fcloud_exec.py的heredoc方式在多行脚本中存在bash语法错误。解决方案: 先将脚本写入文件再执行。
2. **Wheel缺少.dist-info**: 之前手动zipfile重建仅包含.so文件。通过完整源码重建修复。
3. **首次pause尝试返回504**: 网关暂态超时；重试成功。

## 5. 结果总结

| 指标 | Test 12 基线 | v25 W4A8 REAL |
|------|-------------|---------------|
| 精度 | 79.29% | **81.07%** |
| S1 | 121.71s | **110.79s** |
| S8 | 44.09s | **40.51s** |
| Smax | 35.86s | **32.67s** |
| C | 1.0 | **1.0** |

## 6. 交叉引用

- `CHANGE_W4A8_REAL_FP8_GEMM.en.md` — 完整实现文档
- `CHANGE_W4A8_REAL_FP8_GEMM.zh.md` — 中文版本
- `TEST_RESULTS_TRACKING.md` — 已更新v25条目
- `benchmark/soar/demo_sala/prepare_env.sh` — 默认值改为SOAR_W4A8_REAL_FP8_GEMM=1
- `benchmark/soar/demo_sala/minicpm_sala_submit_v25.tar.gz` — 提交包 (529MB)

## 7. 待办事项

1. 上传`minicpm_sala_submit_v25.tar.gz`至SOAR官方提交网站
2. 融合GEMM反量化kernel (CHANGE_W4A8_FUSED): 预计3-4天工作量，进一步提升S1 5-10%
3. v25提交后追踪官方排行榜
