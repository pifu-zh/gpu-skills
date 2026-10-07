---
name: gpu_tech_extract
description: 从 GPU 库研究中提取可移植技术与思路，强制对比 NVIDIA 生态等价物。Use when 用户说"提取可用技术""对比 NVIDIA/CUDA 的做法""这个库有什么值得学的""沉淀技术思路"——是研究/编译/测试/性能各阶段之后的沉淀阶段。
---

# GPU 技术提取（gpu_tech_extract）

核心命题：研究的最终产出不是"了解了这个库"，而是**可复用的技术、可借鉴的思路、与 NVIDIA 生态的映射**。输入是研究档案（gpu_lib_research 产出），输出是 `~/workspace/docs/research/<库名>/tech-extract.md`。

## 提取三维度（每条提取物归入其一）

1. **机制**（怎么实现的）：值得记住的设计——如 UCX 的 UCM 钩子只拦 2 个 HSA 函数就覆盖全部 hipMalloc（拦截底层而非上层 API）
2. **技术**（可移植的 trick）：能搬到自己项目里的具体做法——如 "小消息 CPU memcpy"（HSA 一致地址空间使 GPU 显存 host 可见）、IPC handle 三级缓存（pgtable + memcmp 命中）
3. **思路**（设计哲学）：指导未来决策的判断——如 "copy 与 ipc 互补而非竞争（能力集+可达性判据天然分流）"、"优雅回退优于硬注册"（aiter dtypes.py 教训）

每条格式：**一句话描述 → 源码/文档证据（文件:行号）→ 可移植性判断（对谁的什么场景有用）→ NVIDIA 生态对照**。

## NVIDIA 对照（强制环节）

为每个重要机制/技术找 NVIDIA/CUDA 生态的等价物或差异点，建映射表：

| 本库（ROCm 侧） | NVIDIA 生态对照 | 差异要点 |
|---|---|---|
| 例：uct/rocm_ipc（HSA IPC attach） | ucx 同仓 `src/uct/cuda/` 的 cuda_ipc | 机制同构（GPU IPC 句柄），细节差异在此 |
| 例：CK（Composable Kernel） | CUTLASS | 模板 GEMM 生成 vs 张量模板库 |
| 例：aiter FlyDSL | Triton（原版 NVIDIA 系） | 嵌入式 DSL 的两种宿主形态 |

常用对照锚点：hipMemcpy↔cudaMemcpy、HSA signal↔cudaEvent/steam callback、rocminfo↔nvidia-smi/deviceQuery、hipcc/clang↔nvcc、MFMA(CDNA)↔Tensor Core、WMMA↔wmma API、ROCm-SMI↔nvidia-smi。库内自带对照的优先（**同一仓库的 cuda/ 目录就是官方对照实现**——ucx 实例，逐文件对读最可靠）。

## 提取来源（按优先级）

1. 研究档案中的"发现"节与实测数据（已有证据，直接提炼）
2. 库内 ROCm↔CUDA 对等实现的源码对读（同仓库最优）
3. 性能实测数字 vs NVIDIA 侧公开数字/本机基线（gpu-perf 的 L4 baseline）
4. 移植过程中的经验（什么能移植什么不能、为什么）

## 产出验收

- [ ] 三维度各有条目，每条带证据与可移植性判断
- [ ] NVIDIA 映射表 ≥ 覆盖库的核心机制
- [ ] 明确标注"对本机/本项目有用"的短清单（与 L4 基线、正在做的项目关联）
- [ ] 可反哺 skill 的条目已回写对应 skill（gpu-perf 坑表、gpu_lib_* 流程）
