# NVIDIA / CUDA 专用模块

**状态：未实测**（当前环境无 N 卡）。命令以 `nsys --help` / `ncu --help` / 官方文档为准；六阶段方法论照主 SKILL.md 执行，本文件只给工具映射。分析方法论（基线、达标率、四层归因、反汇编证据）完全通用，两边一致。

## Phase 0/1：基线与环境（CUDA 侧）

```bash
nvidia-smi -q                     # 显存/时钟上限/功耗墙/PCIe 代数宽度（理论规格来源）
nvidia-smi dmon -s pucvmet        # 实时利用率/时钟/温度（实测行为）
bandwidthTest                     # CUDA samples 官方带宽基准（或自写 cudaMemcpy 大块 stream）
nvcc --version; nvidia-smi        # 运行时↔驱动版本匹配（CUDA Toolkit 对驱动有最低版本要求）
nvidia-smi -q -d SUPPORTED_CLOCKS # 可锁时钟范围（锁定用 -lgc，需 root）
```

工具-架构支持检查：`ncu --query-metrics` 对目标 chip 报可用指标集（对应 ROCm 的 rocprofv3-avail）。

## Phase 2：测量

| 用途 | 命令 |
|---|---|
| 时间线全景（kernel+API+memcpy+NVTX） | `nsys profile -t cuda,nvtx,osrt -o rep ./app` |
| 热点汇总表（对应 gpu_profile_report.py） | `nsys stats --report cuda_gpu_kern_sum,cuda_gpu_mem_time_sum rep.nsys-rep` |
| 每 kernel 计数器深挖（对应 --pmc） | `ncu --set full -o krep ./app`（Speed-of-Light/Roofline 面板内置） |
| 只看重点指标 | `ncu --metrics sm__throughput.avg.pct_of_peak_sustained_elapsed,dram__bytes.sum ./app` |
| 用户标记 | NVTX（`nvtxRangePush`），`nsys -t nvtx` 采集 |

对应概念：`dram__bytes.sum` ≈ FETCH_SIZE；`sm__throughput...pct_of_peak` ≈ 利用率；`achieved_occupancy` ≈ OccupancyPercent。达标率照通用流程算：实测吞吐 ÷ 实测基线。

## Phase 4：编译器/驱动深挖

- ISA 反汇编（对应 hipcc --save-temps）：`nvcc -arch=sm_XX -ptx`（PTX 中间层）→ `cuobjdump -sass binary`（真实 SASS）；查 LDG.E.128（向量化访存）、FFMA（融合乘加）、局部内存 spill
- 编译 flag：`-arch=sm_XX` / `-O3`；寄存器压制 `__launch_bounds__`、`-maxrregcount`
- 驱动层：时钟锁定 `nvidia-smi -lgc`（解决 DVFS 陷阱，比 ROCm 方便）；持久模式 `nvidia-smi -pm 1`

## 与 ROCm 的概念对照

| 概念 | AMD | NVIDIA |
|---|---|---|
| 波前/线程束 | wavefront（wave64/32） | warp（32） |
| 计算单元 | CU | SM |
| 共享内存 | LDS | shared memory |
| 标量/向量寄存器 | SGPR / VGPR | uniform / 32-bit 寄存器 |
| ISA | GCN/RDMA assembly | SASS（PTX 为中间层） |
| 用户标记 | ROCTx | NVTX |
