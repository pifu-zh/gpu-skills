---
name: gpu-programming
description: "Use when 用户学GPU编程/CUDA/PTX/SASS/写核函数、或要求解 SASS 汇编/分析 kernel 瓶颈。NVIDIA GPU编程课程体系，从CUDA基础到PTX/SASS全链路；性能分析实操与 gpu-perf skill 联动。"
---

# GPU Programming — CUDA → PTX → SASS 课程体系

## 课程定位

从零基础到能理解PTX和SASS的全链路GPU编程训练。
不重理论推导，重代码和工程实践。

## 课程大纲（4阶段16课）

### 第一阶段：CUDA基础（第1-4课）
**目标：** 写出第一个正确的CUDA程序，理解GPU执行模型。
- 第1课：GPU vs CPU思维转换、SIMT模型、第一个kernel
- 第2课：内存模型、全局内存、cudaMemcpy
- 第3课：共享内存、同步、bank conflict
- 第4课：warp执行、divergence、occupancy
**验收：** 能在Nsight Compute中看懂kernel瓶颈

### 第二阶段：进阶CUDA（第5-8课）
**目标：** 写出高性能算子，善用GPU硬件特性。
- 第5课：stream与并发
- 第6课：常量内存与纹理内存
- 第7课：原子操作与互斥
- 第8课：动态并行、CUDA Graph
**验收：** 写出性能接近cuBLAS的矩阵乘法

### 第三阶段：PTX编程（第9-12课）
**目标：** 理解CUDA C到PTX的编译过程，能在需要时操控PTX指令。
- 第9课：从CUDA到PTX、cuobjdump、inline asm
- 第10课：PTX寄存器与内存访问指令
- 第11课：PTX算术指令、WMMA/WGMMA/TensorCore
- 第12课：PTX同步与通信原语（shfl, bar, match, redux）
**验收：** 能用PTX在关键路径上替换编译器生成的代码

### 第四阶段：SASS与性能分析（第13-16课）
**目标：** 读懂GPU实际执行的指令，精确分析瓶颈。
- 第13课：从PTX到SASS、ILP
- 第14课：指令吞吐与延迟隐藏
- 第15课：汇编优化实战、寄存器压力
- 第16课：算子融合与kernel合并
**验收：** 用denvdis分析任意kernel的SASS，指出瓶颈和优化方案

## 配套资源（2026-09-10 实测：全部在阿里云，本地未迁移）

| 资源 | 位置（阿里云 root@120.55.67.84） | 用途 |
|------|------|------|
| CUDA Programming Guide 中文版 | /root/workspace/learn/CUDA-Programming-Guide-in-Chinese/ | 理论基础 |
| how-to-optim-algorithm-in-cuda | /root/workspace/learn/how-to-optim-algorithm-in-cuda/ | 实战案例 |
| ptx-isa-markdown | /root/workspace/ptx-isa-markdown/ | PTX指令查表 |
| denvdis（SASS 反汇编工具，注意拼写不是 denvvis） | /root/workspace/denvdis/ | SASS反汇编 |
| 课程大纲文件 | /root/workspace/learn/gpu-programming-curriculum.md | 详细课程 |

需要本地使用时：`rsync -avz root@120.55.67.84:/root/workspace/learn ~/workspace/learn`（或按需单拉）。

## 教材与阶段对应

| 阶段 | 对应教材章节 |
|------|-------------|
| 第一阶段 | CUDA Guide第1-3章（基础概念、编程模型、接口）|
| 第二阶段 | CUDA Guide第3章后半 + 第5章（stream、性能）|
| 第三阶段 | CUDA Guide附录B（C++扩展、inline PTX基础）|
| 第四阶段 | CUDA Guide第4章 + 附录K（硬件实现、计算能力）|

## 推荐节奏

全职（每天4-6小时）：8-10周完成全部四阶段
兼职（每天1-2小时）：4-6个月

每个阶段的验收标准是"能写出演示代码"而非"记住概念"。
所有课程内容都应该能编译、profiling、优化、再profiling。

## 使用说明

当用户问GPU/CUDA/PTX相关问题时：
1. 首先确定在哪个阶段范围（基础/进阶/PTX/SASS）
2. 提供代码示例 + 预期输出 + profiling方法
3. 引用配套资源中的具体文件路径
4. 用ptx-isa-markdown查PTX指令，用denvdis分析SASS

## 与 skill 生态联动

- **第四阶段的 profiling 实操 → gpu-perf**：本机是 RDNA2（无 N 卡），Nsight Compute 不可用——性能分析用 gpu-perf 的流程与命令落地（rocprofv3 采集、DVFS 时钟还原、bound 判据表）；SASS 分析方法论平移到 GCN/RDNA ISA（对照 `gpu-perf/platforms/rocm/gfx1030.md` 的 ISA 判读表），NVIDIA 概念 ↔ RDNA 术语映射查 `gpu-perf/platforms/cuda.md`
- **性能优化 → gpu_lib_optimize**：第二阶段"接近 cuBLAS 的矩阵乘"在 RDNA2 侧的对照基线见 gpu-perf L4（rocblas sgemm 20.8 TFLOPS）
- **PTX/SASS 知识的实战出口 → gpu_lib_port**：跨架构移植（如 DeepGEMM/FlyDSL 适配）中读懂原 kernel 的指令意图是改写回退实现的前提
