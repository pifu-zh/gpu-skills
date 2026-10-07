---
name: gpu_lib_build
description: GPU 库的编译与架构适配工艺（autotools/cmake/setup.py 三大体系、GPU 开关、目标架构注册修复、编译错误修复循环）。Use when 用户要编译/构建 GPU 相关库、适配新 GPU 架构（如把库移植到 gfx1030）、解决编译错误、确认构建配置——是 gpu_lib_research 之后、gpu_lib_test 之前的流水线阶段。
---

# GPU 库编译与适配（gpu_lib_build）

前提：已按 gpu_lib_research 完成盘查（知道构建系统、GPU 开关、现状）。本阶段产物：**可用的构建产物 + 编译问题清单与解法（写入研究档案）**。

## 第一步：环境确认（错环境全白干）

1. 编译发生地：主机 or 容器？容器内编译要确认挂载路径（如 /home/zhuo→/home/ubuntu）、GPU 设备直通、容器内工具链版本
2. 工具链版本三件套：ROCm 版本（`rocminfo | grep Runtime`）、python 版本（ROCm 附带工具常只支持 3.10-3.13）、编译器（hipcc/amdclang）
3. GPU 空闲检查（`rocm-smi`）——编译本身不用 GPU，但构建系统可能跑探测程序

## 第二步：按构建体系走（三大体系的已验证差异）

**autotools（C 系，如 ucx）**：`./autogen.sh && ./configure`（GPU 开关是 `--with-rocm` 类，探测逻辑在 `configure.m4` 的 `UCX_CHECK_ROCM` 式宏）；推荐 out-of-tree（`mkdir build && cd build && ../configure`）。**坑：libtool wrapper**——build 目录下的"可执行文件"多为 wrapper 脚本，真实 ELF 在同路径 `.libs/` 子目录；直接跑报库找不到时用 `LD_LIBRARY_PATH` 指向各 `.libs`。

**cmake（C++ 系）**：**必须显式 `CMAKE_BUILD_TYPE=Release`**——默认空 = -O0 设备码，性能分析全作废（盲测实证）。GPU 开关常是 `-D GPU_TARGETS=` / `-D CMAKE_HIP_ARCHITECTURES=`，值用 `rocminfo` 查到的架构号。

**setup.py/pip（Python+HIP 系，如 aiter）**：架构相关环境变量先设（如 `GPU_ARCHS=gfx1030`、禁用不支持后端如 `ENABLE_CK=0`）；预编译开关（如 `PREBUILD_KERNELS=0`）关掉可跳过无产物的架构。编译可能在**首次 import 时才发生**（JIT 库）——build 成功 ≠ 可用，必须 import/调用验证。

## 第三步：架构适配（移植类任务的核心）

目标架构不被库认识时的标准修复位（aiter/gfx1030 实战清单，模式可迁移）：

1. **架构注册表**：找库的 架构枚举/GFX_MAP/allowed_archs（通常在 build 工具文件），加目标架构条目
2. **运行时探测表**：get_gfx/芯片信息函数（rocminfo 解析处）同名追加
3. **硬件参数表**：LDS 容量、CU 数、warp size 等按架构查表的——漏了会在运行时 ValueError
4. **dtype/特性回退**：目标架构缺的特性（如 RDNA2 无 fp8）确认**优雅回退**存在——有时不注册反而是对的（aiter dtypes.py 教训：加了键反而语义错误）
5. 方法：搜上游对相邻架构的支持提交（git log 已知架构名），照抄结构

每处修复记录：文件:行号、改动内容、为什么（写进研究档案）。修复后**最小验证**先行（import / 单个算子），再全套。

## 第四步：编译错误修复循环

错误分类 → 对策：
- **依赖缺失**（头文件/库/子模块）：装依赖或启用替代宏（如 CK 缺失时的 `-DAITER_CK_FREE=1` shim 机制——找库自己的 graceful-degradation 开关）
- **架构不支持**（MFMA builtin、fp8 intrinsic、预编译产物缺架构）：关对应后端/走回退路径，**不要硬改 backend 源码**
- **编译器差异**（ROCm 版本间 flag 变化）：查 `-O`/std 版本，别动源码逻辑
- 修复循环纪律：一次一个变量；每次修复记录（症状→根因→解法）；卡住超 3 轮退回研究阶段补盘查

## 高频坑（14 库流水线实测）

1. **hip 包的 GPU_BUILD_TARGETS 缓存烘焙**：`hip-config-amd.cmake` 用 `set(... CACHE STRING ...)` 无 FORCE——首次 configure 未传架构时展开为 "all"（17 架构），之后改传也不生效，每个 TU 编 17 遍设备码（单测文件 15min+）。解法：首配就传 `-DGPU_BUILD_TARGETS:STRING=<架构>`（或删构建树重配）
2. **clients 无条件 `enable_language(Fortran)`** 且全无 Fortran 源——CMake 直接报错；同款守卫补丁（`if(NOT CMAKE_Fortran_COMPILER) return()` 类）在 rocsolver/hipsolver/hipblaslt 三处通用
3. **apt 提取 deb 的 cmake config 系统性 multiarch 前缀错位**（GTest→fmt→CHOLMOD 三例：config 找的是 /usr/lib/<multiarch> 而实际在 /usr/lib）——运行时还需软链回 install/lib
4. **新架构移植第一检查项：上游调参数据是否存在**——"代码能编译 ≠ 有 kernel"（hipblaslt 实例：TensileLite logic 库无 navi21 条目 → 生成的 hsaco 0 字节 kernel → 运行时 INVALID_VALUE）
5. **重型 HIP 模板编译单进程可达数 GB 内存**——多构建并行会 OOM killer（且 make 可能误报 exit 0，日志特征"已杀死"）；串行或低 -j
6. **hipcc `-c` 与 `--genco` 的加载语义差异**：hipModuleLoad(Data) 只吃 `--genco` 产的完整 metadata code object；`-c` 的 relocatable 被拒（"no kernel image/invalid"）——官方 module_api README 明示 --genco；对 IR 文本输入用 `clang --target=amdgcn-amd-amdhsa -c` 等价路径
7. **code object 版本必须与内核驱动匹配**：runtime 新/驱动老时新版本产物可能被拒（invalid）或部分支持（加载执行可但隐式参数失效）——显式 `-mcode-object-version=5` 对齐老驱动
8.5. **AMDGPU target 的 alloca 地址空间契约**：NVPTX 容忍 generic addrspace 的 alloca，AMDGPU module verifier 直接拒绝（本地帧必须 addrspace(5) scratch）——从 NVIDIA IR 形态移植/翻译到 AMDGPU 时，任何带本地帧的 kernel 都会撞；这是 IR 翻译管线的必备改写步骤（cuda-oxide 批量实证）
8. **bindgen 类 crate 的 CUDA 头最小凑齐法**（无 CUDA toolkit 的 ROCm 机器）：从 NVIDIA repo（developer.download.nvidia.com/compute/cuda/repos/ubuntu2404/x86_64/）拆 deb 取头——cuda.h 在 `cuda-cudart-dev`、curand.h 在 `libcurand-dev`、`crt/` 在 `cuda-cccl`；CUDA 13 已把 crt/ 扁平化到 include 根，老式 `#include <crt/...>` 用 symlink 兼容层补。只装头不装库（运行时 dlopen 的 crate 不需要 libcuda）。CUDA 13 拆包后组件名变化（13.x 打包重组实测地图）：**ptxas 在 cuda-nvcc 包**（不在 cuda-nvcc-tools）、**libdevice.10.bc 在 libnvvm-13-2 包**（cuda-nvvm-13-x 包名已消失）、cuda.h 在 cuda-cudart-dev、curand.h 在 libcurand-dev、crt/ 在 cuda-cccl；先 curl repo 目录索引 grep 找真实文件名再下

## 产物验收标准

- [ ] 构建零错误（警告记录）
- [ ] 最小验证通过（import / 链接 / smoke run）
- [ ] 编译问题清单与解法写入研究档案（含环境、版本、所有非默认 flag——复现的最小信息集）
- [ ] 确认优化级别非 -O0
