# Rust GPU 生态专用模块

**状态：框架占位，未实测**。Rust 没有单一 GPU 栈，按运行时分三路；profiling 落在各路底层对应的平台工具。首次在真实环境使用时逐环节实测替换（与 rocm.md 同标准）。

## 三条路线与 profiling 归属

| 路线 | 库 | 底层 | profiling 用 |
|---|---|---|---|
| 原生 CUDA kernel | rust-cuda / cudarc | CUDA driver/runtime | `platforms/cuda.md` 全套（nsys/ncu） |
| 跨平台计算着色 | wgpu (wgpu-hal) | Vulkan/DX12/Metal | Vulkan 层：RenderDoc（帧/_dispatch 级）、平台计数器（AMD：RGP/RGA 未验证；NVIDIA：NSight Graphics） |
| HIP 绑定 | hip-rs 等 | HIP | `platforms/rocm.md` 全套 |

关键认知：**wgpu 路线的 kernel 时间线在图形 API 口径下**（dispatch 而非 kernel launch），rocprofv3/nsys 直接采不到，需要图形侧工具或 Vulkan 层 trace——这是它与传统计算栈 profiling 最大的差异点。

## 六阶段框架（待填充）

| 阶段 | 该做的 | 状态 |
|---|---|---|
| Phase 0 基线 | 同底层平台探针；wgpu 侧注意适配器后端（Vulkan/HIP vs CUDA）决定基线来源 | 未验证 |
| Phase 1 环境 | rustc/crate 版本；wgpu `Adapter::request_device` 报告的 limits vs 卡实际能力；rust-cuda 的 sm/gfx 目标与卡一致 | 未验证 |
| Phase 2 测量 | rust-cuda/cudarc → 对应平台模块；wgpu → RenderDoc/平台图形 profiler；Rust 侧计时（std::time + 同步点）只当粗粒度对照 | 未验证 |
| Phase 3 判定 | 同通用流程 | 通用 |
| Phase 4 深挖 | rust-cuda 生成的 PTX 可 `--print-asm`/nvdisasm 检查；wgpu → naga 生成代码质量是额外一层（Naga IR → 后端 ISA）——比传统栈多一层编译器嫌疑 | 未验证 |
| Phase 5 循环 | 同通用流程 | 通用 |

## 已知设计注意（来自方法论，非实测）

- rust-cuda 的 kernel 是 Rust 宏展开后再走 NVVM/LLVM——达标率异常时检查展开产物（`cargo expand`）是否有意外装箱/边界检查代码混入 kernel（Phase 4 第 1/2 层）
- cudarc 的 launch 开销（driver API FFI）比 C++ 裸调用高，短 kernel 密集提交时先量 API 开销占比（Phase 2 的 API 时间线）
