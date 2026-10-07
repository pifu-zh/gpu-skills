# Mojo GPU 专用模块

**状态：框架占位，未实测**。Mojo 的 GPU 栈演化快，命令以 [Modular 官方文档](https://docs.modular.com/mojo/) 为准。首次在有 Mojo GPU 环境的机器上使用本模块时：跑通一次下列框架的每个环节，把实测命令替换进来并去掉"未验证"标注（与 rocm.md 同等标准）。

## 生态定位

Mojo 通过 `gpu` 标准库包（`gpu.host`、`gpu.id` 等）写 GPU kernel，基于 MLIR 编译到 PTX/AMDGCN；底层运行时可选 CUDA（NVIDIA）或 HIP（AMD）——**Mojo 程序的 profiling 最终落在宿主运行时对应的平台工具上**：

- Mojo on CUDA → 用 `platforms/cuda.md` 的 nsys/ncu 流程
- Mojo on HIP → 用 `platforms/rocm.md` 的 rocprofv3 流程（kernel 名是 Mojo mangled 名，报告脚本已做 demangle 兼容）

## 六阶段框架（待填充）

| 阶段 | 该做的 | 状态 |
|---|---|---|
| Phase 0 基线 | 底层是哪张卡先跑对应平台探针；Mojo 侧带宽微基准可用 `gpu.host` 的 memcpy 或 Benchmark 模块自写 | 未验证 |
| Phase 1 环境 | `mojo --version`；PIXEL/HIP 运行时可见性；目标 arch（`--gpu-arch`?）与实际卡一致 | 未验证 |
| Phase 2 测量 | 宿主侧 profiler 采集（见上）；Mojo Benchmark 模块给 kernel 级计时 | 未验证 |
| Phase 3 判定 | 与通用流程相同；注意 Mojo kernel 名 mangling | 未验证 |
| Phase 4 深挖 | 编译器层证据：MLIR/LLVM IR/生成汇编导出选项待查；ISA 判读回对应平台模块 | 未验证 |
| Phase 5 循环 | 同通用流程 | 通用 |

## 已知设计注意（来自方法论，非实测）

- Mojo 抽象层多（tensor/layout/algorithms），达标率低时先分清是"算法布局层"还是"生成代码层"——对照手写 HIP/CUDA 同算法 kernel 是最有效的基线对比
- Elementwise 算子应有 vectorize+tile 自动策略；未达标先查 layout 是否导致非合并访存（对应 Phase 4 第 1 层）
