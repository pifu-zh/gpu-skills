---
name: gpu_lib_optimize
description: GPU 库/算子的性能优化工艺（与 gpu-perf 配对：分析定位→本 skill 实施优化→复测验证的单变量循环；优化手段分层清单）。Use when 性能分析已完成且瓶颈已定位、用户要求优化库/算子性能、或说"把它优化到 90%""提速"——是 gpu-perf（分析）之后的流水线阶段。
---

# GPU 库性能优化（gpu_lib_optimize）

与 gpu-perf 是配对关系：**gpu-perf 负责"哪里慢、为什么慢、离天花板多远"，本 skill 负责"怎么改"，改完必须回 gpu-perf 复测**。没有分析结论不动手——凭直觉的优化是赌博。

## 前提检查

- [ ] gpu-perf 分析完成：热点、bound 判定（memory/compute/latency/overhead）、达标率与天花板
- [ ] 被测程序编译级别正确（非 -O0）、测量含同进程 warmup
- [ ] UT 通过（优化一个功能坏掉的库没有意义）

## 优化循环（每轮固定节拍）

```
选最高优先证据（gpu-perf 报告）→ 预估收益 → 改一个变量 → gpu-perf 同命令复测 → 记录（改动|指标|达标率 前→后）
```

- 收敛判据：连续两轮提升 <2% 或达标率进入该类型天花板 → 停
- 每轮一行记录；负结果也记录（排除的假设比猜测值钱）
- 改动前留基准数字与基准代码（git stash/branch），永远可回退

## 手段清单（按 gpu-perf 四层组织；标注✅的是有本机实测背书的，其余为通用手段待实战回填实例）

**算法与数据布局层（收益通常最大）**
- ✅ 消除 O(n×window) 型计算膨胀：滑动窗口增量更新（moving_average 实测 ~48× 计算冗余）
- ✅ 消除访存冗余：实际访存÷最小必需 >2× 时先修访问模式（histogram 实测 8× 跨步非合并）
- tiling 提高数据复用；SoA/AoS 选择；合并小 kernel 减少启动开销

**kernel 代码层**
- 访存合并：同 wave 相邻线程访问相邻地址（连续 thread-contiguous 下标）
- ✅ 直通 LDS（buffer_load...lds）替代"load 寄存器 + ds_write"中转：省 VGPR 抬占用率——gfx1030 硬件支持且常被库忽略（aiter gfx1030 路径未启用，实测模板见 mha op_lds.hpp:188-235）
- ✅ 软件流水两级预取：s_clause 批量 + vmcnt(N) 部分排空，等待深度按"下一 tile 消费点"定（aiter mha pipeline.hpp 实测编排）
- 数据类型：逐字节 RMW → 合适宽度的 word 类型（histogram 教训：uchar bins 导致逐字节 LDS 操作）
- 归约结构：防 bank conflict（同 lane 落同 bank 的布局）、树形归约
- 占用率调参：block size 扫描（256/128/64…）、`__launch_bounds__` 限 VGPR、动态共享内存换静态

**编译器层**
- ✅ 编译级别（-O0 → -O3 一项可达 5-20×，histogram 盲测实证）——先于一切代码优化
- 向量化：标量 `global_load_dword` → `float4`/`dwordx4`（ISA 直方图验证）
- 显式 `--offload-arch`、查 VGPR spill（scratch 段）、循环展开过度/不足

**驱动/运行时层**
- ✅ DVFS：同进程 warmup+多迭代（跨进程无效，盲测实证）
- ✅ dtype 选择是单变量大杠杆：bf16→fp16 单变量 3.41×（aiter.add 实测——RDNA2 bf16 路径读放大 7.4×，fp16 路径 amp 1.00；非普适但值得首轮即试）
- 并行度：blocks ≥ CU 数的若干倍（histogram 实测 8 blocks vs 80 CU 空转 90%）
- 拷贝路径：大块走 SDMA/copy engine，避免 copy kernel；合并小传输

## 优化报告（写进研究档案）

```
## 性能优化：<对象>
- 基线：X us（达标率 Y%，gpu-perf 报告链接）
- 第 1 轮：<改动> → Z us（达标率 Y→Z%）| 依据：<哪条证据>
- ...
- 终态：达标率 X%，收敛原因；未竟事项与假设
```
