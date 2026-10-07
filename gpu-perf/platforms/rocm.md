# ROCm/HIP 专用模块（层级 L2：平台）

层级：GPU（SKILL.md）→ **本文件：平台** → 架构（platforms/rocm/<架构号>.md，如 gfx1030.md）→ 型号/机器（项目本地 gpu-perf-baseline.md）。
本文件只写 ROCm 平台工具链知识（对任何 ROCm 机器成立）；架构相关（计数器语义、ISA 判读、工具支持矩阵、架构属性）去对应架构文件；机器数据永不入 skill。标注"实测"的命令在 RDNA2 消费卡 + ROCm 7.14 上验证通过。

## Phase 0：物理层基线实操

**理论规格来源**：

```bash
rocminfo | grep -A20 "Marketing Name"   # 架构/CU/SIMD/时钟/heap（显存）大小
rocm-smi                                # 当前时钟、功耗、温度、PwrCap、VRAM 占用
rocm-smi --showclocks                   # sclk/mclk 当前频率
```

**实测基线生成流程**（结果保存到项目本地 `gpu-perf-baseline.md`）：

```bash
# --bench 传任意带宽微基准（官方 rocm-bandwidth-test 或自写的 copy kernel 基准均可）
python3 <skill>/scripts/gpu_probe.py --bw <spec带宽GB/s> --bench <带宽微基准路径>
```

微基准首选官方 `rocm-bandwidth-test`（ROCm 全栈分发或单独构建）。**口径注意**：其 D2D 数字是 copy 口径（1 字节拷贝 = 1 读 + 1 写显存流量），与 spec 单向带宽对比前先除以 2 或注明口径。小块（<256MB）与 paged 内存的测量远低于稳态值，不可当基线。

基线文件应包含：设备台账（探针输出）、三个带宽实测值、时钟行为（空闲/短负载/持续负载）、本机工具限制（哪些分析工具不支持本架构）、可复用基准脚本路径。写法示例见 `~/workspace/docs/gpu-perf-baseline-gfx1030.md`。

**实测基线**：

```bash
python3 <skill>/scripts/gpu_probe.py --bench ~/workspace/src/rocm-examples/build/bin/HIP-Basic/hip_bandwidth
```

hip_bandwidth 实测参考值（paged 内存，1MB/5MB 两档）：H2D 6.0→13.6 GB/s、D2H 5.5→13.7 GB/s、D2D 91→158 GB/s。要点：小缓冲远低于稳态值，**测基线必须用大块（≥256MB）+ 多次迭代取稳定段**；上表数字本身演示了"传输尺寸对带宽的影响"。

**延迟参考**（从实测 API 时间线，hip_saxpy 首次运行）：hipMalloc 首次 ~4 ms/次（上下文初始化）、hipLaunchKernel 首次 ~669 us。**稳态测量必须 warmup 后再测**，首次调用数字只用于发现"没做 warmup"这类问题。

## Phase 1：环境匹配清单（ROCm）

| 检查项 | 命令 | 通过标准 |
|---|---|---|
| 驱动加载 | `lsmod | grep amdgpu` | amdgpu 已加载 |
| ROCm 版本 | `rocminfo | grep "Runtime Version"` | 与编译/分析工具同栈 |
| GPU 可见 | `rocminfo | grep gfx` | 列出 GPU agent（gfx 后跟架构号） |
| 设备权限 | `ls -l /dev/kfd /dev/dri/*` | 进程属主有 rw（通常 video/render 组） |
| 计数器通道 | `rocprofv3-avail list --pmc` | 非空；消费卡多数不在 rocprof-compute 支持矩阵（其报错会列出支持架构），不支持时通道见下方"计数器采集" |
| 时钟策略 | `rocm-smi --showclocks` | sclk 未被钉死在低频；需要锁定时用 sysfs `power_dpm_force_performance_level`（root，未验证） |
| 容器直通 | `docker inspect <c> --format '{{.HostConfig.Devices}}'` | /dev/kfd + /dev/dri 都在；宿主目录挂载路径因环境而异（报告脚本随挂载可见即可在容器内直跑） |

## Phase 2：测量实操

### 标准采集（时间线全景）

```bash
mkdir -p /tmp/prof && rocprofv3 \
  --kernel-trace --hip-runtime-trace --memory-copy-trace \
  -f csv -d /tmp/prof -- ./程序 [args]
```

产出 `/tmp/prof/<主机名>/<pid>_*.csv`：`kernel_trace`（Kernel_Name、Start/End_Timestamp(ns)、Grid/Workgroup、VGPR/SGPR）、`hip_api_trace`（Function、时间戳）、`memory_copy_trace`（Direction、时间戳，**无 Size 列**）。

```bash
python3 <skill>/scripts/gpu_profile_report.py --csv-dir /tmp/prof/<主机名> [--api]
python3 <skill>/scripts/gpu_profile_report.py --db /tmp/prof/<主机名>/<pid>_results.db   # rocpd 格式（含 --pmc 值）
```

报告脚本零依赖，容器内可直接跑（挂载路径 `/home/ubuntu/.zcode/...`，免去 docker cp）。

### PyTorch 算子 profiling（实测流程，需容器/环境内有 torch+ROCm）

三步：写 bench 脚本（**warmup ≥10 次让 DVFS 爬升** + `torch.cuda.synchronize()` 包住计时）→ `rocprofv3 --kernel-trace --pmc GRBM_GUI_ACTIVE FETCH_SIZE OccupancyPercent -d out -- python3 bench.py` → 报告脚本读 db。两条实测得出的通用结论：

- torch.matmul 走 hipBLASLt，大 GEMM 在 RDNA2 消费卡上可达实际时钟理论的 80%+（大 tile occupancy ~20% 是常态，不是问题信号）
- **复合算子（torch.fft 等）先做 kernel 分解再判定**：管线 kernel（copyBuffer/elementwise/类型转换）占比常常远超计算 kernel 本身——瓶颈在管线不在核心；计算 kernel 自身是否达标要用其独立时间 × 数据量单独算

### 计数器采集（消费卡通用通道：rocprofv3 --pmc）

```bash
rocprofv3-avail list --pmc        # 全量清单
# 计算强度+占用+访存一次看全（实测可单 pass）：
rocprofv3 --pmc SQ_INSTS_VALU SQ_INSTS_SALU SQ_INSTS_LDS SQ_INSTS_FLAT \
  FETCH_SIZE GRBM_GUI_ACTIVE OccupancyPercent MemUnitBusy -d /tmp/prof -- ./程序
```

`--pmc` 的值**只进 rocpd db**（`rocpd_pmc_event` 表），用 `gpu_profile_report.py --db` 读。单次别超过 ~8 个硬件 block 计数器。常用计数器清单、单位与解读陷阱 → `platforms/rocm/<架构号>.md`（架构层）。

### rocpd 数据库速查

默认输出格式（无 `-f` 时）。表名带 UUID 后缀（LIKE 匹配），核心 join（实测）：`kernel_dispatch.kernel_id = info_kernel_symbol.id`；`pmc_event.pmc_id = info_pmc.id`；`pmc_event.event_id = kernel_dispatch.event_id`；`memory_copy.name_id = string.id`。默认不含 HIP API 明细（要 API 用 `-f csv`）。

rocpd2* 官方工具只支持 Python 3.10–3.13，在系统 python 3.14 的机器上直接 ImportError（修复需旧版 python + 库路径 + sqlite 预加载，且之后还缺 pandas）——**别修，直接 `-f csv` 或用报告脚本**。

## Phase 4：编译器/驱动深挖实操

### 源码与产物分析（预处理去宏 + ISA 直方图，均实测）

```bash
# 1) 宏展开后看真实逻辑：HIP_CHECK/__global__ 等宏全部展开，只留本文件的代码
#    <gfx目标> 换成你的架构（rocminfo 里 "Name: gfxXXXX" 给出）
hipcc -E -O3 --offload-arch=<gfx目标> -I <Common目录> main.hip > /tmp/pp.txt
awk '/^# [0-9]+ "/ {f=$3; gsub(/"/,"",f); inmain=(f ~ /main\.hip$/); next} /^#/{next} inmain && NF' /tmp/pp.txt
# 注意：linemarker 格式是 `# <行号> "<文件>"`，文件名字段是 $3；末尾正则按源文件名改

# 2) ISA 指令直方图：反汇编的主视图，一眼看指令混合
awk '/^\t[a-z]/ {print $1}' <src>-hip-amdgcn-amd-amdhsa-<gfx目标>.s | sort | uniq -c | sort -rn | head -15
```

### 编译器层

```bash
# 显式目标架构（<gfx目标> 见 rocminfo）+ 保留中间产物
hipcc --save-temps -O3 --offload-arch=<gfx目标> -I <Common目录> main.hip -o app
# 产出 <src>-hip-amdgcn-amd-amdhsa-<gfx目标>.s 即设备 ISA；备选：llvm-objdump -d *.o
/opt/rocm/llvm/bin/llvm-objdump -d app-hip-amdgcn-amd-amdhsa-<gfx目标>.o
```

ISA 特征判读（向量化/融合乘加/等待屏障/spill 等指令模式）→ `platforms/rocm/<架构号>.md`（架构层）。反汇编对不上源码时的第一嫌疑：编译 flag 没带 `--offload-arch`（生成通用目标）或优化级别不是 -O3。

### 驱动/运行时层

- 实际时钟：`GRBM_GUI_ACTIVE ÷ kernel时长`（最可靠），`rocm-smi --showclocks` 看当前
- 传输引擎：D2D 大拷贝确认是否走了 copy kernel（`__amd_rocclr_copyBuffer`，受 kernel 效率拖累）——大块 D2H/H2D 应走 SDMA（memory_copy_trace 有记录而 kernel_trace 无对应行为）
- kernel 太碎/小传输风暴：API 时间线里 hipMemcpy/hipLaunchKernel 次数远大于热点 kernel 数
- 时钟锁定与功耗策略：sysfs `power_dpm_force_performance_level`（root，未验证，动前先确认必要）

## 坑表（ROCm 7.x 实测）

0. **临时产物及时清理**：`--save-temps`/`-E`/`--pmc` 产生的 .i/.bc/.o/.s/.hipfb/pp.txt/results.db 全部写 /tmp 下的专用目录，分析结束 `rm -rf` 掉；repo 目录内不留任何生成物
1. rocprofv3 7.x 默认输出 rocpd SQLite，**要 CSV 必须显式 `-f csv`**
2. rocpd2* 系列 Python 3.14 直接 ImportError；修复配方复杂——用 `-f csv` 或报告脚本
3. **rocprof-compute 支持矩阵有限**（不含多数消费卡，报错信息会列出支持架构）——不支持时硬件计数器唯一通道 `rocprofv3 --pmc`；具体架构支持情况查架构层文件
4. `--pmc` 的计数器值不进 CSV，只进 rocpd db
5. 时间戳全为纳秒；多流重叠时 kernel 时间求和略大于墙钟
7. **DVFS 短负载陷阱**：短负载时钟可能停在峰值的一小部分——按 spec 峰值折算达标率会误判。测量要 warmup+多迭代，判定用 `GRBM_GUI_ACTIVE÷时长` 还原实际时钟（本架构具体行为查架构层文件）。**warmup 必须在同一进程内**——跨进程重复运行不继承时钟状态（盲测实证：同一程序连跑 20 次、每次新进程，时长无下降趋势）
8. 编第三方示例源码缺头文件时，先查其仓库根的公共 include 目录（如 rocm-examples 的 `-I <repo>/Common`）
9. **CSV/rocpd 的 Grid_Size_\* 字段是总线程数（gridDim×blockDim），不是 HIP 语义的 blocks 数**（实测三例吻合：3907 blocks 报 1000192、8 blocks 报 1024、64×128 blocks 报 1024×2048）——报告脚本已自动换算；判定并行度（blocks vs CU 数）时务必用换算值，否则会把"严重欠载"误判为"并行度足够"。LDS_Block_Size/Scratch_Size 两列语义也未按字面（32KB LDS 实报 LDS=0/Scratch=1024），用前与源码对照
10. **CMake 项目默认可能是 -O0 设备码**（CMAKE_BUILD_TYPE 空）：ISA 特征为无 `ds_` 指令（共享内存走泛型 flat 寻址）、逐字节 flat 访存、scratch 溢出、`__syncthreads` 未内联、零循环展开——此时一切测量反映编译器的锅，先 `CMAKE_BUILD_TYPE=Release` 重编再分析。不重编译也可从二进制提取 `.hip_fatbin`（objcopy）后用 `llvm-objdump -d --triple=amdgcn` 反汇编验证（盲测实证此路径可行）
