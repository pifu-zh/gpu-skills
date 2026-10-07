---
name: gpu_lib_port
description: 把 GPU 库适配/移植到它未支持的平台或架构的工艺（跨平台语义移植：宏隔离、最小修改、回退 kernel、UT 驱动、升级友好）。Use when 用户要把库移植到新硬件/架构（如把 CUDA 库适配到 gfx1030）、库不支持我们的卡需要适配、或说"移植/适配/跑通 XX 库"——与 gpu_lib_build（同平台构建适配）区分，port 是跨平台语义移植，build 是编译工艺。
---

# GPU 库跨平台移植（gpu_lib_port）

核心命题：移植不是重写，是**在不惊动原代码的前提下，为目标平台造一条语义等价的路**。原代码的完整性就是升级能力——删掉的每一行都是未来同步上游时的一笔债。

## 移植五纪律（用户校准，硬约束）

1. **禁止删除源代码**：用源平台宏（`__CUDA_ARCH__`/SM90 类）与目标平台宏（`__HIP_DEVICE_COMPILE__` 类）隔离，必要时加专用宏（如 `DEEPGEMM_ENABLE_RDNA2`）
2. **最小修改**：新文件 + 路由点分支 优先于 内联散改；upstream tracked 文件的改动以"行数"计量并向用户汇报
3. **UT 驱动**：适配的每个功能必须有测试验证；先手工最小数值验证（与参考实现 allclose），再上游测试，再固化为永久单测
4. **可随上游升级**：改动集中、统一标记（如 `[PORT <库名>]` 注释）、产出脆弱点清单
5. **不支持 ≠ 放弃，硬件缺失 ≠ 功能缺失**：GPU 功能语义没有真正无解——矩阵乘/异步拷贝/barrier/精度格式全部有软件等价路径（FFMA 展开、软件流水、位操作解码、蝶形归约……实证清单见 references/nv-amdgcn-map.md 第六节）。做法分两层：①参考已有库的同类处理（aiter 是范本：fp8 回退 uint8、非 gfx9 回退 Triton、CK 缺失 shim）②没有现成就自己写软件等价实现（正确性优先，性能损失如实记录）。"stub 抛错"只允许出现在软件等价**尚未编写**的过渡态，且必须登记为待办——它是待办标记，不是终点

## 移植流程（六步）

### 1. 侦查：三方差距表 + 难度定级
- **上游现状**：构建体系、条件编译分布（grep 源平台宏）、是否已有目标平台的半支持、**旧移植尝试的漂浮文件处置**（不被编译体系引用的通常已损坏——弃用不接通）
- **目标平台能力差距表**：逐项列 指令（矩阵核/异步拷贝/同步原语）、数据类型（fp8/bf16）、构建链、运行时 API；用编译探针法实证（最小 builtin 调用，报错直接给缺失 feature 名）
- **路由点定位**：库在哪里决定"用哪套实现"（架构判断/kernel 选择/JIT 编译命令组装）——这里是你唯一被允许大量动刀的地方
- **构建期环境写操作检查（pip 系必做）**：grep setup.py/pyproject 的模块级 `subprocess`/`pip install` 调用——安装动作可能连锁替换环境已有包（FA 实例：FA→强制装旧版 aiter→卸载环境 triton 重装，两级链）。防护手法：`PIP_CONFIG_FILE`（prefix+ignore-installed 让强制安装落隔离目录）+ 官方开关（AITER_USE_SYSTEM_TRITON 类）+ `pip wheel --no-deps`；包属 root 不可覆盖时用 root exec 或 --target+PYTHONPATH 接入
- **对嵌入产物先 objcopy/提取再 A/B 对照**：宿主侧契约缺口的假设（如"launch 少传参数"），提取嵌入 bundle 里的 code object、跑 6 参 vs 7 参对照实验，一次实验直接证明——比代码推理快一个量级
- **替换 FFI crate 前清点依赖方自己的 build 脚本探针**：薄替身 crate 消不掉上游 crate 自己 build.rs 里的 CUDA 头探针（multicast cfg 类）——patch 表只换调用面，build 面要单独处理
- **探针必须镜像源语义的全部分支（含假值/非阻塞路径）**：编译探针 PASS ≠ 语义正确——test_wait 类谓词原语的软件实现若写成阻塞自旋，真值路径探针全绿、假值路径 GPU 死锁（99% 挂起实证）。最小探针的分支覆盖=源语义分支覆盖
- **对象化同步原语（mbarrier 类 {count,phase} 状态机）优先单字状态**：塞进原对象字宽做单原子操作（借硬件布局注释），side-car 双字在相位边界的竞态窗口编译期不可见、只在真机暴露
- **探针/改写的执行顺序也是规格的一部分**：手工链路跑在 opt 之后，集成化时 prep pass 自然写在 opt 前——设备函数内联时机差会让签名追加型改写追加到错误函数上，产生 verifier 查不出的静默 ABI 失配（call 自带 callee 类型自洽，LLVM 不查 arity）。Stage 1 实证：prep pass 必须拆两段（alloca 修复在 opt 前=verifier 契约、sreg/签名改写在 opt 后=内联收敛后）
- **环境 Inventory 前置（判定测试失败/管线缺口归因前）**：先盘点三件套——libdevice.10.bc / ptxas / 同 major llvm-link。实证：一个环境缺口（CUDA13 拆包缺 libdevice+ptxas）同时伪装成"测试失败"（libdevice_linking 5 failed）和"管线缺口"（math_tan 转绿后暴露真因）两个领域的故障——补两件环境件，两处全绿。产出分级：**"产物存在"≠"可加载"**（llvm-nm --undefined-only 把 hsaco-pass 分成自包含/需链接两级，否则链接期失败藏进通过率）
- **环境 Inventory 前置（判定测试失败/管线缺口归因前）**：先盘点三件套——libdevice.10.bc / ptxas / 同 major llvm-link。实证：一个环境缺口（CUDA13 拆包缺 libdevice+ptxas）同时伪装成"测试失败"（libdevice_linking 5 failed）和"管线缺口"（math_tan 转绿后暴露真因）两个领域的故障——补两件环境件，两处全绿。产出分级：**"产物存在"≠"可加载"**（llvm-nm --undefined-only 把 hsaco-pass 分成自包含/需链接两级，否则链接期失败藏进通过率）
- **批量工具的产物选择必须对同目录多管线产物免疫**：精确文件名匹配，不用宽 glob——上一轮集成管线遗留的 `.amdgcn.preopt.opt.ll` 排在 glob 最前，会让"Python 管线回归"的统计混入双重改写产物，**测量污染伪装成语义回归**（cuda-oxide 实证：v4 记录 9/17 回归，修正产物选择后干净基线实为 12/14，真形态差异仅 4 处）
- **重放验证工艺（跨基线移植的标准验收手段）**：重放完成后，逐 commit 机械对比——`diff <(git show 原sha --format=) <(git show 重放sha --format=)` 逐字节 diff + **差异三分类**（行号偏移 / 基线字段缺失 / 语义分歧），把"冲突解决是否丢语义"从人工推理降为机械检查。实证：cuda-oxide 重放演练 10 commits——6 个逐字节一致、4 个差异全归因基线（`grid_constant_kernels`/`module_disambiguator`/ntid 参数形态=基线本来没有的正确形态），零语义丢失可证。配套：产物链可作物证（`.amdgcn.preopt.ll`/`.amdgcn.preopt.opt.ll` 固化 prep pass 拆段顺序）；基线差异三形态先验清单（重放分支少的东西不一定是丢失，可能是基线本来就没有）
- **批量前先单例 dry-run + 形态盲区检查**：批量跑 N 个对象前，先对至少 1 个"非理想形态"样本做完整 dry-run（批量工具对 .opt.ll 完美形态工作≠对普通 .ll 工作——签名追加若不容忍 flags 会批量翻车，cuda-oxide 实证：批量才发现、单例漏检）
- **难度定级（动手前必做）**：先跑目标平台的最小用例定级——开箱即用（只补架构门）/小修（注册表+编码分支）/深修（需回退 kernel）/不可行（硬件能力缺失）。FlyDSL 实例：预判"编译器级大工程"，实测 pip wheel 开箱即用 vectorAdd，真正缺口仅 3 处 Python 层架构门——定级失误会导致工程量错配
- **上游同步后的例源语义演化**：回归归因先分辨"形态漂移"（改写器正则失配）与"用例语义升级"（例源借 #NNN 演进改变行为，如 1-D launch 假设类折叠在 3-D launch 下变错误语义）——两者的修法完全不同（前者适配正则，后者修正语义预期）
- **纯 Python 层判断**：若目标库是"Python + 预编译绑定"形态（自包含 wheel），C++/MLIR 组件无需重编——适配面收缩到 Python 层架构门 + 编码分支
- **依赖隔离**：容器/环境里已装的发布版一字不动（下游如 aiter 可能 pin 它），源码仓适配用 PYTHONPATH/venv 接入验证

### 2. 差距 → 替代决策树（每项缺失能力过一遍）
```
指令缺失（矩阵核/TMA/异步原语）
  → 回退 kernel（FFMA 等通用指令实现，正确性优先）
  → 软件流水（类 TMA 6 条技术：buffer_load_lds 直通/s_clause 批量/vmcnt 部分排空/描述符 OOB 填零/多 buffer 旋转/地址递归链防 spill）
数据类型缺失（fp8/bf16）
  → 软件解码（平台官方头，如 hip_fp8.h 的位操作）/ 存储回退（uint8 容器 + 上层转换，aiter 式）
整条实现路径缺失
  → 新回退 kernel + 同签名包装挂路由层（保持上游函数签名，调用方无感）
判断依据：搜已有库对同类缺口的处理（同生态的移植分支、官方软件实现头文件）；替代方案的"语义契约"必须从原实现和测试生成器双向核对（scale 布局、stride 约定这类细节是数值 bug 高发区）
```

**适配资源清单**（查替代/API 对照的权威来源）：
- **NV→AMDGCN 映射注册表：references/nv-amdgcn-map.md**——已实证的 intrinsic/原语/数据搬运映射单一事实源（含 TMA 类软件流水范式引用）。**闭环纪律：改写/适配完成后必须做"残留审计"（grep 目标平台不认识的源平台符号）——残留项先查注册表找替代，注册表没有的探针实证后回填注册表**。"未触发"≠"不存在"：缺口只有遇到才会暴露，暴露一条注册一条
- **CUDA2HIP 映射**：HIPIFY 生成（`hipify-clang --md --doc-format=full`），CUDA API→HIP 对照的权威字典；hipify-perl 可做快速预处理草稿——但机械转换是起点，语义适配（架构指令/内存模型）必须手做
- 同生态库的移植分支/文档（如 aiter 的 docs_cn 设计分析、上游对相邻架构的支持提交）
- 平台官方软件实现头（fp8→hip_fp8.h 位操作解码等）
- 上游注释/文档的**能力声明要探针实测**——aiter 实例：BufferCopyLDS 标注 CDNA3-only，gfx1030 实测可用（"注释被打脸"）；反向也成立（声称支持的可能有静默守卫）

### 2.5 决定性探针先行（跨编译链适配的核心工作法）
集成化之前，**手工改中间产物验证整条链的每一段**（cuda-oxide 实证：先于任何源码修改，用 ~15 行 Python 脚本手改 LLVM IR 的 triple/CC/intrinsic，llc 编 code object，HIP host 加载执行——每一步都可证伪）：
- 每个替换点（triple、调用约定、intrinsic 映射）都是 Phase 2 自动化的预演——探针脚本的每个替换就是未来代码的规格说明
- 探针暴露的真问题（如驱动拒收某版本 code object）在写一行正式代码前就被发现，代价最小
- 探针通过 ≠ 集成化可跳过：探针的手工替换在集成化时全部变成路由点实现

### 3. 实施：路由层 + 新文件
- 路由点用 宏 + env + capability 判断三件套隔离（注意：**torch 会把 gfx1030 报成 capability (10,3)**——arch major 判断会误入 Blackwell 分支，需显式映射）
- 回退实现放新文件；与上游同签名包装（调用方无感）；upstream 文件只允许 #ifdef 级包裹
- host 侧构建小心 **torch CUDAExtension 的 hipify 会就地改写源文件**（静默覆写，移植中最大陷阱）——用纯 setuptools.Extension 绕过
- 模块加载异步竞态（ROCm 7.x 实测）：新模块立即 launch 可能静默空转——同模块 warmup kernel + 主机验证

### 3.5 IR 层适配（编译器/DSL 类项目）
**地址空间契约（NVPTX 容忍 / AMDGPU module verifier 拒绝，cuda-oxide 批量实证）**：①generic addrspace 的 alloca 必须 addrspace(5) scratch（align 在前：`align N, addrspace(5)`），每 alloca 的 use 插 addrspacecast；②AS(3) 全局量不可 `zeroinitializer`（LDS 不可静态初始化 → undef）。**层叠契约**：修一层后才看见下一层（alloca→membar/isspacep、barrier→LDS zeroinit）——批量工具的逐例 first_error 归类是必要设计。

当目标项目自己产出中间表示（LLVM IR/PTX/bitcode）时，移植的关键是**切入层分析**：找到"工具无关的中间层"——该层以上平台无关，以下换成目标平台的工具链（cuda-oxide 实例：.ll 出口是文本 LLVM IR，llc 换 `-march=amdgcn` 即切后端；且 intrinsic 分发常有现成挂点，如 IntrinsicBackend 枚举）。

IR 改写工艺（实测）：
- **同名替换保 SSA 引用不变**：`%vN = call @llvm.nvvm.sreg.tid.x()` → `%vN = call @llvm.amdgcn.workitem.id.x()`——只换被调名，所有引用不用改
- 无等价物的（如 blockDim 由 host 决定）→ **改签名新增参数**由 host 经 kernarg 传入
- launch 维度判断等语义分支 → 常量化（`or i32 0, 1` 原地替换，同理保引用）
- 调用约定与头部：`ptx_kernel` → `amdgpu_kernel`、triple/datalayout 按目标替换
- **改写脚本必须 idempotent**：从原始产物单向生成目标文件，禁止对目标文件多轮补丁——多轮补丁会引入 SSA 重名/残留引用等次生错误（实证：改写文件被 4 轮补丁搞乱后，从源产物一次性重生成才收敛）

### 4. UT 驱动验证
- 层级：手工最小数值（vs torch/参考实现 allclose）→ 上游测试子集（测试太重可用运行时 monkeypatch 缩样，**零测试文件改动**）→ 永久单测固化（`test_<port>.py`，下次上游同步后的回归防线）
- 失败三分类照 gpu_lib_test；数值 bug 高发区：数据布局契约（scale 布局、分组 stride——如"k_grouped 是每组行主拼接的扁平缓冲，m-stride=该组 K"这类隐藏约定）

### 5. 性能抽样
跑通优先，数字如实记录（回退实现的预期就是低：DeepGEMM FFMA fallback 3.05 TFLOPS vs SM90 原生 100+）；对照目标平台理论峰值给达标率；优化另立项（gpu_lib_optimize）

### 6. 沉淀：升级友好性包
- **脆弱点清单**（按失效概率排序）：fork 的整文件（需 diff 重移植）> 路由块 > 数据布局契约 > 架构映射表 > 构建脚本
- 每处修改带 `[PORT]` 注释与理由；复现命令写死在档案
- 上游同步 SOP：git pull → 重跑永久单测 → 按脆弱点清单 diff 检查

## 高频坑速查（两次移植实测）

1. torch CUDAExtension HIP 构建 = 全项目 hipify 就地改写源码（DeepGEMM 实证）
2. torch capability 误判（gfx1030 → (10,3) → Blackwell 路径）
3. 架构注册扫描法：所有架构键控表逐个过（GFX_MAP/allowed_archs/CU 数表/LDS 容量表——aiter 实测 5 处，漏一处运行时 ValueError）
4. 包内预编译产物两宗罪：host-only .so 的 undefined symbol（清产物触发 JIT）；调参数据库不含目标架构 → 零 kernel（hipblaslt 教训："代码能编译 ≠ 有 kernel"）
5. 旧移植尝试的漂浮半成品文件：弃用，不接通（几乎必然已损坏且违背最小修改）
6. 显式声明不支持的边界：stub 抛错带原因 > 静默错误输出（排障成本天差地别）；库自身对不支持的架构若用进程崩溃式报错（LLVM fatal abort），适配时先换成显式异常（FlyDSL 实例：MFMA 撞墙从 fatal 改 ValueError）
7. **ISA/编码与 intrinsic 语义参数差异是 DSL/汇编层的高发缺口**：①同名指令不同架构位域编码不同（gfx10 s_waitcnt vmcnt 分裂编码 vs gfx11 连续编码，llvm-mc 探针实证）；②同名 intrinsic 语义参数不同（warp shuffle：NV 传 lane 号、AMD 传字节偏移 lane*4；NVVM shfl 实产签名 4 参 vs PTX 助记直觉 2 参）——**映射表必须从实产调用形态抓取**（grep 真实 .ll），先编译探针验证签名再写映射；③预编译产物可能内嵌 PTX asm（f64 shuffle），跨架构需按模式展开（cuda-oxide：f64 shfl → 双 ds_bpermute 拆分）
8. **pip 安装链的隐性污染**：setup.py 模块级的强制 pip install 会连锁卸载/替换环境已有包（flash-attention→aiter→triton 两级链实证），且 PIP_PREFIX 挡不住 uninstall 步骤——装前必 grep，装后必验证环境终态（关键包版本 diff）
9. **code object 版本的 partial-support 陷阱**：新版本产物被老驱动"部分支持"——加载 ✓、执行 ✓、结果 ✗（v6 在旧内核驱动上隐式参数失效，索引恒 0；cuda-oxide 实证）。比直接拒绝更阴险。判定法：**官方栈同逻辑 kernel 对照实验**（官方产物正常→我们的产物有问题，一步隔离）；一切 device 产物显式版本对齐（`-mcode-object-version=5` 类），并在 UT 里断言结果而非仅断言"跑完"
10. **构建期 SDK 探测的绕过**：外部 FFI crate 的 build script 探测 CUDA/SDK 时，从官方 repo 拆包下载最小头集即可满足 bindgen/编译期（CUDA13 拆包实证：cuda.h 在 cuda-cudart-dev、curand.h 在 libcurand-dev、crt/ 已扁平化需手工 symlink 兼容层；build 期探测的 runtime 库可用 dlopen 不参与链接）——不要为此装整个 SDK，也不要手写假头
