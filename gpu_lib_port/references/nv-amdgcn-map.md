# NV → AMDGCN (gfx1030/RDNA2) 映射注册表

单一事实源：跨项目适配时**先查本表再探针**。每条映射标注实证状态；新映射发现后必须回填本表。
探针法兜底：本表没有的 intrinsic，写最小调用编译验证（报错直接给缺失 feature/说明）。

## 状态图例
✅ 已实证可用（真机+数值）｜🔶 回退实现（正确性优先）｜❌ 硬件缺失（显式报错）｜⏳ 未实现（有方案）

## 一、special registers / 索引
| NV (llvm.nvvm.read.ptx.sreg.*) | AMDGCN (llvm.amdgcn.*) | 状态 |
|---|---|---|
| tid.x/y/z | workitem.id.x/y/z | ✅ |
| ctaid.x/y/z | workgroup.id.x/y/z | ✅ |
| ntid.x/y/z | workgroup.size.x/y/z（或 host 经 kernarg 传） | ✅ |
| nctaid.* | （host 经 kernarg / 常量化） | ✅ |
| laneid | mbcnt.lo(-1, 0) | ✅ |
| lanemask_lt/eq | mbcnt.lo / (1<<laneid) | 🔶 |

## 二、同步 / barrier
| NV | AMDGCN | 状态 |
|---|---|---|
| barrier0 / bar.sync 0 | llvm.amdgcn.s.barrier() | ✅ |
| barrier.cta.sync.count / arrive.count（编号/计数 barrier） | LDS 计数器+世代自旋软回退（防 ABA 全 seq_cst 原子） | 🔶 |
| mbarrier.init/arrive(.noComplete)/test_wait.parity/inval | 🔶 **软件单字 i64 状态机**：对象自身 8B = pending[31:0]/expected[47:32]/phase[63:48]，单原子操作消除双字竞态；init=整字原子写 (n<<32)\|n；arrive=fence(release)+atomicrmw sub（pending==1 观察者补完成写+phase+1）；noComplete=纯 sub 永不补完成写；**test_wait=非阻塞单次读+相位比较（阻塞语义属于调用方循环——实现成阻塞式在 noComplete 分裂模式会死锁，GPU 99% 实证）**；inval=no-op；expect_tx 字节级事务不支持（等价=vmcnt 部分排空）；expected≤2^16（强于硬件 1 位 parity） | 🔶 |
| mbarrier.expect_tx（字节级事务） | vmcnt(N) 部分排空等价（异步拷贝场景）| 🔶 |
| membar.gl / fence.acq_rel.gpu | fence seq_cst syncscope("agent") | ✅ |
| membar.cta / fence.acq_rel.cta | fence seq_cst syncscope("workgroup") | ✅ |
| membar.sys / fence.sc.sys | fence seq_cst（空 scope=system） | ✅ |

## 三、warp 原语
| NV | AMDGCN | 状态 |
|---|---|---|
| shfl.sync.idx（f32/i32） | ds_bpermute（**index 参数是字节偏移 lane×4，NV 是 lane 号**）；段式：seg=lane&-32 | ✅ |
| shfl.sync.bfly / up / down | 同表段内 lane^d / 越界回自身 | ✅ |
| shfl f64 | NV 侧内嵌 PTX asm → 双 ds_bpermute 拆分 lo/hi 重组 | ✅ |
| redux.sync.add | 5 轮 xor-butterfly ds_bpermute 回退（O(log32)，性能待测） | 🔶 |
| vote.ballot | llvm.amdgcn.s.ballot/balli？探针后定 | ⏳ |

## 四、数据搬运 / "TMA 类"
| NV | RDNA2 等价 | 状态 |
|---|---|---|
| cp.async（gmem→smem 异步标量） | 软件流水：s_clause 批量 + vmcnt(N) 部分排空 + 寄存器中转 | 🔶 **已实现+量化**（微基准 365 GB/s = plain 基线 78%，流水 A→C +30% 有效；瓶颈结构性：直通 LDS 仅 b32 宽度，指令密度 4-7×，发射端口打满——soft-tma.md） |
| cp.async.bulk.tensor（TMA） | 🔶 软件等价已实现+量化：双缓冲深流水 365 GB/s（78% plain 基线，未达 90% 线）；瓶颈 = 直通 LDS 仅 b32 宽度（汇编器无 dwordx2/x4 编码，指令密度 4-7×）。六条技术全文见 docs/research/aiter/tma-like.md §5，微基准数据见 docs/research/cuda-oxide/soft-tma.md | 🔶 |
| TensorMap 越界 fill | V# 描述符 OOB 硬件填零（经 builtin 恒开、不可关） | ✅ |
| TensorMap 硬件 swizzle | ❌ 无——软件 padding（行距 272/144B 实证先例） | ✗ |
| mbarrier expect-tx | vmcnt(N) 部分排空（粒度=条数非字节） | 🔶 |
| libdevice __nv_* 数学 | ocml.bc 的 __ocml_*_f32/f64（**llc 不行，须 clang -c 或 llvm-link 连 oclc 配置 bc**——裸 ocml.bc 留 __oclc_* 未定义） | ✅（__nv_tan→__ocml_tan_f32 实证） |

## 五、内存 / 原子
| NV | AMDGCN | 状态 |
|---|---|---|
| isspacep.local / .shared / .global | llvm.amdgcn.is.private / .shared / .global（签名 i1(ptr)） | ✅ |
| atom.global.add.* 等 | atomicrmw（scope 映射：gpu→agent、cta→workgroup） | ✅ |
| NV 内嵌 PTX asm 原子族（fence.acq_rel/ld.acquire/st.release b32/b64） | LLVM 原子指令改写（ld.acquire → load atomic acquire 等） | ✅ |
| idp4a.s.s / .u.u（int8 dot4） | sdot4/udot4 intrinsic（lowering 到 v_dot4c_i32_i8） | ✅ |
| idp2a（16×8 混合） | 乘加展开（无对应指令） | 🔶 |
| alloca（generic） | **必须 addrspace(5)**（align N 在前）+ use 插 addrspacecast | ✅ |
| AS(3) 全局量 zeroinitializer | **必须 undef**（LDS 不可静态初始化） | ✅ |

## 六、硬件缺失项的软件等价路线（没有"放弃"，只有"还没实现"）

**原则：GPU 功能语义不存在真正无解——矩阵乘/异步拷贝/barrier/精度格式全部有软件等价路径（本机全部实证过）。** "显式报错"只是软件等价**尚未编写时**的过渡保护（防静默错误数值，反面教材：CK 的 amdgcn_mma dummy 透传返回 C 打 WARNING）；每个 ❌ 的终态是升级为 🔶 软件实现 + 性能损失如实记录。

| 硬件缺失 | 软件等价实现 | 状态 | 性能预期 |
|---|---|---|---|
| MFMA/WMMA（矩阵核） | FFMA 循环展开 GEMM（向量单元硬扛） | ✅ DeepGEMM fallback 实证 | ~1/8 吞吐（20.8 vs 预期矩阵核数倍） |
| TMA / cp.async（异步批量拷贝） | 软件流水六条：s_clause 批量 + vmcnt 部分排空 + buffer_load_lds 直通 + 描述符 OOB 填零 + 多 buffer 旋转 + 地址递归链防 spill | ✅ 范式实证（aiter mha pipeline） | 延迟隐藏接近，指令数多 |
| mbarrier（相位/字节级事务） | vmcnt(N) 部分排空 + s_barrier + LDS 旗标组合（aiter pipeline.hpp:381-496 编排实证） | ✅ 可用 | 编排复杂度高 |
| cluster（跨 CTA 组共享） | 单 workgroup 降级；跨 CTA 耦合改 global 原子通信 | 🔶 逐 kernel 评估 | 视耦合度 |
| redux.sync 单指令 | 5 轮 xor-butterfly ds_bpermute | ✅ 已实现 | O(log32) LDS 往返 |
| fp8 硬件转换 | 软件位操作解码（hip_fp8.h / 手工 e4m3） | ✅ DeepGEMM 实证 | 解码开销计入主循环 |
| 硬件 swizzle | 软件 padding（272/144B 行距）+ v_perm 转置 | ✅ aiter mha 实证 | LDS 容量换 bank conflict |
| bf16 原生 FMA | f32 转换路径（或选 fp16 变体） | ✅ 实测 | ~1/2，dtype 杠杆优先 fp16 |

另见：atomics 语义/scope 映射（第四节）、idp4a 硬件 dot4（第五节）——均为"缺失→映射/回退"已闭环实例。
