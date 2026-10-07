---
name: ai-lineage-game
description: Use when 用户推进 AI 溯源游戏（ai-lineage）——开新站/写任务书/出骨架与验收测试/判费曼门/记 ledger，或说"从地基开始学X""沿X的发展史学X"立项新溯源项目. 站式溯源学习法（ZCode 正本，迁自 Hermes lineage-learning v1.0.0）.
version: 1.1.0
author: zhuo + ZCode
tags: [learning, methodology, history, papers, reimplementation]
---

# 溯源学习法 · AI 溯源游戏（ZCode 正本）

> 谱系：2026-09-07 Hermes 于远程立项并完成首战（站 0 M-P 神经元，参考实现 24/24 全绿）；2026-09-08 ZCode 盲区审计将路线图扩编为 v2，正本迁至本地。首战完整实录见 `references/station-architecture.md`。

## 正本与镜像

- **正本**：`/home/zhuo/workspace/ai-lineage/`（主阵地 = ZCode + 本地 Linux）
- **镜像**：`/root/workspace/ai-lineage/`（阿里云 `ssh root@120.55.67.84`，Hermes 宿主）
- 同步：`rsync -avz --exclude='__pycache__' src/ dst/`；**推送到远程后必须 chown root:root 改过的文件**（本机 uid 1000 会被 -a 带过去）
- 文献书库：`~/workspace/books/ai/`（项目内 `papers` 软链指向它，单一副本）；盲区审计与证据链：项目内 `audit-20260907.md`

## 行动前必读

开任何动作前先读正本的 `README.md`（路线图 v2 + 当前进度）和 `ledger.md`（错票与踩坑）。当前：站 0（M-P 神经元）进行中，等 USER 填 `station-0-mp-neuron/mp_neuron.py` → `python3 test_mp.py`。

## 游戏规则（对 AGENT 同样生效）

- **每站三件套**：① 原文（PDF + 必要时中译 + OCR 纯文本，三件同存书库）② 导读（阅读路线：必读/选读/永远跳过 + 与前站勾连 + 与用户藏书互证，如 LeCun 自传行号）③ 复现代码：`skeleton`（函数签名 + 详细 TODO + 论文公式对照，USER 亲手填；**卡 30 分钟才许看 solution，看了要说**）+ `solution`（参考实现）+ `tests`（分关验收，易→难，报错信息带关卡号）
- **过站双门**：测试全绿 + 费曼门（STATION.md 末尾三问，问"为什么/意味着什么"不问"是什么"，答不上不算过站）
- **记分**：断言须有出处（文献页码/实测输出/书证行号），无出处记错票入 ledger；AGENT 断言写错自己记（先例：ledger #4——查证前先 grep 自家书架，别把在手的一手书证标成"未检索"）

## 测试设计两坑（首战实测）

1. **时序窗口截短**：含 `N(t-Δ)` 的公式输出在 Δ+1 拍后才浮现；模拟必须跑足**延迟+1 拍**，否则断言误报（例：热觉公式含 N2(t−3)，冷 1 拍刺激的热觉 t=3 才出现，跑到 t=2 就断言→误报）
2. **断言 = 论文文本，不得加码**：首战把"刺激期间无热觉"写成"全程无热觉"，实际模型在刺激移除后 2 拍冒出热觉闪现——**矛盾热觉（paradoxical heat）**，真实生理现象。修正法：逐句对照论文原文划断言边界，测试注释里引论文原话

## AGENT 交付前双跑验证

1. solution 覆盖 skeleton 跑测试 → 必须全绿
2. 恢复 skeleton 再跑 → 必须按预期报"未开工"（证明骨架真空着，不是坏的）
- 验证脚本写 /tmp 临时文件执行，勿用多行 python3 -c 内联；对 PASS 数做断言时写实际数字（如 `'通过 24 / 共 24'`），勿写死旧数

## 论文获取与中译流水线

检索原文 → 大学课程页直链（CMU 等常挂经典 PDF）→ curl 下载 → `file` 验证真 PDF；pymupdf/pypdf 提取（写临时脚本）；中译规范：导读先行 + 一句话版本，〔译注〕=译者解释，【OCR注】=扫描件公式受损处**复原意图不逐字假译，诚实标注**——宁缺毋编。

## 埋种子的艺术

每站埋下一站钩子：先让用户亲手碰壁（站 0 用 M-P 单元写 XOR 必败），下一站解释这堵墙（站 2 明斯基《感知器》）。失败体验先于理论解释。

## v2 路线图速查（以 README.md 为准）

图灵(-1) → M-P(0🔨) → 赫布(0.5) → 感知机(1) → XOR寒冬(2) → backprop+Kaiming init 附加关(3) → Hopfield(3.5) → LeNet(4) ∥ SVM对打(C1) → LSTM(5) ∥ TD/Q-learning(R1) → Attention(6) → DBN+AlexNet(6.5a) → 何恺明 init/ResNet/MAE(6.5b) → scaling+RLHF(6¾) ∥ Lighthill+万能逼近(附录) → 闭环：M-P 图灵段 + 孩子机器(7)。副站随相邻主线站开站，支线与附录随时可开；新站目录命名如 `station-3.5-hopfield`、`station-6.5b-kaiming`。
