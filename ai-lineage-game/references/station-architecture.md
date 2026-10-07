# 站模式架构详解（首战实录）

> 首战：AI 溯源游戏（2026-09-07 立项），站0=M-P神经元1943。正本 `/home/zhuo/workspace/ai-lineage/`（ZCode 本地）；本实录书写时项目尚在远程镜像 `/root/workspace/ai-lineage/`。

## 首战目录结构（可复制改造）

```
ai-lineage/
├── README.md          章程：目的/七站路线图表/每站三件套说明/目录结构/记分规则/当前进度
├── ledger.md          记分册：游戏阶段断言对账表（断言/裁决/证据）+ 学习阶段踩坑表（站/事件/备注）
├── papers/            英文PDF + 中译本md + OCR纯文本txt
└── station-0-mp-neuron/
    ├── STATION.md     阅读顺序 + 四关任务（易→难）+ 费曼门三问
    ├── mp_neuron.py   骨架：class MPNeuron + DelayNeuron + or/and/and_not_gate + heat_illusion + build_all_logic
    ├── mp_solution.py 参考实现
    └── test_mp.py     分关验收（关1神经元8测/关2元件8测/关3错觉5测/关4逻辑门3测）
```

## 骨架设计要点（首战实测有效）

- 每个 TODO 注释里给**论文公式原文对照**（如"图 1d: N3(t) = N1(t-1) AND NOT N2(t-1)"），用户不用翻论文就能写
- 时序元件用 class（状态存对象里，step() 一拍一拍推进），静态门用函数——设计任务书时点明这个区别，它就是"神经活动"与"逻辑函数"的差别
- 分关递进：单神经元 → 逻辑元件 → 论文具体案例 → 综合应用（埋下一站种子）
- 测试报错信息带关卡号：`(关1未完成)`，用户一眼知道进度

## 首战费曼门三问（站0）

1. 为什么"全或无"这一条生理假设，是整个逻辑大厦的第一块砖？
2. 热错觉 N3 公式里 `AND NOT N2(t-2)` 那个抑制项在模拟什么？
3. M-P 网络和普通数字电路本质区别是什么？（为什么说"几乎没有"）

设计原则：问"为什么"和"意味着什么"，不问"是什么"——是什么翻书就有。

## 首战 ledger 记录格式

游戏阶段表：`| # | 断言 | 裁决 | 证据 |`（证据注明检索来源+藏书行号）
学习阶段表：`| 站 | 事件 | 备注 |`——坑1时序窗口截短、坑2断言加码（AGENT错票）、验收结果，全如实记。

## 双跑验证脚本模式（/tmp 临时文件）

```python
import shutil, subprocess, sys, os
base = '<station-dir>'
# 1) solution 覆盖跑测试
shutil.copy(f'{base}/mp_neuron.py', f'{base}/mp_neuron.py.bak')
shutil.copy(f'{base}/mp_solution.py', f'{base}/mp_neuron.py')
r1 = subprocess.run([sys.executable, f'{base}/test_mp.py'], capture_output=True, text=True)
shutil.copy(f'{base}/mp_neuron.py.bak', f'{base}/mp_neuron.py')
os.remove(f'{base}/mp_neuron.py.bak')
# 2) skeleton 原样跑 → 应报"未开工"
r2 = subprocess.run([sys.executable, f'{base}/test_mp.py'], capture_output=True, text=True)
# 断言: r1 全绿 + r2 含"骨架尚未开工"
```

注意：对 PASS 数做字符串包含断言时，写实际数字（如 '通过 24 / 共 24'），别写死旧数——首战验证脚本写死 21 导致 solution 实际全绿却报 FAIL，白疑惑一轮。

## 时序电路测试的正确写法

```python
def run(n2_seq, extra=3):
    """刺激结束后多跑几拍——含 N(t-Δ) 的输出在延迟后才浮现"""
    total = len(n2_seq) + extra
    for t in range(total):
        hist.append(n2_seq[t] if t < len(n2_seq) else 0)
        ...
```

## 首战矛盾热觉案例（断言加码错票）

论文原文承诺："若贴得较久，则只有冷的感觉，没有任何先行的暖意"——只约束**刺激期间**。
首战 AGENT 写成"长冷刺激全程无热觉"。实际模型（N3(t)=N2(t-3)·¬N2(t-2)）在刺激移除后 2 拍冒出热觉闪现——这是矛盾热觉（paradoxical heat），真实生理现象。
教训：**断言边界=论文句子边界**。修复时在测试注释里引论文原话，说明为什么新断言才是忠实翻译。

## 中译本首战规格（约3万字规格样本）

书库 `/root/workspace/books/ai/1943-神经活动内在思想的逻辑演算-中译本.md`：
- 头部：原文刊页信息 + 译注体例说明（〔译注〕/【OCR注】）
- §0 导读：阅读路线（必读/选读/永远跳过）+ 一句话版本 + 与 LeCun 自传行号接口
- 摘要→引言→各节全文翻译，公式受损处复原意图不逐字假译
- 附录A 术语对照表（19条：原文/中译/一句话解释）
- 附录B 阅读路线与自测五问
- 附录C 学科史定位图（ASCII 时间线，1936→2012+）
