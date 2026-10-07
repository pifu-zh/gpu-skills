#!/usr/bin/env python3
"""GPU 物理层基线探针（零依赖）。Phase 0/1：理论规格 + 环境匹配 + 可选带宽实测。

用法:
  gpu_probe.py                     # ROCm 探针（当前唯一已实现平台）
  gpu_probe.py --bench PATH        # 附加运行带宽微基准（如 rocm-examples 的 hip_bandwidth）
  gpu_probe.py --bw GBPS           # 显存理论带宽规格（用于达标率计算，查卡 spec 填入）
  gpu_probe.py --platform cuda     # 打印其它平台的手动步骤（探针待实现）
"""
import argparse
import grp
import json
import os
import re
import subprocess
import sys


def run(cmd, timeout=30):
    try:
        r = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
        return r.returncode, r.stdout + (r.stderr if r.returncode else "")
    except (OSError, subprocess.TimeoutExpired) as e:
        return 127, str(e)


def parse_value(text, key):
    """抓 rocminfo 风格的 '  Key:  Value' 行。"""
    m = re.search(rf"^\s*{re.escape(key)}\s*:\s*(.+?)\s*$", text, re.M)
    return m.group(1).strip() if m else None


def probe_rocm():
    # --- rocminfo: GPU agent 段 ---
    rc, info = run(["rocminfo"], timeout=60)
    if rc != 0 or "Marketing Name" not in info:
        print("✗ rocminfo 不可用或无 GPU agent —— 环境不匹配，先修环境再谈测量", file=sys.stderr)
        return 1
    agents = re.split(r"\*{5,}\s*Agent\s+\d+\s*\*{5,}", info)
    gpu = next((a for a in agents if re.search(r"Device Type:\s*GPU", a)), None)
    if gpu is None:
        print("✗ rocminfo 里没有 Device Type: GPU 的 agent —— 环境不匹配，先修环境再谈测量", file=sys.stderr)
        return 1

    g = lambda k: parse_value(gpu, k)
    if not (g("Name") or "").strip().lower().startswith("gfx"):
        print(f"✗ 解析到的 agent 架构可疑: {g('Name')!r} —— 字段格式变了，检查 rocminfo 输出", file=sys.stderr)
        return 1
    name = g("Marketing Name") or "?"
    arch = (g("Name") or "?").split()[0]
    chip = g("Chip ID") or "?"
    cus = int(re.match(r"\d+", g("Compute Unit") or "0").group() or 0)
    simd_per_cu = int(re.match(r"\d+", g("SIMDs per CU") or "0").group() or 0)
    wave = int(re.match(r"\d+", g("Wavefront Size") or "0").group() or 0)
    max_clk_mhz = float(re.match(r"[\d.]+", g("Max Clock Freq. (MHz)") or "0").group() or 0)
    l1 = g("L1") or "?"; l2 = g("L2") or "?"; l3 = g("L3") or "?"
    max_waves = g("Max Waves Per CU") or "?"

    # --- rocm-smi JSON ---
    _, js = run(["rocm-smi", "--showclocks", "--json"])
    clocks = {}
    try:
        card = next(iter(json.loads(js).values()))
        clocks = {k: v for k, v in card.items()}
    except Exception:
        pass
    sclk = next((v for k, v in clocks.items() if k.startswith("sclk")), "?")
    pcie = next((v for k, v in clocks.items() if "pcie" in k.lower()), "?")

    _, js = run(["rocm-smi", "--showmeminfo", "vram", "--json"])
    vram_total = vram_used = None
    try:
        card = next(iter(json.loads(js).values()))
        vram_total = int(card.get("VRAM Total Memory (B)", 0)) or None
        vram_used = int(card.get("VRAM Total Used Memory (B)", 0)) or None
    except Exception:
        pass

    # --- 环境匹配 ---
    drv = open("/sys/module/amdgpu/version").read().strip() if os.path.exists("/sys/module/amdgpu/version") else "?"
    kfd_ok = os.access("/dev/kfd", os.R_OK | os.W_OK) if os.path.exists("/dev/kfd") else False
    try:
        groups_now = {grp.getgrgid(gid).gr_name for gid in os.getgroups()}
    except Exception:
        groups_now = set()
    grp_ok = bool(groups_now & {"video", "render"})
    rc_pmc, out_pmc = run(["rocprofv3-avail", "list", "--pmc"], timeout=60)
    pmc_n = len([ln for ln in out_pmc.splitlines() if re.match(r"^\S+\s", ln)]) if rc_pmc == 0 else 0

    # --- 打印 ---
    GiB = vram_total / 2**30 if vram_total else 0
    fp32_tflops = 2 * cus * 64 * (max_clk_mhz / 1000) / 1000 if cus and max_clk_mhz else 0
    clean = lambda s, unit="": re.match(rf"[\d.]+\s*{unit}", s or "").group().strip() if re.match(rf"[\d.]+\s*{unit}", s or "") else (s or "?")
    sclk_clean = re.search(r"[\d.]+Mhz", sclk or "")
    print("=" * 56)
    print("GPU 物理层基线探针 (ROCm)")
    print("=" * 56)
    print(f"[设备]")
    print(f"  名称        {name}  ({arch}, {chip})")
    print(f"  计算单元    {cus} CU × {simd_per_cu} SIMD, wavefront {wave}, Max Waves/CU {clean(max_waves)}")
    print(f"  显存        {GiB:.1f} GiB" + (f" (已用 {vram_used/2**20:.0f} MiB)" if vram_used else ""))
    print(f"  缓存        L1 {clean(l1)}KB / L2 {clean(l2)}KB / L3 {clean(l3)}KB")
    print(f"  时钟        max {max_clk_mhz:.0f} MHz, 当前 sclk {sclk_clean.group() if sclk_clean else sclk}")
    print(f"  互联        PCIe {pcie}")
    print(f"[理论峰值]")
    print(f"  FP32        ≈ {fp32_tflops:.1f} TFLOPS  (2×{cus}×64×{max_clk_mhz/1000:.2f}GHz, 按 Max Clock)")
    if args_bw:
        print(f"  显存带宽    {args_bw} GB/s (规格值)")
    else:
        print(f"  显存带宽    未提供 —— 用 --bw GBPS 填入卡规格值（查卡 spec）")
    print(f"[环境匹配]")
    ok = lambda b: "✓" if b else "✗"
    print(f"  {ok(drv != '?')} amdgpu 驱动  {drv}")
    rt = re.search(r"Runtime Version:\s*(\S+)", info)
    print(f"  {ok(bool(rt))} HSA runtime {rt.group(1) if rt else '?'}")
    print(f"  {ok(kfd_ok)} /dev/kfd 访问；{ok(grp_ok)} video/render 组 {sorted(groups_now & {'video','render'})}")
    print(f"  {ok(pmc_n > 0)} 计数器通道 rocprofv3 --pmc ({pmc_n} 个可用; 清单: rocprofv3-avail list --pmc)")
    print(f"  提示        rocprof-compute 对消费卡支持有限，用前先跑一次确认；短负载 DVFS 见坑表")
    return 0


def run_bench(path):
    print("\n[实测基线] 带宽微基准: " + path)
    rc, out = run([path], timeout=300)
    lines = [ln for ln in out.splitlines() if "GB/s" in ln or "bandwidth" in ln.lower()]
    if rc != 0:
        print(f"  ✗ 退出码 {rc}: {out.strip()[:200]}")
        return
    for ln in lines or out.splitlines()[-5:]:
        print("  " + ln.strip())
    print("  注: 小缓冲/短测远低于稳态值；正式基线用 ≥256MB 传输 + 多次迭代")


def main():
    global args_bw
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--bench", metavar="PATH", help="带宽微基准可执行文件")
    ap.add_argument("--bw", type=float, metavar="GBPS", help="显存理论带宽规格值")
    ap.add_argument("--platform", default="rocm", choices=["rocm", "cuda", "mojo", "rust"],
                    help="目标平台 (cuda/mojo/rust 探针待实现)")
    args = ap.parse_args()
    args_bw = args.bw

    if args.platform != "rocm":
        print(f"{args.platform} 平台探针待实现。手动步骤见 platforms/{args.platform}.md 对应模块；"
              f"方法论（六阶段/达标率）照 SKILL.md 执行。")
        return 1

    ret = probe_rocm()
    if args.bench:
        run_bench(args.bench)
    return ret


if __name__ == "__main__":
    sys.exit(main())
