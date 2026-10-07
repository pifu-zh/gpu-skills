#!/usr/bin/env python3
"""rocpd / rocprofv3 输出分析报告（零依赖，python3 标准库）。

用法:
  gpu_profile_report.py --csv-dir DIR [--api]   # 分析 rocprofv3 -f csv 的输出目录
  gpu_profile_report.py --db FILE [--api]       # 分析 rocpd *_results.db（含 --pmc 计数器）
"""
import argparse
import csv
import glob
import os
import sqlite3
import sys


def fmt_time(ns):
    ns = float(ns)
    for unit, div in (("s", 1e9), ("ms", 1e6), ("us", 1e3)):
        if ns >= div:
            return f"{ns / div:.3f} {unit}"
    return f"{ns:.0f} ns"


def fmt_bytes(b):
    b = float(b)
    for unit, div in (("GB", 1e9), ("MB", 1e6), ("KB", 1e3)):
        if b >= div:
            return f"{b / div:.2f} {unit}"
    return f"{b:.0f} B"


def demangle(name):
    if name.startswith("_Z"):
        try:
            import subprocess
            r = subprocess.run(["c++filt", name], capture_output=True, text=True, timeout=2)
            if r.returncode == 0 and r.stdout.strip():
                return r.stdout.strip().removesuffix(".kd")
        except Exception:
            pass
    return name


def shorten(name, width=52):
    return name if len(name) <= width else name[: width - 1] + "…"


def print_kernel_table(rows, title):
    """rows: list of dict(name, calls, total_ns, avg_ns, grid, vgpr, sgpr)"""
    print(f"\n== {title} ==")
    if not rows:
        print("(无 kernel dispatch 记录)")
        return
    grand = sum(r["total_ns"] for r in rows)
    rows.sort(key=lambda r: -r["total_ns"])
    hdr = f"{'kernel':<54}{'calls':>7}{'total':>12}{'avg':>12}{'%GPU':>7}{'blocks':>12}{'vgpr':>6}{'sgpr':>6}"
    print(hdr)
    print("-" * len(hdr))
    for r in rows:
        pct = 100.0 * r["total_ns"] / grand if grand else 0.0
        print(
            f"{shorten(r['name']):<54}{r['calls']:>7}{fmt_time(r['total_ns']):>12}"
            f"{fmt_time(r['avg_ns']):>12}{pct:>6.1f}%{r['grid']:>12}{r['vgpr']:>6}{r['sgpr']:>6}"
        )
    print(f"\nGPU kernel 总时间(求和): {fmt_time(grand)}  (多流重叠时会大于墙钟时间)")


def print_copy_table(rows):
    print("\n== Memory Copy ==")
    if not rows:
        print("(无 memcpy 记录)")
        return
    hdr = f"{'kind':<32}{'calls':>7}{'total':>12}{'avg':>12}{'bytes':>12}{'achieved':>14}"
    print(hdr)
    print("-" * len(hdr))
    rows.sort(key=lambda r: -r["total_ns"])
    for r in rows:
        bw = r["bytes"] / r["total_ns"] if r["total_ns"] and r["bytes"] else 0.0  # B/ns = GB/s
        bytes_str = fmt_bytes(r["bytes"]) if r["bytes"] else "-"
        bw_str = f"{bw:>10.1f} GB/s" if bw else "           -"
        print(
            f"{shorten(r['kind'], 32):<32}{r['calls']:>7}{fmt_time(r['total_ns']):>12}"
            f"{fmt_time(r['total_ns'] / r['calls']):>12}{bytes_str:>12}{bw_str}"
        )


def print_api_table(rows, top=12):
    print("\n== HIP API Top ==")
    if not rows:
        print("(无 API 记录)")
        return
    rows.sort(key=lambda r: -r["total_ns"])
    for r in rows[:top]:
        print(f"{shorten(r['name'], 44):<46}{r['calls']:>7}{fmt_time(r['total_ns']):>12}{fmt_time(r['avg_ns']):>12}")


# ---------- CSV 模式 ----------

def load_csv_dir(d):
    kernels, copies, apis = {}, {}, {}
    for path in glob.glob(os.path.join(d, "*_kernel_trace.csv")):
        with open(path, newline="") as f:
            for row in csv.DictReader(f):
                kind = row.get("Kind", "KERNEL_DISPATCH")
                try:
                    dur = int(row["End_Timestamp"]) - int(row["Start_Timestamp"])
                except (TypeError, ValueError):
                    continue
                if "MEMORY_COPY" in kind:
                    key = row.get("Name") or kind
                    c = copies.setdefault(key, {"kind": key, "calls": 0, "total_ns": 0, "bytes": 0})
                    c["calls"] += 1
                    c["total_ns"] += dur
                    c["bytes"] += int(row.get("Size") or row.get("Bytes") or 0)
                else:
                    name = row.get("Kernel_Name") or "unknown"
                    k = kernels.setdefault(name, {"name": name, "calls": 0, "total_ns": 0, "grid": 0, "vgpr": 0, "sgpr": 0})
                    k["calls"] += 1
                    k["total_ns"] += dur
                    # CSV 的 Grid_Size_* 是总线程数(gridDim×blockDim), 除以 workgroup 才是 blocks 数
                    try:
                        wg = int(row.get("Workgroup_Size_X") or 1) or 1
                        k["grid"] += int(row.get("Grid_Size_X") or 0) // wg
                    except ValueError:
                        pass
                    for src, dst in (("VGPR_Count", "vgpr"), ("SGPR_Count", "sgpr")):
                        try:
                            k[dst] += int(row.get(src) or 0)
                        except ValueError:
                            pass
    # memory copy 明细在独立文件，且没有 Size 列（bytes 由 db 模式或程序语义提供）
    for path in glob.glob(os.path.join(d, "*_memory_copy_trace.csv")):
        with open(path, newline="") as f:
            for row in csv.DictReader(f):
                try:
                    dur = int(row["End_Timestamp"]) - int(row["Start_Timestamp"])
                except (TypeError, ValueError):
                    continue
                key = row.get("Direction") or "MEMORY_COPY"
                c = copies.setdefault(key, {"kind": key, "calls": 0, "total_ns": 0, "bytes": 0})
                c["calls"] += 1
                c["total_ns"] += dur
    for path in glob.glob(os.path.join(d, "*_hip_api_trace.csv")):
        with open(path, newline="") as f:
            for row in csv.DictReader(f):
                name = row.get("Function") or "unknown"
                try:
                    dur = int(row["End_Timestamp"]) - int(row["Start_Timestamp"])
                except (TypeError, ValueError):
                    continue
                a = apis.setdefault(name, {"name": name, "calls": 0, "total_ns": 0})
                a["calls"] += 1
                a["total_ns"] += dur
    kr = [dict(r, avg_ns=r["total_ns"] / r["calls"], grid=r["grid"] // max(r["calls"], 1),
               vgpr=r["vgpr"] // max(r["calls"], 1), sgpr=r["sgpr"] // max(r["calls"], 1))
          for r in kernels.values()]
    cp = [dict(r) for r in copies.values()]
    ap = [dict(r, avg_ns=r["total_ns"] / r["calls"]) for r in apis.values()]
    return kr, cp, ap


# ---------- rocpd db 模式 ----------

def load_db(path):
    db = sqlite3.connect(f"file:{path}?mode=ro", uri=True)
    tables = {r[0] for r in db.execute("SELECT name FROM sqlite_master WHERE type='table'")}

    def one(prefix):
        hits = sorted(t for t in tables if t.startswith(prefix))
        if not hits:
            raise SystemExit(f"db 缺少 {prefix}* 表，不是 rocpd 数据库? {path}")
        return hits[0]

    sym, disp = one("rocpd_info_kernel_symbol"), one("rocpd_kernel_dispatch")
    kernels = {}
    # grid_size_x 与 CSV 同语义(总线程数), 除以 workgroup_size_x 得 blocks
    q = f"""SELECT s.kernel_name, COUNT(*), SUM(d.end - d.start),
                   AVG(d.grid_size_x / MAX(d.workgroup_size_x, 1)),
                   AVG(s.arch_vgpr_count), AVG(s.sgpr_count)
            FROM {disp} d JOIN {sym} s ON d.kernel_id = s.id GROUP BY s.kernel_name"""
    for name, calls, total, grid, vgpr, sgpr in db.execute(q):
        name = demangle(name)
        kernels[name] = {"name": name, "calls": calls, "total_ns": total, "avg_ns": total / calls,
                         "grid": int(grid or 0), "vgpr": int(vgpr or 0), "sgpr": int(sgpr or 0)}
    copies = {}
    mc = one("rocpd_memory_copy")
    q = f"""SELECT COALESCE((SELECT string FROM {one('rocpd_string')} WHERE id = m.name_id), 'memcpy'),
                   COUNT(*), SUM(m.end - m.start), SUM(m.size)
            FROM {mc} m GROUP BY 1"""
    for kind, calls, total, size in db.execute(q):
        copies[kind] = {"kind": kind, "calls": calls, "total_ns": total or 0, "bytes": size or 0}
    # ROCm 7.14 rocpd 默认不含 HIP API 明细；要 API 数据用 -f csv 采集
    kr = list(kernels.values())
    cp = list(copies.values())
    pmc = load_pmc(db, one, disp)
    return kr, cp, [], pmc


def load_pmc(db, one, disp):
    tables = {r[0] for r in db.execute("SELECT name FROM sqlite_master WHERE type='table'")}
    pmc_tabs = [t for t in tables if t.startswith("rocpd_pmc_event")]
    if not pmc_tabs:
        return None
    info, ev = one("rocpd_info_pmc"), pmc_tabs[0]
    q = f"""SELECT s.kernel_name, p.name, SUM(v.value), COUNT(*)
            FROM {ev} v
            JOIN {info} p ON v.pmc_id = p.id
            JOIN {disp} d ON v.event_id = d.event_id
            JOIN {one('rocpd_info_kernel_symbol')} s ON d.kernel_id = s.id
            GROUP BY s.kernel_name, p.name ORDER BY s.kernel_name, p.name"""
    out = {}
    try:
        for kname, metric, total, calls in db.execute(q):
            out.setdefault(demangle(kname), []).append((metric, total, calls))
    except Exception as e:
        print(f"(PMC join 失败: {e})", file=sys.stderr)
        return None
    return out or None


def print_pmc(pmc):
    if not pmc:
        return
    print("\n== PMC 计数器（按 kernel）==")
    for kname, metrics in pmc.items():
        print(f"\n[{shorten(kname, 60)}]")
        for metric, total, calls in metrics:
            avg = total / calls if calls else 0
            print(f"  {metric:<28} total={total:>18,.0f}  calls={calls:>6}  avg={avg:,.1f}")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--csv-dir", help="rocprofv3 -f csv 输出目录")
    g.add_argument("--db", help="rocpd *_results.db 文件")
    ap.add_argument("--api", action="store_true", help="显示 HIP API 耗时 top")
    args = ap.parse_args()

    if args.csv_dir:
        kr, cp, apilist = load_csv_dir(args.csv_dir)
        pmc = None
    else:
        kr, cp, apilist, pmc = load_db(args.db)

    print_kernel_table(kr, "Kernel 汇总（按 GPU 时间降序）")
    print_copy_table(cp)
    if args.api:
        print_api_table(apilist)
    print_pmc(pmc)
    print()


if __name__ == "__main__":
    main()
