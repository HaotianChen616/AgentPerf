#!/usr/bin/env python3
"""
Result Analyzer & Visualizer for CPU Oversubscription Tests

Supports both agent-based (oversub_agent_*.json) and simulation-based
(oversub_*.json) result formats. Generates ASCII visualizations:

  - Task latency bar chart (grouped by type)
  - CPU utilization sparkline timeline
  - Turn-level latency heatmap
  - Token consumption breakdown
  - Workload type comparison
"""

import sys
import os
import json
import glob
import statistics
import argparse
from typing import List, Dict, Optional
from datetime import datetime

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
RESULTS_DIR = os.path.join(BASE_DIR, "results")

COLORS = {
    "cpu": "\033[91m",
    "io": "\033[92m",
    "network": "\033[93m",
    "mixed": "\033[96m",
    "memory": "\033[95m",
    "llm": "\033[35m",
    "tool": "\033[33m",
    "framework": "\033[34m",
    "reset": "\033[0m",
    "bold": "\033[1m",
    "dim": "\033[2m",
}

BAR_CHARS = "█▓▒░"
BLOCK_CHARS = "▁▂▃▄▅▆▇█"


def c(name: str) -> str:
    return COLORS.get(name, "")


def bar(value: float, max_val: float, width: int = 40, char: str = "█") -> str:
    if max_val <= 0:
        return char * 0 + "░" * width
    filled = min(int((value / max_val) * width), width)
    return char * filled + "░" * (width - filled)


def sparkline(values: List[float], width: int = 70) -> str:
    if not values:
        return ""
    mn, mx = min(values), max(values)
    rng = mx - mn if mx > mn else 1
    step = max(1, len(values) // width)
    sampled = values[::step][:width]
    return "".join(
        BLOCK_CHARS[min(int((v - mn) / rng * (len(BLOCK_CHARS) - 1)), len(BLOCK_CHARS) - 1)]
        for v in sampled
    )


def sep(char: str = "═", width: int = 78):
    print(f"  {char * width}")


def detect_format(data: dict) -> str:
    instances = data.get("per_instance", [])
    if not instances:
        return "unknown"
    first = instances[0]
    if "total_latency_ms" in first and "total_prompt_tokens" in first:
        return "agent"
    if "latency_ms" in first:
        return "simulation"
    return "unknown"


# ---------------------------------------------------------------------------
# Agent format analysis
# ---------------------------------------------------------------------------

def analyze_agent(filepath: str):
    with open(filepath, "r") as f:
        data = json.load(f)

    config = data.get("test_config", {})
    glob_data = data.get("global", {})
    instances = data.get("per_instance", [])

    print()
    sep("═")
    print(f"  {c('bold')}📊 CPU OVERSUBSCRIPTION TEST REPORT — nanobot-ai Agent{c('reset')}")
    sep("═")
    print(f"  File       : {os.path.basename(filepath)}")
    print(f"  Timestamp  : {config.get('timestamp', 'N/A')}")
    print(f"  Model      : {config.get('model', '?')} ({config.get('provider', '?')})")
    print(f"  Core       : {config.get('core_id', '?')}")
    print(f"  Concurrency: {config.get('concurrency', '?')}")
    print(f"  Duration   : {config.get('duration_sec', '?')}s")
    print(f"  Instances  : {config.get('num_instances', '?')}")
    types = config.get("task_types", [])
    tpt = config.get("tasks_per_type", 0)
    print(f"  Types      : {', '.join(types)} ({tpt} each)")
    sep("─")

    # --- CPU utilization sparkline ---
    cpu_samples = glob_data.get("cpu_samples", [])
    if cpu_samples:
        print(f"\n  {c('bold')}▌ CPU Utilization Timeline (Core {config.get('core_id', 0)}){c('reset')}")
        print(f"    Avg: {c('bold')}{statistics.mean(cpu_samples):.1f}%{c('reset')}  "
              f"Max: {c('bold')}{max(cpu_samples):.1f}%{c('reset')}  "
              f"Min: {min(cpu_samples):.1f}%  "
              f"StdDev: {statistics.stdev(cpu_samples):.1f}%  "
              f"Samples: {len(cpu_samples)}")
        print(f"    {sparkline(cpu_samples, 70)}")
        print(f"    {c('dim')}{'0%':<35}{'100%':>35}{c('reset')}")

    # --- Process / Thread Overhead ---
    has_proc_metrics = any(
        inst.get("num_threads") or inst.get("ctx_switches_vol")
        for inst in instances
    )
    glb_ctx = glob_data if isinstance(glob_data, dict) else {}
    if has_proc_metrics or glb_ctx.get("total_processes"):
        print(f"\n  {c('bold')}▌ Process / Thread Overhead{c('reset')}")
        total_procs = glb_ctx.get("total_processes", len(instances))
        total_threads = glb_ctx.get("total_threads", 0)
        if not total_threads:
            total_threads = sum(i.get("num_threads", 0) for i in instances
                               if isinstance(i.get("num_threads"), int))
        total_vol = glb_ctx.get("total_ctx_switches_vol", 0)
        total_invol = glb_ctx.get("total_ctx_switches_invol", 0)
        if not total_vol:
            total_vol = sum(i.get("ctx_switches_vol", 0) for i in instances)
        if not total_invol:
            total_invol = sum(i.get("ctx_switches_invol", 0) for i in instances)
        cxs_per_sec = glb_ctx.get("ctx_switches_per_sec",
                                   (total_vol + total_invol) / max(1, config.get("duration_sec", 1)))
        mode = config.get("mode", "coro")
        print(f"    Mode               : {c('bold')}{mode}{c('reset')}")
        print(f"    Total Processes    : {total_procs}")
        print(f"    Total Threads      : {total_threads}")
        print(f"    Avg Threads/Proc   : {total_threads/max(1,total_procs):.1f}")
        print(f"    Ctx Switch (vol)   : {total_vol:,}")
        print(f"    Ctx Switch (invol) : {total_invol:,}")
        print(f"    Ctx Switch Total   : {c('bold')}{total_vol+total_invol:,}{c('reset')} "
              f"({cxs_per_sec:.0f}/sec)")
        if total_threads > 0:
            thread_bar_w = 40
            max_thr = max((i.get("num_threads", 0) for i in instances
                          if isinstance(i.get("num_threads"), int)), default=1)
            print(f"\n    Threads per instance:")
            for inst in sorted(instances, key=lambda x: x.get("instance_id", 0)):
                iid = inst.get("instance_id", "?")
                wtype = inst.get("workload_type", "?")
                nthr = inst.get("num_threads", 0)
                if not isinstance(nthr, int):
                    continue
                bar_n = min(int(nthr / max(max_thr, 1) * thread_bar_w), thread_bar_w)
                color = c(wtype)
                blk = chr(9608) * bar_n
                lgt = chr(9617) * (thread_bar_w - bar_n)
                print(f"      {color}#{iid:2d} {wtype:<8}{c('reset')} "
                      f"{blk}{lgt} {nthr}")
    # --- Cache Performance ---
    glb_cache = glb_ctx.get("cache_perf", {}) if isinstance(glb_ctx, dict) else {}
    if glb_cache:
        cr = glb_cache.get("cache-references", 0)
        cm = glb_cache.get("cache-misses", 0)
        l1l = glb_cache.get("L1-dcache-loads", 0)
        l1m = glb_cache.get("L1-dcache-load-misses", 0)
        l2a = glb_cache.get("armv8_pmuv3_0/l2d_cache/", 0)
        l2r = glb_cache.get("armv8_pmuv3_0/l2d_cache_refill/", 0)
        bus = glb_cache.get("armv8_pmuv3_0/bus_access/", 0)
        mem = glb_cache.get("armv8_pmuv3_0/mem_access/", 0)

        print(f"\n  {c('bold')}▌ Cache Performance{c('reset')}")
        cache_miss_rate = cm / max(1, cr) * 100
        l1_miss_rate = l1m / max(1, l1l) * 100
        l2_miss_rate = l2r / max(1, l2a) * 100

        print(f"    {'Event':<25} {'Count':>15} {'Miss Rate':>10}")
        print(f"    {'─'*25} {'─'*15} {'─'*10}")
        print(f"    {'Cache References':<25} {cr:>15,}")
        print(f"    {'Cache Misses':<25} {cm:>15,} {cache_miss_rate:>9.2f}%")
        print(f"    {'L1-dcache Loads':<25} {l1l:>15,}")
        print(f"    {'L1-dcache Misses':<25} {l1m:>15,} {l1_miss_rate:>9.2f}%")
        print(f"    {'L2 Cache Accesses':<25} {l2a:>15,}")
        print(f"    {'L2 Cache Refills (miss)':<25} {l2r:>15,} {l2_miss_rate:>9.2f}%")
        print(f"    {'Bus Accesses':<25} {bus:>15,}")
        print(f"    {'Memory Accesses':<25} {mem:>15,}")

        bar_w = 40
        hit_n = min(int((100 - cache_miss_rate) / 100 * bar_w), bar_w)
        miss_n = bar_w - hit_n
        print(f"\n    Overall Cache: {chr(9608)*hit_n}{chr(9617)*miss_n}  Hit/Miss ({100-cache_miss_rate:.1f}%/{cache_miss_rate:.1f}%)")

        l2_hit_n = min(int((100 - l2_miss_rate) / 100 * bar_w), bar_w)
        l2_miss_n = bar_w - l2_hit_n
        print(f"    L2 Cache:     {chr(9608)*l2_hit_n}{chr(9617)*l2_miss_n}  Hit/Miss ({100-l2_miss_rate:.1f}%/{l2_miss_rate:.1f}%)")

        if cache_miss_rate > 10:
            print(f"\n    {c('bold')}WARNING: Cache miss rate > 10% indicates severe cache contention!{c('reset')}")
            print(f"    Multiple instances competing for L1 ({64}KB) and L2 ({512}KB) per core.")
        elif cache_miss_rate > 5:
            print(f"\n    {c('bold')}Moderate cache contention detected ({cache_miss_rate:.1f}% miss rate){c('reset')}")
            print(f"    L2 ({512}KB) shared by {config.get('concurrency', 'N/A')} processes may cause evictions.")
        else:
            print(f"\n    Cache contention: LOW ({cache_miss_rate:.1f}% miss rate)")

    # --- Memory Usage ---
    has_mem = any(
        isinstance(inst.get("rss_peak_mb"), (int, float)) and inst.get("rss_peak_mb", 0) > 0
        or isinstance(inst.get("rss_mb"), (int, float)) and inst.get("rss_mb", 0) > 0
        for inst in instances
    )
    if has_mem:
        print(f"\n  {c('bold')}▌ Memory Usage{c('reset')}")

        rss_data = []
        for inst in sorted(instances, key=lambda x: x.get("instance_id", 0)):
            rss = inst.get("rss_peak_mb", inst.get("rss_mb", 0))
            vms = inst.get("vms_peak_mb", inst.get("vms_mb", 0))
            if isinstance(rss, (int, float)) and rss > 0:
                rss_data.append((inst, rss, vms))

        if rss_data:
            max_rss = max(r for _, r, _ in rss_data)
            bar_w = 40

            print(f"    {'ID':>3} {'Type':<8} {'RSS_peak':>8} {'RSS_avg':>8} {'VMS_peak':>8}  Bar")
            print(f"    {'─'*3} {'─'*8} {'─'*8} {'─'*8} {'─'*8}  {'─'*bar_w}")

            for inst, rss, vms in rss_data:
                iid = inst.get("instance_id", "?")
                wtype = inst.get("workload_type", "?")
                rss_avg = inst.get("rss_avg_mb", 0)
                color = c(wtype)
                bar_n = min(int(rss / max(max_rss, 1) * bar_w), bar_w)
                blk = chr(9608) * bar_n
                lgt = chr(9617) * (bar_w - bar_n)
                print(f"    {color}{iid:3d} {wtype:<8}{c('reset')} "
                      f"{rss:8.1f} {rss_avg:8.1f} {vms:8.1f}  {blk}{lgt}")

            all_rss = [r for _, r, _ in rss_data]
            all_vms = [v for _, _, v in rss_data if isinstance(v, (int, float))]
            print(f"\n    Total RSS       : {sum(all_rss):.1f} MB")
            print(f"    Avg RSS/Proc    : {statistics.mean(all_rss):.1f} MB")
            print(f"    Max RSS/Proc    : {max(all_rss):.1f} MB")
            if all_vms:
                print(f"    Total VMS       : {sum(all_vms):.1f} MB")
                print(f"    Avg VMS/Proc    : {statistics.mean(all_vms):.1f} MB")

        hot_vals = [i.get("mem_hot_kb", 0) for i in instances if isinstance(i.get("mem_hot_kb"), (int, float))]
        warm_vals = [i.get("mem_warm_kb", 0) for i in instances if isinstance(i.get("mem_warm_kb"), (int, float))]
        cold_vals = [i.get("mem_cold_kb", 0) for i in instances if isinstance(i.get("mem_cold_kb"), (int, float))]
        if hot_vals:
            t_hot = sum(hot_vals)
            t_warm = sum(warm_vals) if warm_vals else 0
            t_cold = sum(cold_vals) if cold_vals else 0
            t_all = t_hot + t_warm + t_cold
            print(f"\n    Memory Hot/Warm/Cold (/proc/PID/smaps_rollup):")
            if t_all == 0:
                print(f"      (all values 0 — sandbox/container may restrict smaps access)")
            else:
                print(f"      Hot  (private_dirty)  : {t_hot:>10,} KB ({t_hot/max(1,t_all)*100:5.1f}%)")
                print(f"      Warm (private_clean)  : {t_warm:>10,} KB ({t_warm/max(1,t_all)*100:5.1f}%)")
                print(f"      Cold (shared_*)       : {t_cold:>10,} KB ({t_cold/max(1,t_all)*100:5.1f}%)")
                hot_bar_w = 35
                h_n = min(int(t_hot / max(1, t_all) * hot_bar_w), hot_bar_w)
                w_n = min(int(t_warm / max(1, t_all) * hot_bar_w), hot_bar_w - h_n)
                c_n = hot_bar_w - h_n - w_n
                print(f"      {chr(9608)*h_n}{chr(9618)*w_n}{chr(9617)*c_n}  Hot/Warm/Cold")

        io_r = [i.get("io_read_bytes", 0) for i in instances if isinstance(i.get("io_read_bytes"), (int, float))]
        io_w = [i.get("io_write_bytes", 0) for i in instances if isinstance(i.get("io_write_bytes"), (int, float))]
        if io_r and sum(io_r) > 0:
            print(f"\n    I/O Bytes:")
            print(f"      Total Read  : {sum(io_r):>12,} bytes ({sum(io_r)/1024/1024:.1f} MB)")
            print(f"      Total Write : {sum(io_w):>12,} bytes ({sum(io_w)/1024/1024:.1f} MB)")

    # --- Per-instance latency bar chart ---
    print(f"\n  {c('bold')}▌ Task Latency by Instance{c('reset')}")
    max_lat = max((i.get("total_latency_ms", 0) for i in instances), default=1)
    max_lat_sec = max_lat / 1000

    for inst in sorted(instances, key=lambda x: x.get("total_latency_ms", 0), reverse=True):
        iid = inst.get("instance_id", "?")
        wtype = inst.get("workload_type", "?")
        lat_ms = inst.get("total_latency_ms", 0)
        lat_s = lat_ms / 1000
        color = c(wtype)
        reset = c("reset")
        b = bar(lat_s, max_lat_sec, 35, "█")
        err = inst.get("error", "")
        status = f"  {c('dim')}ERR: {err[:30]}{reset}" if err else ""
        print(f"    {color}#{iid:2d} {wtype:<8}{reset} │{color}{b}{reset}│ {lat_s:6.1f}s{status}")

    print(f"    {'':>11} {'':>1}{'─' * 35}{'':>1}")
    print(f"    {'':>11} {'':>1}0s{'':>29}{max_lat_sec:.0f}s{'':>2}")

    # --- Per-instance details table ---
    print(f"\n  {c('bold')}▌ Detailed Metrics{c('reset')}")
    print(f"    {'ID':>3} {'Type':<8} {'Time':>7} {'Turns':>5} {'Tools':>5} "
          f"{'PID':>6} {'Thr':>4} {'LLM':>7} {'Tool':>7} {'Fwk':>7}")
    print(f"    {'─'*3} {'─'*8} {'─'*7} {'─'*5} {'─'*5} {'─'*8} {'─'*7} {'─'*8} {'─'*7} {'─'*7} {'─'*7}")

    for inst in sorted(instances, key=lambda x: x.get("instance_id", 0)):
        iid = inst.get("instance_id", "?")
        wtype = inst.get("workload_type", "?")
        lat = inst.get("total_latency_ms", 0) / 1000
        turns = inst.get("num_turns", 0)
        tools = inst.get("total_tool_calls", 0)
        pt = inst.get("total_prompt_tokens", 0)
        ct = inst.get("total_completion_tokens", 0)
        tt = pt + ct
        llm_s = inst.get("total_llm_latency_ms", 0) / 1000
        tool_s = inst.get("total_tool_latency_ms", 0) / 1000
        fwk_s = inst.get("total_framework_latency_ms", 0) / 1000
        pid = inst.get("pid", "-")
        thr = inst.get("num_threads", "-")
        color = c(wtype)
        print(f"    {color}{iid:3d} {wtype:<8}{c('reset')} {lat:7.1f} {turns:5d} {tools:5d} "
              f"{pid:>6} {thr:>4} {llm_s:7.1f} {tool_s:7.1f} {fwk_s:7.1f}")

    # --- Turn-level latency heatmap ---
    print(f"\n  {c('bold')}▌ Turn-Level Latency Heatmap{c('reset')}")
    max_turns = max((len(i.get("turn_latencies_ms", [])) for i in instances), default=0)
    if max_turns > 0:
        all_turn_lats = []
        for inst in instances:
            all_turn_lats.extend(inst.get("turn_latencies_ms", []))
        max_turn_lat = max(all_turn_lats) if all_turn_lats else 1

        header = "    " + " ".join(f"T{t:<5d}" for t in range(max_turns))
        print(header)
        for inst in sorted(instances, key=lambda x: x.get("instance_id", 0)):
            iid = inst.get("instance_id", "?")
            wtype = inst.get("workload_type", "?")
            lats = inst.get("turn_latencies_ms", [])
            color = c(wtype)
            cells = ""
            for t in range(max_turns):
                if t < len(lats):
                    v = lats[t]
                    ratio = v / max_turn_lat if max_turn_lat > 0 else 0
                    idx = min(int(ratio * (len(BLOCK_CHARS) - 1)), len(BLOCK_CHARS) - 1)
                    cells += f"{color}{BLOCK_CHARS[idx]}{c('reset')}     "
                else:
                    cells += "·      "
            print(f"    {color}#{iid:2d}{c('reset')} {cells}")
        print(f"    {c('dim')}Legend: ▁=fast ▄=medium █=slow  (scale: 0-{max_turn_lat/1000:.1f}s){c('reset')}")

    # --- Latency Breakdown: Stacked Bar Chart ---
    has_breakdown = any(
        inst.get("total_llm_latency_ms", 0) > 0 or inst.get("total_tool_latency_ms", 0) > 0
        for inst in instances
    )
    if has_breakdown:
        print(f"\n  {c('bold')}▌ Latency Breakdown — LLM / Tool / Framework{c('reset')}")
    has_init = any(inst.get("init_latency_ms") for inst in instances)
    if has_init:
        print(f"    {c('llm')}██{c('reset')}=LLM  {c('tool')}██{c('reset')}=Tool  {c('framework')}██{c('reset')}=Framework(init)  {c('framework')}██{c('reset')}=Framework(run)")
    else:
        print(f"    {c('llm')}██{c('reset')}=LLM  {c('tool')}██{c('reset')}=Tool  {c('framework')}██{c('reset')}=Framework")
        max_total = max((i.get("total_latency_ms", 0) for i in instances), default=1) / 1000
        bar_width = 45

        for inst in sorted(instances, key=lambda x: x.get("instance_id", 0)):
            iid = inst.get("instance_id", "?")
            wtype = inst.get("workload_type", "?")
            total_s = inst.get("total_latency_ms", 0) / 1000
            llm_s = inst.get("total_llm_latency_ms", 0) / 1000
            tool_s = inst.get("total_tool_latency_ms", 0) / 1000
            fwk_s = inst.get("total_framework_latency_ms", 0) / 1000

            n_llm = min(int((llm_s / max_total) * bar_width), bar_width) if max_total > 0 else 0
            n_tool = min(int((tool_s / max_total) * bar_width), bar_width - n_llm) if max_total > 0 else 0
            n_fwk = min(int((fwk_s / max_total) * bar_width), bar_width - n_llm - n_tool) if max_total > 0 else 0
            n_pad = bar_width - n_llm - n_tool - n_fwk

            stacked = (f"{c('llm')}{'█' * n_llm}{c('reset')}"
                       f"{c('tool')}{'█' * n_tool}{c('reset')}"
                       f"{c('framework')}{'█' * n_fwk}{c('reset')}"
                       f"{'░' * n_pad}")
            color = c(wtype)
            pct_llm = llm_s / total_s * 100 if total_s > 0 else 0
            pct_tool = tool_s / total_s * 100 if total_s > 0 else 0
            print(f"    {color}#{iid:2d} {wtype:<8}{c('reset')} │{stacked}│ "
                  f"{total_s:5.1f}s  {c('llm')}{pct_llm:3.0f}%{c('reset')}/{c('tool')}{pct_tool:3.0f}%{c('reset')}/{c('framework')}{100-pct_llm-pct_tool:3.0f}%{c('reset')}")

        print(f"    {'':>11} {'':>1}{'─' * bar_width}{'':>1}")
        print(f"    {'':>11} {'':>1}0s{'':>{bar_width-3}}{max_total:.0f}s{'':>1}")

    # --- Per-Turn Latency Decomposition Table ---
    has_turns_data = any(
        isinstance(inst.get("turns"), list) and len(inst.get("turns", [])) > 0
        for inst in instances
    )
    if has_turns_data:
        print(f"\n  {c('bold')}▌ Per-Turn Latency Decomposition{c('reset')}")
        for inst in sorted(instances, key=lambda x: x.get("instance_id", 0)):
            iid = inst.get("instance_id", "?")
            wtype = inst.get("workload_type", "?")
            turns = inst.get("turns", [])
            if not turns:
                continue
            color = c(wtype)
            print(f"    {color}#{iid:2d} {wtype:<8}{c('reset')}")
            print(f"      {'Turn':>4} {'Total':>7} {'LLM':>7} {'Tool':>7} {'Fwk':>7} "
                  f"{'LLM%':>5} {'Tool%':>5} {'Tools Used'}")
            print(f"      {'─'*4} {'─'*7} {'─'*7} {'─'*7} {'─'*7} {'─'*5} {'─'*5} {'─'*30}")

            for t in turns:
                t_total = t.get("latency_ms", 0) / 1000
                t_llm = t.get("llm_latency_ms", 0) / 1000
                t_tool = t.get("tool_latency_ms", 0) / 1000
                t_fwk = max(0, t_total - t_llm - t_tool)
                t_names = t.get("tool_names", [])
                names_str = ",".join(t_names) if t_names else "-"
                pct_l = t_llm / t_total * 100 if t_total > 0 else 0
                pct_t = t_tool / t_total * 100 if t_total > 0 else 0
                print(f"      {t.get('turn_index',0):4d} {t_total:7.2f} {c('llm')}{t_llm:7.2f}{c('reset')} "
                      f"{c('tool')}{t_tool:7.2f}{c('reset')} {c('framework')}{t_fwk:7.2f}{c('reset')} "
                      f"{pct_l:5.1f} {pct_t:5.1f} {names_str}")

    # --- Aggregated Latency Breakdown by Type ---
    if has_breakdown:
        by_type_bd: Dict[str, List[Dict]] = {}
        for inst in instances:
            wt = inst.get("workload_type", "unknown")
            by_type_bd.setdefault(wt, []).append(inst)

    # --- Framework Latency Decomposition ---
    has_init = any(isinstance(inst.get("init_latency_ms"), (int, float)) for inst in instances)
    has_gaps = any(inst.get("inter_turn_gaps_ms") for inst in instances)
    if has_init or has_gaps:
        print(f"\n  {c('bold')}▌ Framework Latency Decomposition{c('reset')}")
        print(f"    Framework = Init + Inter-turn gaps (context governance, checkpoint, memory)")
        print()
        print(f"    {'ID':>3} {'Type':<8} {'Init':>7} {'Sum(gaps)':>10} {'Avg(gap)':>9} {'Max(gap)':>9} {'#gaps':>6} {'Fwk_run':>8}")
        print(f"    {'─'*3} {'─'*8} {'─'*7} {'─'*10} {'─'*9} {'─'*9} {'─'*6} {'─'*8}")
        all_init = []
        all_gap_sums = []
        for inst in sorted(instances, key=lambda x: x.get("instance_id", 0)):
            iid = inst.get("instance_id", "?")
            wtype = inst.get("workload_type", "?")
            init_ms = inst.get("init_latency_ms", 0) or 0
            gaps = inst.get("inter_turn_gaps_ms", [])
            if not isinstance(gaps, list):
                gaps = []
            gap_sum = sum(gaps)
            gap_avg = statistics.mean(gaps) if gaps else 0
            gap_max = max(gaps) if gaps else 0
            fwk_run = inst.get("framework_run_ms", 0) or 0
            color = c(wtype)
            print(f"    {color}{iid:3d} {wtype:<8}{c('reset')} "
                  f"{init_ms:7.0f} {gap_sum:10.0f} {gap_avg:9.0f} {gap_max:9.0f} "
                  f"{len(gaps):6d} {fwk_run:8.0f}")
            if isinstance(init_ms, (int, float)) and init_ms > 0:
                all_init.append(init_ms)
            all_gap_sums.append(gap_sum)
        if all_init or all_gap_sums:
            print()
            if all_init:
                print(f"    Avg Init (Nanobot.from_config): {statistics.mean(all_init):.0f}ms")
                print(f"      └─ config loading, provider init, tool registration, session manager")
            if all_gap_sums:
                print(f"    Avg Inter-turn gap sum: {statistics.mean(all_gap_sums):.0f}ms")
                print(f"      ├─ _microcompact()    : compress old tool results")
                print(f"      ├─ _snip_history()    : truncate messages by token estimate")
                print(f"      ├─ estimate_tokens()  : tiktoken encoding for context window")
                print(f"      ├─ _emit_checkpoint() : serialize session to disk (JSONL)")
                print(f"      └─ _try_drain()       : check pending message injections")
        if has_init:
            init_vals = [inst.get("init_latency_ms", 0) or 0 for inst in instances if isinstance(inst.get("init_latency_ms", 0), (int, float))]
            gap_vals = [sum(inst.get("inter_turn_gaps_ms", []) or []) for inst in instances]
            fwk_run_vals = [inst.get("framework_run_ms", 0) or 0 for inst in instances if isinstance(inst.get("framework_run_ms", 0), (int, float))]
            if init_vals and fwk_run_vals:
                total_init = sum(init_vals)
                total_gaps = sum(gap_vals)
                total_fwk_run = sum(fwk_run_vals)
                unaccounted = max(0, total_fwk_run - total_gaps)
                total_all = total_init + total_gaps + unaccounted
                if total_all > 0:
                    print(f"\n    Framework Breakdown (aggregated):")
                    print(f"      Init              : {total_init:>10,.0f}ms ({total_init/total_all*100:5.1f}%)")
                    print(f"      Inter-turn gaps   : {total_gaps:>10,.0f}ms ({total_gaps/total_all*100:5.1f}%)")
                    if unaccounted > 100:
                        print(f"      Other (unaccounted): {unaccounted:>10,.0f}ms ({unaccounted/total_all*100:5.1f}%)")
                    bar_w = 40
                    i_n = min(int(total_init / total_all * bar_w), bar_w)
                    g_n = min(int(total_gaps / total_all * bar_w), bar_w - i_n)
                    u_n = bar_w - i_n - g_n
                    print(f"      {chr(9608)*i_n}{chr(9618)*g_n}{chr(9617)*u_n}  Init/Gaps/Other")

        print(f"\n  {c('bold')}▌ Latency Breakdown by Workload Type{c('reset')}")
        print(f"    {'Type':<8} {'AvgLLM':>8} {'AvgTool':>8} {'AvgFwk':>8} {'AvgTotal':>8} "
              f"{'LLM%':>6} {'Tool%':>6} {'Fwk%':>6}")
        print(f"    {'─'*8} {'─'*8} {'─'*8} {'─'*8} {'─'*8} {'─'*6} {'─'*6} {'─'*6}")

        for wt in sorted(by_type_bd.keys()):
            grp = by_type_bd[wt]
            color = c(wt)
            avg_llm = statistics.mean([i.get("total_llm_latency_ms", 0) / 1000 for i in grp])
            avg_tool = statistics.mean([i.get("total_tool_latency_ms", 0) / 1000 for i in grp])
            avg_fwk = statistics.mean([i.get("total_framework_latency_ms", 0) / 1000 for i in grp])
            avg_total = avg_llm + avg_tool + avg_fwk
            pct_l = avg_llm / avg_total * 100 if avg_total > 0 else 0
            pct_t = avg_tool / avg_total * 100 if avg_total > 0 else 0
            pct_f = avg_fwk / avg_total * 100 if avg_total > 0 else 0
            print(f"    {color}{wt:<8}{c('reset')} {c('llm')}{avg_llm:8.2f}{c('reset')} "
                  f"{c('tool')}{avg_tool:8.2f}{c('reset')} {c('framework')}{avg_fwk:8.2f}{c('reset')} "
                  f"{avg_total:8.2f} {pct_l:5.1f}% {pct_t:5.1f}% {pct_f:5.1f}%")

        max_type_total = max(
            statistics.mean([
                i.get("total_llm_latency_ms", 0) + i.get("total_tool_latency_ms", 0) +
                i.get("total_framework_latency_ms", 0)
                for i in grp
            ]) / 1000
            for grp in by_type_bd.values()
        ) if by_type_bd else 1

        print(f"\n    Average Latency Breakdown by Type (stacked):")
        for wt in sorted(by_type_bd.keys()):
            grp = by_type_bd[wt]
            color = c(wt)
            avg_llm = statistics.mean([i.get("total_llm_latency_ms", 0) / 1000 for i in grp])
            avg_tool = statistics.mean([i.get("total_tool_latency_ms", 0) / 1000 for i in grp])
            avg_fwk = statistics.mean([i.get("total_framework_latency_ms", 0) / 1000 for i in grp])
            bw = 35
            n_llm = min(int((avg_llm / max_type_total) * bw), bw) if max_type_total > 0 else 0
            n_tool = min(int((avg_tool / max_type_total) * bw), bw - n_llm) if max_type_total > 0 else 0
            n_fwk = min(int((avg_fwk / max_type_total) * bw), bw - n_llm - n_tool) if max_type_total > 0 else 0
            n_pad = bw - n_llm - n_tool - n_fwk
            stacked = (f"{c('llm')}{'█' * n_llm}{c('reset')}"
                       f"{c('tool')}{'█' * n_tool}{c('reset')}"
                       f"{c('framework')}{'█' * n_fwk}{c('reset')}"
                       f"{'░' * n_pad}")
            avg_total = avg_llm + avg_tool + avg_fwk
            print(f"      {color}{wt:<8}{c('reset')} │{stacked}│ {avg_total:.2f}s")

    # --- Token consumption ---
    print(f"\n  {c('bold')}▌ Token Consumption{c('reset')}")
    max_tok = max((i.get("total_prompt_tokens", 0) + i.get("total_completion_tokens", 0)
                    for i in instances), default=1)
    for inst in sorted(instances, key=lambda x: x.get("instance_id", 0)):
        iid = inst.get("instance_id", "?")
        wtype = inst.get("workload_type", "?")
        pt = inst.get("total_prompt_tokens", 0)
        ct = inst.get("total_completion_tokens", 0)
        tt = pt + ct
        color = c(wtype)
        b_prompt = bar(pt, max_tok, 25, "█")
        b_comp = bar(ct, max_tok, 10, "▓")
        print(f"    {color}#{iid:2d} {wtype:<8}{c('reset')} prompt {b_prompt} {pt:>6d}")
        print(f"    {'':>11} compl  {b_comp} {ct:>6d}")

    # --- Aggregated by workload type ---
    by_type: Dict[str, List[Dict]] = {}
    for inst in instances:
        wtype = inst.get("workload_type", "unknown")
        by_type.setdefault(wtype, []).append(inst)

    if by_type:
        print(f"\n  {c('bold')}▌ Workload Type Comparison{c('reset')}")
        print(f"    {'Type':<8} {'Count':>5} {'AvgTime':>8} {'AvgTools':>8} "
              f"{'AvgPTok':>8} {'AvgCTok':>8} {'AvgLLM':>8} {'AvgTool':>8} {'AvgFwk':>8} {'Errors':>6}")
        print(f"    {'─'*8} {'─'*5} {'─'*8} {'─'*8} {'─'*8} {'─'*8} {'─'*8} {'─'*8} {'─'*8} {'─'*6}")

        for wtype in sorted(by_type.keys()):
            group = by_type[wtype]
            color = c(wtype)
            avg_time = statistics.mean([i.get("total_latency_ms", 0) / 1000 for i in group])
            avg_tools = statistics.mean([i.get("total_tool_calls", 0) for i in group])
            avg_pt = statistics.mean([i.get("total_prompt_tokens", 0) for i in group])
            avg_ct = statistics.mean([i.get("total_completion_tokens", 0) for i in group])
            avg_llm = statistics.mean([i.get("total_llm_latency_ms", 0) / 1000 for i in group])
            avg_tool = statistics.mean([i.get("total_tool_latency_ms", 0) / 1000 for i in group])
            avg_fwk = statistics.mean([i.get("total_framework_latency_ms", 0) / 1000 for i in group])
            errors = sum(1 for i in group if i.get("error"))
            print(f"    {color}{wtype:<8}{c('reset')} {len(group):5d} {avg_time:8.1f}s "
                  f"{avg_tools:8.1f} {avg_pt:8.0f} {avg_ct:8.0f} "
                  f"{c('llm')}{avg_llm:7.2f}s{c('reset')} {c('tool')}{avg_tool:7.2f}s{c('reset')} "
                  f"{c('framework')}{avg_fwk:7.2f}s{c('reset')} {errors:6d}")

        # Type comparison bar chart
        print(f"\n    Average Task Duration by Type:")
        type_avgs = {}
        for wtype, group in by_type.items():
            type_avgs[wtype] = statistics.mean([i.get("total_latency_ms", 0) / 1000 for i in group])
        max_type_avg = max(type_avgs.values()) if type_avgs else 1
        for wtype in sorted(type_avgs.keys(), key=lambda x: type_avgs[x], reverse=True):
            color = c(wtype)
            b = bar(type_avgs[wtype], max_type_avg, 30, "█")
            print(f"      {color}{wtype:<8}{c('reset')} │{color}{b}{c('reset')}│ {type_avgs[wtype]:.1f}s")

    # --- Global latency distribution ---
    all_lats = [i.get("total_latency_ms", 0) for i in instances if not i.get("error")]
    if all_lats:
        sorted_lats = sorted(all_lats)
        print(f"\n  {c('bold')}▌ Task Latency Distribution (all successful){c('reset')}")
        print(f"    Avg  : {statistics.mean(sorted_lats)/1000:6.1f}s")
        p50 = sorted_lats[len(sorted_lats) // 2]
        p90 = sorted_lats[int(len(sorted_lats) * 0.90)]
        p95 = sorted_lats[int(len(sorted_lats) * 0.95)]
        p99 = sorted_lats[min(int(len(sorted_lats) * 0.99), len(sorted_lats) - 1)]
        print(f"    P50  : {p50/1000:6.1f}s")
        print(f"    P90  : {p90/1000:6.1f}s")
        print(f"    P95  : {p95/1000:6.1f}s")
        print(f"    P99  : {p99/1000:6.1f}s")
        print(f"    Max  : {sorted_lats[-1]/1000:6.1f}s")

        # Histogram
        print(f"\n    Latency Histogram:")
        bins = 8
        mn, mx = sorted_lats[0], sorted_lats[-1]
        rng = mx - mn if mx > mn else 1
        bin_size = rng / bins
        counts = [0] * bins
        for v in sorted_lats:
            idx = min(int((v - mn) / bin_size), bins - 1)
            counts[idx] += 1
        max_count = max(counts) if counts else 1
        for i in range(bins):
            lo = (mn + i * bin_size) / 1000
            hi = (mn + (i + 1) * bin_size) / 1000
            b = bar(counts[i], max_count, 30, "█")
            print(f"      {lo:5.1f}-{hi:5.1f}s │{b}│ {counts[i]}")

    total_tok = sum(i.get("total_prompt_tokens", 0) + i.get("total_completion_tokens", 0) for i in instances)
    dur = config.get("duration_sec", 1)
    avg_total_ms = statistics.mean([i.get('total_latency_ms',0) for i in instances])
    avg_llm_ms = statistics.mean([i.get('total_llm_latency_ms',0) for i in instances])
    avg_tool_ms = statistics.mean([i.get('total_tool_latency_ms',0) for i in instances])
    avg_fwk_ms = statistics.mean([i.get('total_framework_latency_ms',0) for i in instances])
    print(f"\n  {c('bold')}▌ Summary{c('reset')}")
    print(f"    Total Tokens       : {total_tok:,}")
    print(f"    Throughput         : {len(instances)/dur:.2f} tasks/sec")
    print(f"    Avg Task Latency   : {avg_total_ms/1000:.1f}s")
    print(f"    {c('bold')}Latency Breakdown:{c('reset')}")
    pct_l = avg_llm_ms / avg_total_ms * 100 if avg_total_ms > 0 else 0
    pct_t = avg_tool_ms / avg_total_ms * 100 if avg_total_ms > 0 else 0
    pct_f = avg_fwk_ms / avg_total_ms * 100 if avg_total_ms > 0 else 0
    print(f"      {c('llm')}Avg LLM       : {avg_llm_ms/1000:6.1f}s ({pct_l:4.1f}%){c('reset')}")
    print(f"      {c('tool')}Avg Tool      : {avg_tool_ms/1000:6.1f}s ({pct_t:4.1f}%){c('reset')}")
    print(f"      {c('framework')}Avg Framework : {avg_fwk_ms/1000:6.1f}s ({pct_f:4.1f}%){c('reset')}")
    print(f"    Avg Tool Calls     : {statistics.mean([i.get('total_tool_calls',0) for i in instances]):.1f}")
    print(f"    Avg Turns/Task     : {statistics.mean([i.get('num_turns',0) for i in instances]):.1f}")

    glb_ctx2 = glob_data if isinstance(glob_data, dict) else {}
    if glb_ctx2.get("total_processes") or any(i.get("num_threads") for i in instances):
        total_procs = glb_ctx2.get("total_processes", len(instances))
        total_threads = glb_ctx2.get("total_threads", 0) or sum(
            i.get("num_threads", 0) for i in instances if isinstance(i.get("num_threads"), int))
        total_vol_s = glb_ctx2.get("total_ctx_switches_vol", 0) or sum(
            i.get("ctx_switches_vol", 0) for i in instances)
        total_invol_s = glb_ctx2.get("total_ctx_switches_invol", 0) or sum(
            i.get("ctx_switches_invol", 0) for i in instances)
        print(f"    {c('bold')}Process Overhead:{c('reset')}")
        print(f"      Mode            : {config.get('mode', 'coro')}")
        print(f"      Processes       : {total_procs}")
        print(f"      Total Threads   : {total_threads}")
        print(f"      Ctx Switches    : {total_vol_s + total_invol_s:,} "
              f"({(total_vol_s+total_invol_s)/max(1,dur):.0f}/sec)")

    print()
    sep("═")
    print()


# ---------------------------------------------------------------------------
# Simulation format analysis (legacy)
# ---------------------------------------------------------------------------

def analyze_simulation(filepath: str):
    with open(filepath, "r") as f:
        data = json.load(f)

    config = data.get("test_config", {})
    glob_data = data.get("global", {})
    instances = data.get("per_instance", [])

    print()
    sep("═")
    print(f"  {c('bold')}📊 CPU OVERSUBSCRIPTION TEST REPORT — Simulation{c('reset')}")
    sep("═")
    print(f"  File      : {os.path.basename(filepath)}")
    print(f"  Timestamp : {config.get('timestamp', 'N/A')}")
    print(f"  Core      : {config.get('core_id', '?')}")
    print(f"  Duration  : {config.get('duration_sec', '?')}s")
    sep("─")

    cpu_samples = glob_data.get("cpu_samples", [])
    if cpu_samples:
        print(f"\n  {c('bold')}▌ CPU Utilization{c('reset')}")
        print(f"    Avg: {c('bold')}{statistics.mean(cpu_samples):.1f}%{c('reset')}  "
              f"Max: {c('bold')}{max(cpu_samples):.1f}%{c('reset')}  "
              f"Min: {min(cpu_samples):.1f}%")
        print(f"    {sparkline(cpu_samples, 70)}")

    print(f"\n  {c('bold')}▌ Per-Instance Latency{c('reset')}")
    print(f"    {'ID':>3} {'Type':<8} {'Ticks':>7} {'Avg':>8} {'P50':>8} {'P95':>8} {'P99':>8} {'Max':>8}")
    print(f"    {'─'*3} {'─'*8} {'─'*7} {'─'*8} {'─'*8} {'─'*8} {'─'*8} {'─'*8}")

    max_avg = max((i.get("latency_ms", {}).get("avg", 0) for i in instances), default=1)

    for inst in instances:
        lats = inst.get("latency_ms", {})
        iid = inst.get("instance_id", "?")
        wtype = inst.get("workload_type", "?")
        ticks = inst.get("total_ticks", 0)
        color = c(wtype)
        avg = lats.get("avg", 0)
        p50 = lats.get("p50", 0)
        p95 = lats.get("p95", 0)
        p99 = lats.get("p99", 0)
        mx = lats.get("max", 0)
        print(f"    {color}{iid:3d} {wtype:<8}{c('reset')} {ticks:7d} {avg:8.2f} {p50:8.2f} "
              f"{p95:8.2f} {p99:8.2f} {mx:8.2f}")

    if max_avg > 0:
        print(f"\n    Latency Bar Chart (avg ms):")
        for inst in instances:
            lats = inst.get("latency_ms", {})
            avg = lats.get("avg", 0)
            iid = inst.get("instance_id", "?")
            wtype = inst.get("workload_type", "?")
            color = c(wtype)
            b = bar(avg, max_avg, 35, "█")
            print(f"      {color}#{iid:2d} {wtype:<8}{c('reset')} │{color}{b}{c('reset')}│ {avg:.2f}ms")

    print()
    sep("═")
    print()


# ---------------------------------------------------------------------------
# Multi-file comparison
# ---------------------------------------------------------------------------

def compare_files(filepaths: List[str]):
    print()
    sep("═")
    print(f"  {c('bold')}📊 MULTI-RUN COMPARISON{c('reset')}")
    sep("═")

    print(f"\n  {'File':<35} {'Core':>4} {'Inst':>4} {'CPU%':>6} {'MaxCPU':>6} "
          f"{'Duration':>8} {'AvgLat':>8} {'LLM%':>5} {'Tool%':>5} {'Fwk%':>5} {'TotTok':>9}")
    print(f"  {'─'*35} {'─'*4} {'─'*4} {'─'*6} {'─'*6} {'─'*8} {'─'*8} {'─'*5} {'─'*5} {'─'*5} {'─'*9}")

    for fp in filepaths:
        with open(fp, "r") as f:
            data = json.load(f)
        cfg = data.get("test_config", {})
        glb = data.get("global", {})
        instances = data.get("per_instance", [])
        fmt = detect_format(data)

        cpu_avg = glb.get("cpu_avg", 0)
        cpu_max = glb.get("cpu_max", 0)
        dur = cfg.get("duration_sec", 0)

        if fmt == "agent":
            all_lats = [i.get("total_latency_ms", 0) for i in instances]
            all_llm = [i.get("total_llm_latency_ms", 0) for i in instances]
            all_tool = [i.get("total_tool_latency_ms", 0) for i in instances]
            all_fwk = [i.get("total_framework_latency_ms", 0) for i in instances]
            all_tok = sum(i.get("total_prompt_tokens", 0) + i.get("total_completion_tokens", 0)
                         for i in instances)
            avg_total = statistics.mean(all_lats) if all_lats else 1
            avg_llm_pct = statistics.mean(all_llm) / avg_total * 100 if avg_total > 0 else 0
            avg_tool_pct = statistics.mean(all_tool) / avg_total * 100 if avg_total > 0 else 0
            avg_fwk_pct = statistics.mean(all_fwk) / avg_total * 100 if avg_total > 0 else 0
        else:
            all_lats = [i.get("latency_ms", {}).get("avg", 0) for i in instances if i.get("latency_ms")]
            all_tok = 0
            avg_llm_pct = 0
            avg_tool_pct = 0
            avg_fwk_pct = 0

        avg_lat = statistics.mean(all_lats) if all_lats else 0
        avg_lat_str = f"{avg_lat/1000:.1f}s" if fmt == "agent" else f"{avg_lat:.1f}ms"

        print(f"  {os.path.basename(fp):<35} {cfg.get('core_id', '?'):>4} "
              f"{cfg.get('num_instances', '?'):>4} {cpu_avg:>5.1f}% {cpu_max:>5.1f}% "
              f"{dur:>7.1f}s {avg_lat_str:>8} "
              f"{c('llm')}{avg_llm_pct:4.0f}%{c('reset')} {c('tool')}{avg_tool_pct:4.0f}%{c('reset')} "
              f"{c('framework')}{avg_fwk_pct:4.0f}%{c('reset')} {all_tok:>9,}")

    print()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Analyze & Visualize CPU Oversubscription Test Results"
    )
    parser.add_argument("files", nargs="*", help="Result JSON files")
    parser.add_argument("--all", "-a", action="store_true", help="Analyze all result files")
    parser.add_argument("--compare", action="store_true", help="Compare multiple runs")
    parser.add_argument("--latest", "-l", action="store_true", help="Analyze latest result only")
    args = parser.parse_args()

    if args.files:
        files = args.files
    elif args.all:
        files = sorted(glob.glob(os.path.join(RESULTS_DIR, "oversub*.json")))
    else:
        agent_files = sorted(glob.glob(os.path.join(RESULTS_DIR, "oversub_agent_*.json")))
        sim_files = sorted(glob.glob(os.path.join(RESULTS_DIR, "oversub_*.json")))
        all_files = sorted(glob.glob(os.path.join(RESULTS_DIR, "oversub*.json")))
        files = [all_files[-1]] if all_files else []

    if not files:
        print("No result files found. Run cpu_oversub.py first.")
        sys.exit(1)

    if args.compare and len(files) > 1:
        compare_files(files)
    else:
        for fp in files:
            with open(fp, "r") as f:
                data = json.load(f)
            fmt = detect_format(data)
            if fmt == "agent":
                analyze_agent(fp)
            elif fmt == "simulation":
                analyze_simulation(fp)
            else:
                print(f"Unknown format: {fp}")


if __name__ == "__main__":
    main()
