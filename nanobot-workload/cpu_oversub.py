#!/usr/bin/env python3
"""
CPU Oversubscription Load Test — Real nanobot-ai Agent Workloads

Constructs a CPU oversubscription scenario: N nanobot agent instances pinned
to a single CPU core, each executing real tasks that exercise different tools:

  - CPU-heavy:  code analysis, sorting algorithms, mathematical computation
  - I/O-heavy:  file read/write, grep, glob (filesystem tools)
  - Network:    web search, web fetch (network tools)
  - Mixed:      tasks that combine multiple tool types

Metrics collected per-instance:
  - End-to-end task latency (time from submission to completion)
  - LLM round-trip time per turn
  - Tool call count and types
  - Tokens consumed (prompt + completion)
  - CPU utilization of pinned core
  - Context switch count

All software is installed under /data2/cht/agentloop.
"""

import sys
import os
import time
import json
import asyncio
import threading
import statistics
import argparse
import traceback
from dataclasses import dataclass, field, asdict
from typing import List, Dict, Optional, Tuple
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, BASE_DIR)
os.environ.setdefault("HOME", BASE_DIR)

import psutil

from nanobot import Nanobot, RunResult
from nanobot.agent.hook import AgentHook, AgentHookContext

CONFIG_PATH = os.path.join(BASE_DIR, ".nanobot", "config.json")
WORKSPACE = os.path.join(BASE_DIR, "workspace")
RESULTS_DIR = os.path.join(BASE_DIR, "results")


# ---------------------------------------------------------------------------
# Task definitions — different workload profiles
# ---------------------------------------------------------------------------

TASK_PROFILES = {
    "cpu": [
        "Write a Python function that computes the first 500 prime numbers using the Sieve of Eratosthenes. Save it to workspace/instance_{id}_primes.py and run it.",
        "Create a Python script that implements a Fibonacci heap with insert, extract-min, and decrease-key operations. Save to workspace/instance_{id}_fibheap.py.",
        "Write a Python script that multiplies two 100x100 matrices using pure Python (no numpy). Save to workspace/instance_{id}_matrix.py and execute it to verify correctness.",
        "Implement a Python LRU cache class from scratch with get, put, and eviction. Save to workspace/instance_{id}_lru.py and test it.",
    ],
    "io": [
        "Create a file workspace/instance_{id}_data.txt with 50 lines of random CSV data (id,name,score). Then read it back and count the lines.",
        "Write a Python script to workspace/instance_{id}_scanner.py that recursively lists all .py files in /data2/cht/agentloop/nanobot/agent/tools/ and counts them.",
        "Read the file /data2/cht/agentloop/nanobot/agent/tools/base.py and summarize what the Tool base class provides. Save summary to workspace/instance_{id}_summary.txt.",
        "Create workspace/instance_{id}_report.md with a markdown table of 10 items (name, value). Then read it back to verify.",
    ],
    "network": [
        "Search the web for 'Python asyncio best practices 2025' and summarize the top 3 results. Save to workspace/instance_{id}_search.md.",
        "Search for 'Linux CPU affinity scheduling' and write a brief summary to workspace/instance_{id}_cpu_affinity.md.",
        "Web search 'docker resource limits CPU shares' and save a short note to workspace/instance_{id}_docker_cpu.md.",
        "Search for 'Python GIL thread performance' and save key findings to workspace/instance_{id}_gil.md.",
    ],
    "mixed": [
        "Search the web for 'quick sort algorithm Python', then write an implementation to workspace/instance_{id}_quicksort.py and run it with a test array.",
        "Read /data2/cht/agentloop/nanobot/config/schema.py, find the AgentDefaults class, and create workspace/instance_{id}_defaults.json with its default values.",
        "Search for 'Python dataclass vs pydantic', read the file /data2/cht/agentloop/nanobot/agent/tools/base.py, then write a comparison to workspace/instance_{id}_compare.md.",
        "Create workspace/instance_{id}_benchmark.py that times list comprehension vs map for 1M items, run it, and save results to workspace/instance_{id}_bench_results.txt.",
    ],
}


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------

@dataclass
class TurnMetrics:
    turn_index: int
    latency_ms: float
    tool_calls: int
    prompt_tokens: int = 0
    completion_tokens: int = 0
    finish_reason: str = ""
    llm_latency_ms: float = 0.0
    tool_latency_ms: float = 0.0
    tool_names: List[str] = field(default_factory=list)


@dataclass
class InstanceMetrics:
    instance_id: int
    workload_type: str
    task_prompt: str
    total_latency_ms: float = 0.0
    turns: List[TurnMetrics] = field(default_factory=list)
    total_tool_calls: int = 0
    total_prompt_tokens: int = 0
    total_completion_tokens: int = 0
    total_llm_latency_ms: float = 0.0
    total_tool_latency_ms: float = 0.0
    total_framework_latency_ms: float = 0.0
    error: str = ""
    response_preview: str = ""


@dataclass
class GlobalMetrics:
    core_id: int = 0
    duration_sec: float = 0.0
    num_instances: int = 0
    concurrency: int = 1
    cpu_samples: List[float] = field(default_factory=list)
    ctx_switches_start: int = 0
    ctx_switches_end: int = 0
    per_instance: List[Dict] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Hook to capture turn-level metrics
# ---------------------------------------------------------------------------

class MetricsHook(AgentHook):
    """Captures timing and token data from the agent loop via AgentHook.

    Uses three hook points to decompose per-turn latency:
      - before_iteration  (T0): marks start of iteration
      - before_execute_tools (T1): marks end of LLM call, start of tool execution
      - after_iteration  (T2): marks end of iteration (after tools + framework)

    Latency decomposition:
      - LLM latency       = T1 - T0   (model API round-trip)
      - Tool latency      = T2 - T1   (tool execution)
      - Framework latency = total_wall - Σ(LLM + Tool)
        (context governance, message building, checkpoint, etc.)
    """

    def __init__(self):
        super().__init__(reraise=False)
        self._t0 = 0.0
        self._t1 = 0.0
        self._has_tools = False
        self.turns: List[TurnMetrics] = []
        self.tool_calls_total = 0
        self.prompt_tokens_total = 0
        self.completion_tokens_total = 0
        self._turn_idx = 0
        self._wall_start = 0.0

    def _now(self) -> float:
        return time.perf_counter()

    async def before_iteration(self, context: AgentHookContext) -> None:
        self._t0 = self._now()
        self._t1 = 0.0
        self._has_tools = False
        if self._turn_idx == 0:
            self._wall_start = self._t0

    async def before_execute_tools(self, context: AgentHookContext) -> None:
        self._t1 = self._now()
        self._has_tools = True

    async def after_iteration(self, context: AgentHookContext) -> None:
        t2 = self._now()
        wall_ms = (t2 - self._t0) * 1000

        if self._has_tools and self._t1 > 0:
            llm_ms = (self._t1 - self._t0) * 1000
            tool_ms = (t2 - self._t1) * 1000
        else:
            llm_ms = wall_ms
            tool_ms = 0.0

        usage = context.usage
        pt = usage.get("prompt_tokens", 0)
        ct = usage.get("completion_tokens", 0)
        tc = len(context.tool_calls) if context.tool_calls else 0
        tool_names = [tc_item.name for tc_item in context.tool_calls] if context.tool_calls else []

        self.turns.append(TurnMetrics(
            turn_index=self._turn_idx,
            latency_ms=round(wall_ms, 2),
            tool_calls=tc,
            prompt_tokens=pt,
            completion_tokens=ct,
            llm_latency_ms=round(llm_ms, 2),
            tool_latency_ms=round(tool_ms, 2),
            tool_names=tool_names,
        ))
        self.tool_calls_total += tc
        self.prompt_tokens_total += pt
        self.completion_tokens_total += ct
        self._turn_idx += 1


# ---------------------------------------------------------------------------
# Core monitor
# ---------------------------------------------------------------------------

class CoreMonitor:
    def __init__(self, core_id: int, interval: float = 1.0):
        self.core_id = core_id
        self.interval = interval
        self.cpu_samples: List[float] = []
        self.running = False
        self._thread = None

    def start(self):
        self.running = True
        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()

    def stop(self):
        self.running = False
        if self._thread:
            self._thread.join(timeout=3)

    def _loop(self):
        while self.running:
            pcts = psutil.cpu_percent(interval=self.interval, percpu=True)
            if pcts and len(pcts) > self.core_id:
                self.cpu_samples.append(pcts[self.core_id])


# ---------------------------------------------------------------------------
# Agent Instance — runs a single task via nanobot
# ---------------------------------------------------------------------------

async def run_agent_task(
    instance_id: int,
    workload_type: str,
    task_prompt: str,
) -> InstanceMetrics:
    metrics = InstanceMetrics(
        instance_id=instance_id,
        workload_type=workload_type,
        task_prompt=task_prompt[:200],
    )
    hook = MetricsHook()
    t0 = time.perf_counter()
    try:
        bot = Nanobot.from_config(config_path=CONFIG_PATH)
        result = await bot.run(task_prompt, session_key=f"oversub:{instance_id}", hooks=[hook])
        metrics.response_preview = result.content[:500] if result.content else ""
    except Exception as e:
        metrics.error = f"{type(e).__name__}: {str(e)[:200]}"
        traceback.print_exc()
    elapsed = (time.perf_counter() - t0) * 1000
    metrics.total_latency_ms = round(elapsed, 2)
    metrics.turns = hook.turns
    metrics.total_tool_calls = hook.tool_calls_total
    metrics.total_prompt_tokens = hook.prompt_tokens_total
    metrics.total_completion_tokens = hook.completion_tokens_total

    total_llm = sum(t.llm_latency_ms for t in hook.turns)
    total_tool = sum(t.tool_latency_ms for t in hook.turns)
    total_framework = max(0, elapsed - total_llm - total_tool)
    metrics.total_llm_latency_ms = round(total_llm, 2)
    metrics.total_tool_latency_ms = round(total_tool, 2)
    metrics.total_framework_latency_ms = round(total_framework, 2)

    return metrics


# ---------------------------------------------------------------------------
# Test Orchestrator
# ---------------------------------------------------------------------------

class OversubscriptionTest:
    def __init__(
        self,
        core_id: int = 0,
        concurrency: int = 10,
        tasks_per_type: int = 3,
        task_types: Optional[List[str]] = None,
    ):
        self.core_id = core_id
        self.concurrency = concurrency
        self.tasks_per_type = tasks_per_type
        self.task_types = task_types or ["cpu", "io", "network", "mixed"]
        self.monitor = CoreMonitor(core_id)
        self.global_metrics = GlobalMetrics(
            core_id=core_id,
            concurrency=concurrency,
        )

    def _pin_process(self):
        try:
            proc = psutil.Process(os.getpid())
            proc.cpu_affinity([self.core_id])
            print(f"  [OK] Process pinned to core {self.core_id}")
            return True
        except Exception as e:
            print(f"  [WARN] Pin failed: {e}")
            return False

    def _build_tasks(self) -> List[Tuple[int, str, str]]:
        tasks = []
        iid = 0
        for wtype in self.task_types:
            templates = TASK_PROFILES.get(wtype, [])
            for i in range(self.tasks_per_type):
                template = templates[i % len(templates)]
                prompt = template.replace("{id}", str(iid))
                tasks.append((iid, wtype, prompt))
                iid += 1
        self.global_metrics.num_instances = len(tasks)
        return tasks

    def _print_header(self):
        print()
        print("=" * 78)
        print("  CPU OVERSUBSCRIPTION LOAD TEST — Real nanobot-ai Agent Workloads")
        print("=" * 78)
        print(f"  Target Core   : {self.core_id}")
        print(f"  Concurrency   : {self.concurrency} simultaneous agents")
        print(f"  Task Types    : {', '.join(self.task_types)}")
        print(f"  Tasks/Type    : {self.tasks_per_type}")
        total = len(self.task_types) * self.tasks_per_type
        print(f"  Total Tasks   : {total}")
        print(f"  Model         : glm-4.7 (via VolcEngine)")
        print(f"  Config        : {CONFIG_PATH}")
        print("=" * 78)
        print()

    async def _run_all(self, tasks: List[Tuple[int, str, str]]) -> List[InstanceMetrics]:
        semaphore = asyncio.Semaphore(self.concurrency)
        results = []

        async def bounded_run(iid, wtype, prompt):
            async with semaphore:
                return await run_agent_task(iid, wtype, prompt)

        coros = [bounded_run(iid, wtype, prompt) for iid, wtype, prompt in tasks]
        for i, coro in enumerate(asyncio.as_completed(coros)):
            result = await coro
            elapsed_s = result.total_latency_ms / 1000
            status = "OK" if not result.error else f"ERR: {result.error[:50]}"
            print(f"  [{i+1:2d}/{len(tasks)}] #{result.instance_id:2d} {result.workload_type:<8} "
                  f"{elapsed_s:6.1f}s  tools={result.total_tool_calls}  {status}")
            results.append(result)

        return results

    def run(self):
        self._print_header()
        self._pin_process()
        os.makedirs(WORKSPACE, exist_ok=True)
        os.makedirs(RESULTS_DIR, exist_ok=True)

        tasks = self._build_tasks()
        if not tasks:
            print("No tasks to run!")
            return

        if psutil:
            proc = psutil.Process(os.getpid())
            self.global_metrics.ctx_switches_start = proc.num_ctx_switches().voluntary

        self.monitor.start()
        start = time.time()

        print(f"  Launching {len(tasks)} agent tasks (max {self.concurrency} concurrent)...\n")

        results = asyncio.run(self._run_all(tasks))

        duration = time.time() - start
        self.global_metrics.duration_sec = round(duration, 2)

        self.monitor.stop()

        if psutil:
            proc = psutil.Process(os.getpid())
            self.global_metrics.ctx_switches_end = proc.num_ctx_switches().voluntary

        for r in results:
            self.global_metrics.per_instance.append({
                "instance_id": r.instance_id,
                "workload_type": r.workload_type,
                "total_latency_ms": r.total_latency_ms,
                "total_tool_calls": r.total_tool_calls,
                "total_prompt_tokens": r.total_prompt_tokens,
                "total_completion_tokens": r.total_completion_tokens,
                "total_llm_latency_ms": r.total_llm_latency_ms,
                "total_tool_latency_ms": r.total_tool_latency_ms,
                "total_framework_latency_ms": r.total_framework_latency_ms,
                "num_turns": len(r.turns),
                "error": r.error,
                "turn_latencies_ms": [t.latency_ms for t in r.turns],
                "turns": [
                    {
                        "turn_index": t.turn_index,
                        "latency_ms": t.latency_ms,
                        "llm_latency_ms": t.llm_latency_ms,
                        "tool_latency_ms": t.tool_latency_ms,
                        "tool_calls": t.tool_calls,
                        "tool_names": t.tool_names,
                        "prompt_tokens": t.prompt_tokens,
                        "completion_tokens": t.completion_tokens,
                    }
                    for t in r.turns
                ],
            })

        self.global_metrics.cpu_samples = self.monitor.cpu_samples

        self._save_results()
        self._print_report(results)

    def _save_results(self):
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        path = os.path.join(RESULTS_DIR, f"oversub_agent_{ts}.json")
        gm = self.global_metrics
        data = {
            "test_config": {
                "core_id": gm.core_id,
                "concurrency": gm.concurrency,
                "num_instances": gm.num_instances,
                "task_types": self.task_types,
                "tasks_per_type": self.tasks_per_type,
                "duration_sec": gm.duration_sec,
                "timestamp": datetime.now().isoformat(),
                "model": "glm-4.7",
                "provider": "volcengine",
            },
            "global": {
                "cpu_avg": round(statistics.mean(self.monitor.cpu_samples), 2) if self.monitor.cpu_samples else 0,
                "cpu_max": round(max(self.monitor.cpu_samples), 2) if self.monitor.cpu_samples else 0,
                "cpu_min": round(min(self.monitor.cpu_samples), 2) if self.monitor.cpu_samples else 0,
                "cpu_stdev": round(statistics.stdev(self.monitor.cpu_samples), 2) if len(self.monitor.cpu_samples) > 1 else 0,
                "cpu_samples": [round(s, 2) for s in self.monitor.cpu_samples],
                "ctx_switches_delta": gm.ctx_switches_end - gm.ctx_switches_start,
                "duration_sec": gm.duration_sec,
            },
            "per_instance": gm.per_instance,
        }
        with open(path, "w") as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
        print(f"\n  Results saved: {path}")

    def _print_report(self, results: List[InstanceMetrics]):
        gm = self.global_metrics
        print()
        print("=" * 78)
        print("  TEST RESULTS")
        print("=" * 78)

        samples = self.monitor.cpu_samples
        if samples:
            print(f"\n  CPU Core {gm.core_id} Utilization:")
            print(f"    Average : {statistics.mean(samples):6.1f}%")
            print(f"    Maximum : {max(samples):6.1f}%")
            print(f"    Minimum : {min(samples):6.1f}%")
            if len(samples) > 1:
                print(f"    StdDev  : {statistics.stdev(samples):6.1f}%")

        ctx = gm.ctx_switches_end - gm.ctx_switches_start
        print(f"\n  Context Switches : {ctx}")
        print(f"  Total Duration   : {gm.duration_sec:.1f}s")

        print(f"\n  Per-Instance Results:")
        print(f"  {'ID':>3} {'Type':<8} {'Time(s)':>8} {'Turns':>5} {'Tools':>5} "
              f"{'PromptT':>8} {'CompT':>7} {'LLM(s)':>7} {'Tool(s)':>7} {'Fwk(s)':>7} {'Error':<20}")
        print(f"  {'---':>3} {'--------':<8} {'--------':>8} {'-----':>5} {'-----':>5} "
              f"{'--------':>8} {'-------':>7} {'-------':>7} {'-------':>7} {'-------':>7} {'--------------------':<20}")

        for r in sorted(results, key=lambda x: x.instance_id):
            err_display = r.error[:20] if r.error else ""
            print(f"  {r.instance_id:3d} {r.workload_type:<8} "
                  f"{r.total_latency_ms/1000:8.1f} {len(r.turns):5d} {r.total_tool_calls:5d} "
                  f"{r.total_prompt_tokens:8d} {r.total_completion_tokens:7d} "
                  f"{r.total_llm_latency_ms/1000:7.1f} {r.total_tool_latency_ms/1000:7.1f} "
                  f"{r.total_framework_latency_ms/1000:7.1f} {err_display:<20}")

        by_type: Dict[str, List[InstanceMetrics]] = {}
        for r in results:
            by_type.setdefault(r.workload_type, []).append(r)

        if by_type:
            print(f"\n  Aggregated by Workload Type:")
            print(f"  {'Type':<8} {'Count':>5} {'AvgTime':>8} {'AvgTools':>8} "
                  f"{'AvgPTok':>8} {'AvgCTok':>8} {'AvgLLM':>8} {'AvgTool':>8} {'AvgFwk':>8} {'Errors':>6}")
            print(f"  {'-'*8} {'-'*5} {'-'*8} {'-'*8} {'-'*8} {'-'*8} {'-'*8} {'-'*8} {'-'*8} {'-'*6}")
            for wtype, group in sorted(by_type.items()):
                avg_time = statistics.mean([r.total_latency_ms / 1000 for r in group])
                avg_tools = statistics.mean([r.total_tool_calls for r in group])
                avg_pt = statistics.mean([r.total_prompt_tokens for r in group])
                avg_ct = statistics.mean([r.total_completion_tokens for r in group])
                avg_llm = statistics.mean([r.total_llm_latency_ms / 1000 for r in group])
                avg_tool = statistics.mean([r.total_tool_latency_ms / 1000 for r in group])
                avg_fwk = statistics.mean([r.total_framework_latency_ms / 1000 for r in group])
                errors = sum(1 for r in group if r.error)
                print(f"  {wtype:<8} {len(group):5d} {avg_time:8.1f} {avg_tools:8.1f} "
                      f"{avg_pt:8.0f} {avg_ct:8.0f} {avg_llm:8.1f} {avg_tool:8.1f} "
                      f"{avg_fwk:8.1f} {errors:6d}")

        all_latencies = [r.total_latency_ms for r in results if not r.error]
        if all_latencies:
            sorted_lats = sorted(all_latencies)
            print(f"\n  Task Latency Distribution (successful):")
            print(f"    Avg  : {statistics.mean(sorted_lats)/1000:.1f}s")
            print(f"    P50  : {sorted_lats[len(sorted_lats)//2]/1000:.1f}s")
            print(f"    P95  : {sorted_lats[int(len(sorted_lats)*0.95)]/1000:.1f}s")
            print(f"    P99  : {sorted_lats[int(len(sorted_lats)*0.99)]/1000:.1f}s")
            print(f"    Max  : {sorted_lats[-1]/1000:.1f}s")

        total_tokens = sum(r.total_prompt_tokens + r.total_completion_tokens for r in results)
        print(f"\n  Total Tokens Consumed: {total_tokens}")
        print(f"  Throughput: {len(results)/gm.duration_sec:.2f} tasks/sec")
        print()
        print("=" * 78)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="CPU Oversubscription Load Test with Real nanobot-ai Agents"
    )
    parser.add_argument("--core", "-c", type=int, default=0, help="CPU core to pin to")
    parser.add_argument("--concurrency", "-n", type=int, default=10, help="Max concurrent agents")
    parser.add_argument("--tasks-per-type", "-t", type=int, default=3, help="Tasks per workload type")
    parser.add_argument(
        "--types", type=str, default="cpu,io,network,mixed",
        help="Comma-separated workload types (default: cpu,io,network,mixed)"
    )
    args = parser.parse_args()

    types = [t.strip() for t in args.types.split(",")]
    test = OversubscriptionTest(
        core_id=args.core,
        concurrency=args.concurrency,
        tasks_per_type=args.tasks_per_type,
        task_types=types,
    )
    test.run()


if __name__ == "__main__":
    main()
