#!/usr/bin/env python3
"""
CPU Oversubscription Load Test - Multi-Process Model

Each nanobot agent instance runs as an independent subprocess with its own:
  - Process ID (PID), event loop, memory space
  - Workspace directory (file isolation)
  - .nanobot config (session isolation)
  - LLM concurrency gate

This mirrors real data-center deployment where each agent is an isolated process/container.
"""

from __future__ import annotations

import argparse
import json
import os
import statistics
import subprocess
import sys
import threading
import time
import traceback
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

try:
    import psutil
except ImportError:
    psutil = None

BASE_DIR = Path(__file__).resolve().parent
CONFIG_PATH = str(BASE_DIR / ".nanobot" / "config.json")
WORKSPACE_ROOT = BASE_DIR / "workspace"
RESULTS_DIR = BASE_DIR / "results"

TASK_PROFILES = {
    "cpu": [
        "Write a Python function that implements the merge sort algorithm. Save it to workspace/instance_{id}_mergesort.py, then run it with a test array [38, 27, 43, 3, 9, 82, 10] using exec.",
        "Implement a Python script that computes the first 100 Fibonacci numbers using memoization. Save to workspace/instance_{id}_fib.py and execute it.",
        "Write a Python function to find the longest common subsequence of two strings. Save to workspace/instance_{id}_lcs.py and test with 'ABCBDAB' and 'BDCAB'.",
    ],
    "io": [
        "Create a file workspace/instance_{id}_data.txt with 50 lines of random CSV data (name,age,city). Then read it back and count the lines using grep or read_file.",
        "Write a short log file workspace/instance_{id}_app.log with 20 timestamped entries, then use grep to find all ERROR entries in it.",
        "Create workspace/instance_{id}_notes.md with some markdown content, then use list_dir to see the workspace and read_file to verify the content.",
    ],
    "network": [
        "Search the web for 'Python asyncio best practices 2025' using web_search, summarize the top 3 results, and save the summary to workspace/instance_{id}_summary.md.",
        "Use web_search to find information about 'Linux CPU scheduling CFS', then fetch one of the result pages with web_fetch and save key points to workspace/instance_{id}_cfs.md.",
        "Search for 'nanobot AI agent framework' using web_search, and write a brief comparison with other agent frameworks to workspace/instance_{id}_comparison.md.",
    ],
    "mixed": [
        "Search the web for 'Python quicksort implementation', then write the code to workspace/instance_{id}_quicksort.py, execute it, and report the output.",
        "Use web_search to find 'Python thread pool best practices', create a demo script at workspace/instance_{id}_threadpool.py, run it, and save the output.",
        "Search for 'Linux /proc/cpuinfo format', create a script that parses it at workspace/instance_{id}_cpuparser.py, and run it.",
    ],
}


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


WORKER_SCRIPT = '''
import asyncio
import json
import os
import sys
import time
import traceback

BASE_DIR = os.environ.get("NANOBOT_BASE_DIR", ".")
sys.path.insert(0, BASE_DIR)
os.environ["HOME"] = BASE_DIR

from nanobot import Nanobot
from nanobot.agent.hook import AgentHook, AgentHookContext


class MetricsHook(AgentHook):
    def __init__(self):
        super().__init__(reraise=False)
        self._t0 = 0.0
        self._t1 = 0.0
        self._has_tools = False
        self.turns = []
        self.tool_calls_total = 0
        self.prompt_tokens_total = 0
        self.completion_tokens_total = 0
        self._turn_idx = 0

    def _now(self):
        return time.perf_counter()

    async def before_iteration(self, context):
        self._t0 = self._now()
        self._t1 = 0.0
        self._has_tools = False

    async def before_execute_tools(self, context):
        self._t1 = self._now()
        self._has_tools = True

    async def after_iteration(self, context):
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
        self.turns.append({
            "turn_index": self._turn_idx,
            "latency_ms": round(wall_ms, 2),
            "llm_latency_ms": round(llm_ms, 2),
            "tool_latency_ms": round(tool_ms, 2),
            "tool_calls": tc,
            "tool_names": tool_names,
            "prompt_tokens": pt,
            "completion_tokens": ct,
        })
        self.tool_calls_total += tc
        self.prompt_tokens_total += pt
        self.completion_tokens_total += ct
        self._turn_idx += 1


async def main():
    instance_id = int(sys.argv[1])
    workload_type = sys.argv[2]
    task_prompt = sys.argv[3]
    config_path = sys.argv[4]
    workspace_dir = sys.argv[5]
    result_file = sys.argv[6]

    os.makedirs(workspace_dir, exist_ok=True)
    os.chdir(workspace_dir)

    hook = MetricsHook()
    result_data = {
        "instance_id": instance_id,
        "workload_type": workload_type,
        "task_prompt": task_prompt[:200],
        "pid": os.getpid(),
    }

    t_init_start = time.perf_counter()
    try:
        bot = Nanobot.from_config(config_path=config_path)
    except Exception as e:
        result_data["error"] = f"InitError: {type(e).__name__}: {str(e)[:200]}"
        traceback.print_exc()
        with open(result_file, "w") as f:
            json.dump(result_data, f, indent=2)
        return
    init_ms = (time.perf_counter() - t_init_start) * 1000

    t_run_start = time.perf_counter()
    try:
        result = await bot.run(task_prompt, session_key=f"oversub:{instance_id}", hooks=[hook])
        result_data["response_preview"] = result.content[:500] if result.content else ""
    except Exception as e:
        result_data["error"] = f"{type(e).__name__}: {str(e)[:200]}"
        traceback.print_exc()
    run_ms = (time.perf_counter() - t_run_start) * 1000
    elapsed_ms = init_ms + run_ms

    total_llm = sum(t["llm_latency_ms"] for t in hook.turns)
    total_tool = sum(t["tool_latency_ms"] for t in hook.turns)
    total_fwk_run = max(0, run_ms - total_llm - total_tool)

    result_data.update({
        "total_latency_ms": round(elapsed_ms, 2),
        "init_latency_ms": round(init_ms, 2),
        "run_latency_ms": round(run_ms, 2),
        "total_tool_calls": hook.tool_calls_total,
        "total_prompt_tokens": hook.prompt_tokens_total,
        "total_completion_tokens": hook.completion_tokens_total,
        "total_llm_latency_ms": round(total_llm, 2),
        "total_tool_latency_ms": round(total_tool, 2),
        "total_framework_latency_ms": round(total_fwk_run + init_ms, 2),
        "framework_run_ms": round(total_fwk_run, 2),
        "num_turns": len(hook.turns),
        "turns": hook.turns,
    })

    with open(result_file, "w") as f:
        json.dump(result_data, f, indent=2)


asyncio.run(main())
'''


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

    def _pin_process(self):
        try:
            proc = psutil.Process(os.getpid())
            proc.cpu_affinity([self.core_id])
            print(f"  [OK] Orchestrator pinned to core {self.core_id}")
        except Exception as e:
            print(f"  [WARN] Pin failed: {e}")

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
        return tasks

    def _setup_instance_dirs(self, instance_id: int) -> Tuple[str, str, str]:
        inst_dir = WORKSPACE_ROOT / f"instance_{instance_id}"
        inst_dir.mkdir(parents=True, exist_ok=True)
        nanobot_dir = inst_dir / ".nanobot"
        nanobot_dir.mkdir(parents=True, exist_ok=True)
        config_dst = nanobot_dir / "config.json"
        if not config_dst.exists():
            import shutil
            shutil.copy2(CONFIG_PATH, config_dst)
        result_file = inst_dir / "result.json"
        return str(config_dst), str(inst_dir), str(result_file)

    def _read_result(self, result_file: str) -> dict:
        try:
            with open(result_file, "r") as f:
                return json.load(f)
        except Exception:
            return {"error": "Failed to read result file"}

    def _collect_process_metrics(self, pid: int) -> dict:
        if not psutil:
            return {}
        try:
            proc = psutil.Process(pid)
            ctx = proc.num_ctx_switches()
            return {
                "num_threads": proc.num_threads(),
                "ctx_switches_vol": ctx.voluntary,
                "ctx_switches_invol": ctx.involuntary,
            }
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            return {}

    def run(self):
        print()
        print("=" * 78)
        print("  CPU OVERSUBSCRIPTION LOAD TEST - Multi-Process Model")
        print("  Each agent instance = independent subprocess (real isolation)")
        print("=" * 78)
        print(f"  Target Core   : {self.core_id}")
        print(f"  Concurrency   : {self.concurrency} simultaneous processes")
        print(f"  Task Types    : {', '.join(self.task_types)}")
        print(f"  Tasks/Type    : {self.tasks_per_type}")
        total = len(self.task_types) * self.tasks_per_type
        print(f"  Total Tasks   : {total}")
        print("=" * 78)
        print()

        self._pin_process()
        WORKSPACE_ROOT.mkdir(parents=True, exist_ok=True)
        RESULTS_DIR.mkdir(parents=True, exist_ok=True)

        tasks = self._build_tasks()
        if not tasks:
            print("No tasks to run!")
            return

        worker_script = BASE_DIR / "_worker_agent.py"
        with open(worker_script, "w") as f:
            f.write(WORKER_SCRIPT)

        self.monitor.start()
        start_time = time.time()

        print(f"  Launching {len(tasks)} agent processes (max {self.concurrency} concurrent)...\n")

        running: Dict[int, subprocess.Popen] = {}
        pending: List[Tuple[int, str, str, str, str, str]] = []
        completed: List[Tuple[int, str, dict]] = []

        for iid, wtype, prompt in tasks:
            config_dst, inst_dir, result_file = self._setup_instance_dirs(iid)
            pending.append((iid, wtype, prompt, config_dst, inst_dir, result_file))

        task_idx = 0
        active_count = 0

        def launch_next():
            nonlocal task_idx, active_count
            while task_idx < len(pending) and active_count < self.concurrency:
                iid, wtype, prompt, config_dst, inst_dir, result_file = pending[task_idx]
                env = os.environ.copy()
                env["NANOBOT_BASE_DIR"] = str(BASE_DIR)
                env["HOME"] = str(inst_dir)
                env["PYTHONPATH"] = str(BASE_DIR)
                try:
                    proc = subprocess.Popen(
                        [sys.executable, str(worker_script),
                         str(iid), wtype, prompt, config_dst, inst_dir, result_file],
                        env=env,
                        stdout=subprocess.PIPE,
                        stderr=subprocess.PIPE,
                    )
                    try:
                        p = psutil.Process(proc.pid)
                        p.cpu_affinity([self.core_id])
                    except Exception:
                        pass
                    running[iid] = proc
                    active_count += 1
                except Exception as e:
                    completed.append((iid, wtype, {"error": str(e)[:200]}))
                task_idx += 1

        launch_next()

        while running or task_idx < len(pending):
            finished = []
            for iid, proc in list(running.items()):
                if proc.poll() is not None:
                    finished.append(iid)

            for iid in finished:
                proc = running.pop(iid)
                active_count -= 1
                wtype = None
                result_file = None
                for t in pending:
                    if t[0] == iid:
                        wtype = t[1]
                        result_file = t[5]
                        break

                proc_metrics = self._collect_process_metrics(proc.pid) if proc.pid else {}
                result_data = self._read_result(result_file) if result_file else {}
                result_data.update(proc_metrics)
                result_data["pid"] = proc.pid
                result_data["exit_code"] = proc.returncode
                completed.append((iid, wtype, result_data))

                elapsed_s = result_data.get("total_latency_ms", 0) / 1000
                status = "OK" if not result_data.get("error") else f"ERR: {result_data['error'][:50]}"
                tools = result_data.get("total_tool_calls", "?")
                threads = proc_metrics.get("num_threads", "?")
                print(f"  [{len(completed):2d}/{len(tasks)}] #{iid:2d} {wtype:<8} "
                      f"{elapsed_s:6.1f}s  tools={tools}  threads={threads}  {status}")

                launch_next()

            if running:
                time.sleep(0.3)

        duration = time.time() - start_time
        self.monitor.stop()

        try:
            worker_script.unlink()
        except Exception:
            pass

        self._aggregate_and_save(completed, duration, len(tasks))

    def _aggregate_and_save(self, completed, duration, num_tasks):
        samples = self.monitor.cpu_samples
        cpu_avg = round(statistics.mean(samples), 2) if samples else 0
        cpu_max = round(max(samples), 2) if samples else 0
        cpu_min = round(min(samples), 2) if samples else 0
        cpu_stddev = round(statistics.stdev(samples), 2) if len(samples) > 1 else 0

        results = []
        total_vol = 0
        total_invol = 0
        total_threads = 0
        by_type: Dict[str, list] = {}

        for iid, wtype, data in sorted(completed, key=lambda x: x[0]):
            data["instance_id"] = iid
            data["workload_type"] = wtype
            results.append(data)
            by_type.setdefault(wtype, []).append(data)
            total_vol += data.get("ctx_switches_vol", 0)
            total_invol += data.get("ctx_switches_invol", 0)
            total_threads += data.get("num_threads", 0) if isinstance(data.get("num_threads"), int) else 0

        print(f"\n{'='*78}")
        print(f"  TEST RESULTS")
        print(f"{'='*78}")

        print(f"\n  CPU Core {self.core_id} Utilization:")
        print(f"    Average : {cpu_avg:>6.1f}%")
        print(f"    Maximum : {cpu_max:>6.1f}%")
        print(f"    Minimum : {cpu_min:>6.1f}%")
        print(f"    StdDev  : {cpu_stddev:>6.1f}%")

        print(f"\n  Process / Thread Overhead:")
        print(f"    Total Processes   : {len(completed)}")
        print(f"    Total Threads     : {total_threads}")
        print(f"    Avg Threads/Proc  : {total_threads/max(1,len(completed)):.1f}")
        print(f"    Ctx Switches (vol)   : {total_vol:,}")
        print(f"    Ctx Switches (invol) : {total_invol:,}")
        print(f"    Ctx Switch Overhead  : {total_vol+total_invol:,} total "
              f"({(total_vol+total_invol)/max(1,duration):.0f}/sec)")

        print(f"\n  Per-Instance Results:")
        print(f"  {'ID':>3} {'Type':<8} {'Time(s)':>8} {'Turns':>5} {'Tools':>5} "
              f"{'Thr':>4} {'PID':>6} {'LLM(s)':>7} {'Tool(s)':>7} {'Fwk(s)':>7} {'Error':<20}")
        print(f"  {'---':>3} {'--------':<8} {'--------':>8} {'-----':>5} {'-----':>5} "
              f"{'----':>4} {'------':>6} {'-------':>7} {'-------':>7} {'-------':>7} {'--------------------':<20}")

        for r in results:
            err_display = r.get("error", "")[:20] if r.get("error") else ""
            print(f"  {r['instance_id']:3d} {r['workload_type']:<8} "
                  f"{r.get('total_latency_ms',0)/1000:8.1f} {r.get('num_turns',0):5d} "
                  f"{r.get('total_tool_calls',0):5d} {r.get('num_threads','?'):>4} "
                  f"{r.get('pid','?'):>6} "
                  f"{r.get('total_llm_latency_ms',0)/1000:7.1f} "
                  f"{r.get('total_tool_latency_ms',0)/1000:7.1f} "
                  f"{r.get('total_framework_latency_ms',0)/1000:7.1f} "
                  f"{err_display:<20}")

        if by_type:
            print(f"\n  Aggregated by Workload Type:")
            print(f"  {'Type':<8} {'Count':>5} {'AvgTime':>8} {'AvgTools':>8} "
                  f"{'AvgThr':>6} {'AvgLLM':>8} {'AvgTool':>8} {'AvgFwk':>8} {'Errors':>6}")
            print(f"  {'--------':<8} {'-----':>5} {'--------':>8} {'--------':>8} "
                  f"{'------':>6} {'--------':>8} {'--------':>8} {'--------':>8} {'------':>6}")
            for wtype, group in sorted(by_type.items()):
                avg_time = statistics.mean([r.get("total_latency_ms", 0) / 1000 for r in group])
                avg_tools = statistics.mean([r.get("total_tool_calls", 0) for r in group])
                thr_vals = [r.get("num_threads", 0) for r in group if isinstance(r.get("num_threads"), int)]
                avg_thr = statistics.mean(thr_vals) if thr_vals else 0
                avg_llm = statistics.mean([r.get("total_llm_latency_ms", 0) / 1000 for r in group])
                avg_tool = statistics.mean([r.get("total_tool_latency_ms", 0) / 1000 for r in group])
                avg_fwk = statistics.mean([r.get("total_framework_latency_ms", 0) / 1000 for r in group])
                errors = sum(1 for r in group if r.get("error"))
                print(f"  {wtype:<8} {len(group):5d} {avg_time:8.1f} {avg_tools:8.1f} "
                      f"{avg_thr:6.1f} {avg_llm:8.2f} {avg_tool:8.2f} {avg_fwk:8.2f} {errors:6d}")

        all_lats = [r.get("total_latency_ms", 0) for r in results if not r.get("error")]
        if all_lats:
            sorted_lats = sorted(all_lats)
            print(f"\n  Task Latency Distribution (successful):")
            print(f"    Avg  : {statistics.mean(all_lats)/1000:.1f}s")
            print(f"    P50  : {sorted_lats[len(sorted_lats)//2]/1000:.1f}s")
            print(f"    P95  : {sorted_lats[int(len(sorted_lats)*0.95)]/1000:.1f}s")
            print(f"    Max  : {max(all_lats)/1000:.1f}s")

        total_tok = sum(r.get("total_prompt_tokens", 0) + r.get("total_completion_tokens", 0) for r in results)
        print(f"\n  Total Tokens Consumed: {total_tok:,}")
        print(f"  Throughput: {len(results)/max(1,duration):.2f} tasks/sec")

        timestamp = time.strftime("%Y%m%d_%H%M%S")
        result_path = RESULTS_DIR / f"oversub_agent_{timestamp}.json"

        output = {
            "test_config": {
                "core_id": self.core_id,
                "concurrency": self.concurrency,
                "tasks_per_type": self.tasks_per_type,
                "task_types": self.task_types,
                "num_instances": num_tasks,
                "duration_sec": round(duration, 2),
                "mode": "multiprocess",
            },
            "global": {
                "cpu_avg": cpu_avg,
                "cpu_max": cpu_max,
                "cpu_min": cpu_min,
                "cpu_stddev": cpu_stddev,
                "cpu_samples": samples,
                "total_ctx_switches_vol": total_vol,
                "total_ctx_switches_invol": total_invol,
                "total_threads": total_threads,
                "total_processes": len(completed),
                "ctx_switches_per_sec": round((total_vol + total_invol) / max(1, duration), 1),
            },
            "per_instance": results,
        }

        with open(result_path, "w") as f:
            json.dump(output, f, indent=2, default=str)
        print(f"\n  Results saved: {result_path}")
        print(f"{'='*78}\n")


def main():
    parser = argparse.ArgumentParser(description="CPU Oversubscription Load Test (Multi-Process)")
    parser.add_argument("-c", "--core", type=int, default=0, help="CPU core to pin (default: 0)")
    parser.add_argument("-n", "--concurrency", type=int, default=10, help="Max concurrent processes (default: 10)")
    parser.add_argument("-t", "--tasks-per-type", type=int, default=3, help="Tasks per workload type (default: 3)")
    parser.add_argument("--types", type=str, default="cpu,io,network,mixed", help="Workload types (comma-separated)")
    args = parser.parse_args()

    task_types = [t.strip() for t in args.types.split(",") if t.strip()]
    test = OversubscriptionTest(
        core_id=args.core,
        concurrency=args.concurrency,
        tasks_per_type=args.tasks_per_type,
        task_types=task_types,
    )
    test.run()


if __name__ == "__main__":
    main()
