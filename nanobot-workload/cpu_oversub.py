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
MOCK_LLM = os.environ.get("NANOBOT_MOCK_LLM", "0") == "1"
MOCK_DELAY_MS = int(os.environ.get("NANOBOT_MOCK_DELAY_MS", "50"))

sys.path.insert(0, BASE_DIR)
os.environ["HOME"] = BASE_DIR

from nanobot import Nanobot
from nanobot.agent.hook import AgentHook, AgentHookContext

if MOCK_LLM:
    from nanobot.providers.base import LLMProvider, LLMResponse, ToolCallRequest, GenerationSettings

    class MockLLMProvider(LLMProvider):
        """Mock LLM that returns scripted tool calls without network I/O."""

        TOOL_SEQUENCES = {
            "cpu": [
                [ToolCallRequest(id="call_mock_0", name="write_file", arguments={"path": "workspace/mock_code.py", "content": "import sys\\nprint('mock cpu task')\\nfor i in range(1000): x=i**2"})],
                [ToolCallRequest(id="call_mock_0", name="exec", arguments={"command": "python3 workspace/mock_code.py"})],
                [ToolCallRequest(id="call_mock_0", name="read_file", arguments={"path": "workspace/mock_code.py"})],
            ],
            "io": [
                [ToolCallRequest(id="call_mock_0", name="write_file", arguments={"path": "workspace/mock_data.txt", "content": "name,age,city\\nAlice,30,NYC\\nBob,25,LA\\nCarol,35,Chicago"})],
                [ToolCallRequest(id="call_mock_0", name="read_file", arguments={"path": "workspace/mock_data.txt"})],
                [ToolCallRequest(id="call_mock_0", name="exec", arguments={"command": "ls -la workspace/ && wc -l workspace/mock_data.txt"})],
            ],
            "network": [
                [ToolCallRequest(id="call_mock_0", name="write_file", arguments={"path": "workspace/mock_search.md", "content": "# Mock Search Results\\n- Result 1: Python asyncio guide\\n- Result 2: Best practices"})],
                [ToolCallRequest(id="call_mock_0", name="read_file", arguments={"path": "workspace/mock_search.md"})],
                [ToolCallRequest(id="call_mock_0", name="exec", arguments={"command": "cat workspace/mock_search.md | grep -i python"})],
            ],
            "mixed": [
                [ToolCallRequest(id="call_mock_0", name="write_file", arguments={"path": "workspace/mock_mixed.py", "content": "def quicksort(arr):\\n    if len(arr) <= 1: return arr\\n    pivot = arr[0]\\n    left = [x for x in arr[1:] if x <= pivot]\\n    right = [x for x in arr[1:] if x > pivot]\\n    return quicksort(left) + [pivot] + quicksort(right)"})],
                [ToolCallRequest(id="call_mock_0", name="exec", arguments={"command": "python3 -c \\"from workspace.mock_mixed import quicksort; print(quicksort([3,1,4,1,5,9,2,6]))\\""})],
                [ToolCallRequest(id="call_mock_0", name="read_file", arguments={"path": "workspace/mock_mixed.py"})],
            ],
        }

        def __init__(self):
            self._turn = 0
            self._sequence = None
            self._model = "mock-llm"
            self.generation = GenerationSettings(max_tokens=2048, temperature=0.3)

        def get_model(self):
            return self._model

        def get_default_model(self):
            return self._model

        async def chat(self, messages, tools=None, **kwargs):
            await asyncio.sleep(MOCK_DELAY_MS / 1000)
            self._turn += 1
            if self._sequence is None:
                for wt, seq in self.TOOL_SEQUENCES.items():
                    for msg in messages:
                        if wt in msg.get("content", "").lower():
                            self._sequence = seq
                            break
                    if self._sequence:
                        break
                if self._sequence is None:
                    self._sequence = self.TOOL_SEQUENCES["io"]

            if self._turn <= len(self._sequence):
                tool_calls = self._sequence[self._turn - 1]
                return LLMResponse(
                    content="",
                    tool_calls=tool_calls,
                    usage={"prompt_tokens": 800, "completion_tokens": 150},
                    finish_reason="tool_calls",

                )
            return LLMResponse(
                content="Task completed successfully. All operations finished.",
                tool_calls=[],
                usage={"prompt_tokens": 800, "completion_tokens": 100},
                finish_reason="stop",

            )

        async def chat_stream(self, messages, tools=None, **kwargs):
            result = await self.chat(messages, tools, **kwargs)
            yield result

        async def chat_with_retry(self, messages, tools=None, retry_mode="standard", on_retry_wait=None, **kwargs):
            return await self.chat(messages, tools, **kwargs)

        async def chat_stream_with_retry(self, messages, tools=None, on_content_delta=None, **kwargs):
            result = await self.chat(messages, tools, **kwargs)
            if on_content_delta and result.content:
                await on_content_delta(result.content)
            yield result

        async def _safe_chat(self, **kwargs):
            return await self.chat(**kwargs)

        _SENTINEL = object()


class MetricsHook(AgentHook):
    def __init__(self):
        super().__init__(reraise=False)
        self._t0 = 0.0
        self._t1 = 0.0
        self._has_tools = False
        self._last_t2 = 0.0
        self.turns = []
        self.tool_calls_total = 0
        self.prompt_tokens_total = 0
        self.completion_tokens_total = 0
        self._turn_idx = 0
        self.inter_turn_gaps = []

    def _now(self):
        return time.perf_counter()

    async def before_iteration(self, context):
        self._t0 = self._now()
        self._t1 = 0.0
        self._has_tools = False
        if self._last_t2 > 0:
            gap_ms = (self._t0 - self._last_t2) * 1000
            self.inter_turn_gaps.append(gap_ms)

    async def before_execute_tools(self, context):
        self._t1 = self._now()
        self._has_tools = True

    async def after_iteration(self, context):
        t2 = self._now()
        self._last_t2 = t2
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
        "mock_llm": MOCK_LLM,
    }

    t_init_start = time.perf_counter()
    try:
        bot = Nanobot.from_config(config_path=config_path)
        if MOCK_LLM:
            mock_provider = MockLLMProvider()
            bot._loop.provider = mock_provider
            bot._loop.runner.provider = mock_provider
            import sys as _sys
            print(f"  [MOCK] Provider replaced: {type(bot._loop.provider).__name__}", file=_sys.stderr)
    except Exception as e:
        result_data["error"] = f"InitError: {type(e).__name__}: {str(e)[:200]}"
        traceback.print_exc()
        with open(result_file, "w") as f:
            json.dump(result_data, f, indent=2)
        return
    init_ms = (time.perf_counter() - t_init_start) * 1000

    t_run_start = time.perf_counter()
    import threading as _threading
    _mem_samples = {"rss": [], "vms": []}
    _mem_stop = _threading.Event()

    def _sample_mem():
        _pid = os.getpid()
        while not _mem_stop.is_set():
            try:
                with open("/proc/self/status") as _sf:
                    for _line in _sf:
                        if _line.startswith("VmRSS:"):
                            _mem_samples["rss"].append(int(_line.split()[1]) * 1024)
                        elif _line.startswith("VmSize:"):
                            _mem_samples["vms"].append(int(_line.split()[1]) * 1024)
            except Exception:
                pass
            _mem_stop.wait(0.5)

    _mem_thread = _threading.Thread(target=_sample_mem, daemon=True)
    _mem_thread.start()

    try:
        result = await bot.run(task_prompt, session_key=f"oversub:{instance_id}", hooks=[hook])
        result_data["response_preview"] = result.content[:500] if result.content else ""
    except Exception as e:
        result_data["error"] = f"{type(e).__name__}: {str(e)[:200]}"
        traceback.print_exc()
    _mem_stop.set()
    _mem_thread.join(timeout=2)
    run_ms = (time.perf_counter() - t_run_start) * 1000
    elapsed_ms = init_ms + run_ms

    total_llm = sum(t["llm_latency_ms"] for t in hook.turns)
    total_tool = sum(t["tool_latency_ms"] for t in hook.turns)
    total_fwk_run = max(0, run_ms - total_llm - total_tool)

    try:
        with open("/proc/self/status") as _sf:
            _status = _sf.read()
        _vmrss = 0
        _vmhwm = 0
        _threads = 0
        for _line in _status.split("\n"):
            if _line.startswith("VmRSS:"):
                _vmrss = int(_line.split()[1])
            elif _line.startswith("VmHWM:"):
                _vmhwm = int(_line.split()[1])
            elif _line.startswith("Threads:"):
                _threads = int(_line.split()[1])
        rss_peak_mb = round(max(_mem_samples["rss"]) / 1024 / 1024, 1) if _mem_samples["rss"] else round(_vmrss / 1024, 1)
        rss_avg_mb = round(sum(_mem_samples["rss"]) / max(1, len(_mem_samples["rss"])) / 1024 / 1024, 1) if _mem_samples["rss"] else 0
        vms_peak_mb = round(max(_mem_samples["vms"]) / 1024 / 1024, 1) if _mem_samples["vms"] else 0
        result_data["rss_mb"] = rss_peak_mb
        result_data["rss_avg_mb"] = rss_avg_mb
        result_data["vms_mb"] = vms_peak_mb
        result_data["rss_hwm_mb"] = round(_vmhwm / 1024, 1)
        result_data["num_threads"] = _threads
        result_data["rss_final_kb"] = _vmrss
    except Exception as e:
        result_data["mem_error"] = f"{type(e).__name__}: {str(e)[:100]}"

    try:
        with open("/proc/self/smaps_rollup") as _sf:
            _smaps = _sf.read()
        _private_dirty = 0
        _private_clean = 0
        _shared_clean = 0
        _shared_dirty = 0
        for _line in _smaps.split("\n"):
            if _line.startswith("Private_Dirty:"):
                _private_dirty += int(_line.split()[1])
            elif _line.startswith("Private_Clean:"):
                _private_clean += int(_line.split()[1])
            elif _line.startswith("Shared_Clean:"):
                _shared_clean += int(_line.split()[1])
            elif _line.startswith("Shared_Dirty:"):
                _shared_dirty += int(_line.split()[1])
        result_data["mem_hot_kb"] = _private_dirty
        result_data["mem_warm_kb"] = _private_clean
        result_data["mem_cold_kb"] = _shared_clean + _shared_dirty
    except Exception as e:
        result_data["smaps_error"] = f"{type(e).__name__}: {str(e)[:100]}"

    try:
        with open("/proc/self/io") as _sf:
            _io = _sf.read()
        for _line in _io.split("\n"):
            if _line.startswith("read_bytes:"):
                result_data["io_read_bytes"] = int(_line.split()[1])
            elif _line.startswith("write_bytes:"):
                result_data["io_write_bytes"] = int(_line.split()[1])
    except Exception:
        pass

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
        "inter_turn_gaps_ms": [round(g, 2) for g in hook.inter_turn_gaps],
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
        mock_llm: bool = False,
        mock_delay_ms: int = 50,
    ):
        self.core_id = core_id
        self.concurrency = concurrency
        self.tasks_per_type = tasks_per_type
        self.task_types = task_types or ["cpu", "io", "network", "mixed"]
        self.mock_llm = mock_llm
        self.mock_delay_ms = mock_delay_ms
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

    def _parse_perf_stat(self, perf_file: Path) -> dict:
        result = {}
        with open(perf_file) as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith("#") or "seconds" in line:
                    continue
                parts = line.split(",")
                if len(parts) >= 2:
                    try:
                        val = int(parts[0].replace(",", "").strip())
                        event = parts[-1].strip()
                        result[event] = val
                    except ValueError:
                        pass
        return result

    def _collect_process_metrics(self, pid: int, proc_obj=None) -> dict:
        if not psutil:
            return {}
        try:
            if proc_obj is None:
                proc = psutil.Process(pid)
            else:
                proc = proc_obj
            mem = proc.memory_info()
            ctx = proc.num_ctx_switches()
            result = {
                "num_threads": proc.num_threads(),
                "ctx_switches_vol": ctx.voluntary,
                "ctx_switches_invol": ctx.involuntary,
                "rss_mb": round(mem.rss / 1024 / 1024, 1),
                "vms_mb": round(mem.vms / 1024 / 1024, 1),
            }
            try:
                mmaps = proc.memory_maps(grouped=True)
                result["mem_hot_kb"] = sum(getattr(m, 'private_dirty', 0) for m in mmaps)
                result["mem_warm_kb"] = sum(getattr(m, 'private_clean', 0) for m in mmaps)
                result["mem_cold_kb"] = sum(getattr(m, 'shared_clean', 0) + getattr(m, 'shared_dirty', 0) for m in mmaps)
            except Exception:
                pass
            return result
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
        if self.mock_llm:
            print(f"  Mock LLM      : ENABLED (delay={self.mock_delay_ms}ms)")
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

        perf_events = (
            "cache-references,cache-misses,"
            "L1-dcache-loads,L1-dcache-load-misses,"
            "armv8_pmuv3_0/l2d_cache/,armv8_pmuv3_0/l2d_cache_refill/,"
            "armv8_pmuv3_0/bus_access/,armv8_pmuv3_0/mem_access/"
        )
        perf_proc = None
        perf_output_file = BASE_DIR / "_perf_stat.txt"
        try:
            perf_proc = subprocess.Popen(
                ["perf", "stat", "-e", perf_events,
                 "-C", str(self.core_id), "-x", ",",
                 "-o", str(perf_output_file),
                 "sleep", str(max(1, int(self.tasks_per_type * 5 + 120)))],
                stdout=subprocess.PIPE, stderr=subprocess.PIPE,
            )
        except Exception:
            pass

        start_time = time.time()

        print(f"  Launching {len(tasks)} agent processes (max {self.concurrency} concurrent)...\n")

        running: Dict[int, subprocess.Popen] = {}
        running_procs: Dict[int, psutil.Process] = {}
        running_mem_samples: Dict[int, List[dict]] = {}
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
                if self.mock_llm:
                    env["NANOBOT_MOCK_LLM"] = "1"
                    env["NANOBOT_MOCK_DELAY_MS"] = str(self.mock_delay_ms)
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
                        running_procs[iid] = p
                        running_mem_samples[iid] = []
                        try:
                            mi = p.memory_info()
                            running_mem_samples[iid].append({"rss_mb": round(mi.rss / 1024 / 1024, 1), "vms_mb": round(mi.vms / 1024 / 1024, 1)})
                        except Exception:
                            pass
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
                elif iid in running_procs:
                    try:
                        p = running_procs[iid]
                        mi = p.memory_info()
                        running_mem_samples[iid].append({
                            "rss_mb": round(mi.rss / 1024 / 1024, 1),
                            "vms_mb": round(mi.vms / 1024 / 1024, 1),
                        })
                    except Exception:
                        pass

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

                result_data = self._read_result(result_file) if result_file else {}

                proc_metrics = {}
                mem_samples = running_mem_samples.pop(iid, [])
                if mem_samples:
                    rss_vals = [s["rss_mb"] for s in mem_samples]
                    vms_vals = [s["vms_mb"] for s in mem_samples]
                    proc_metrics["rss_peak_mb"] = max(rss_vals)
                    proc_metrics["rss_avg_mb"] = round(statistics.mean(rss_vals), 1)
                    proc_metrics["vms_peak_mb"] = max(vms_vals)
                    proc_metrics["mem_samples"] = len(mem_samples)

                try:
                    proc_obj = running_procs.pop(iid, None)
                    if proc_obj:
                        ctx = proc_obj.num_ctx_switches()
                        proc_metrics["num_threads"] = proc_obj.num_threads()
                        proc_metrics["ctx_switches_vol"] = ctx.voluntary
                        proc_metrics["ctx_switches_invol"] = ctx.involuntary
                        try:
                            mmaps = proc_obj.memory_maps(grouped=True)
                            proc_metrics["mem_hot_kb"] = sum(getattr(m, 'private_dirty', 0) for m in mmaps)
                            proc_metrics["mem_warm_kb"] = sum(getattr(m, 'private_clean', 0) for m in mmaps)
                            proc_metrics["mem_cold_kb"] = sum(getattr(m, 'shared_clean', 0) + getattr(m, 'shared_dirty', 0) for m in mmaps)
                        except Exception:
                            pass
                        try:
                            with open(f"/proc/{proc_obj.pid}/io") as _iof:
                                for _line in _iof:
                                    if _line.startswith("read_bytes:"):
                                        proc_metrics["io_read_bytes"] = int(_line.split()[1])
                                    elif _line.startswith("write_bytes:"):
                                        proc_metrics["io_write_bytes"] = int(_line.split()[1])
                        except Exception:
                            pass
                except Exception:
                    pass

                for k, v in proc_metrics.items():
                    if k not in result_data or (isinstance(v, (int, float)) and v > 0):
                        result_data[k] = v
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

        perf_cache = {}
        if perf_proc and perf_proc.poll() is None:
            perf_proc.terminate()
            perf_proc.wait(timeout=5)
        if perf_output_file.exists():
            try:
                perf_cache = self._parse_perf_stat(perf_output_file)
                perf_output_file.unlink()
            except Exception:
                pass

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

        if perf_cache:
            print(f"\n  Cache Performance (core {self.core_id}):")
            cr = perf_cache.get("cache-references", 0)
            cm = perf_cache.get("cache-misses", 0)
            l1l = perf_cache.get("L1-dcache-loads", 0)
            l1m = perf_cache.get("L1-dcache-load-misses", 0)
            l2a = perf_cache.get("armv8_pmuv3_0/l2d_cache/", 0)
            l2r = perf_cache.get("armv8_pmuv3_0/l2d_cache_refill/", 0)
            bus = perf_cache.get("armv8_pmuv3_0/bus_access/", 0)
            mem = perf_cache.get("armv8_pmuv3_0/mem_access/", 0)
            print(f"    Cache References  : {cr:>15,}")
            print(f"    Cache Misses      : {cm:>15,}  ({cm/max(1,cr)*100:.2f}%)")
            print(f"    L1-dcache Loads   : {l1l:>15,}")
            print(f"    L1-dcache Misses  : {l1m:>15,}  ({l1m/max(1,l1l)*100:.2f}%)")
            print(f"    L2 Cache Accesses : {l2a:>15,}")
            print(f"    L2 Cache Refills  : {l2r:>15,}  ({l2r/max(1,l2a)*100:.2f}%)")
            print(f"    Bus Accesses      : {bus:>15,}")
            print(f"    Memory Accesses   : {mem:>15,}")
            print(f"    Cache Miss Rate   : {cm/max(1,cr)*100:.2f}%  (higher = more contention)")
            print(f"    L2 Miss Rate      : {l2r/max(1,l2a)*100:.2f}%  (higher = more L3/DRAM pressure)")

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
              f"{'Thr':>4} {'PID':>6} {'RSS_MB':>7} {'LLM(s)':>7} {'Tool(s)':>7} {'Fwk(s)':>7} {'Error':<20}")
        print(f"  {'---':>3} {'--------':<8} {'--------':>8} {'-----':>5} {'-----':>5} "
              f"{'----':>4} {'------':>6} {'-------':>7} {'-------':>7} {'-------':>7} {'-------':>7} {'--------------------':<20}")

        for r in results:
            err_display = r.get("error", "")[:20] if r.get("error") else ""
            rss = r.get("rss_mb", "-")
            print(f"  {r['instance_id']:3d} {r['workload_type']:<8} "
                  f"{r.get('total_latency_ms',0)/1000:8.1f} {r.get('num_turns',0):5d} "
                  f"{r.get('total_tool_calls',0):5d} {r.get('num_threads','?'):>4} "
                  f"{r.get('pid','?'):>6} {rss:>7} "
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

        rss_vals = [r.get("rss_peak_mb", r.get("rss_mb", 0)) for r in results
                    if isinstance(r.get("rss_peak_mb"), (int, float)) or isinstance(r.get("rss_mb"), (int, float))]
        vms_vals = [r.get("vms_peak_mb", r.get("vms_mb", 0)) for r in results
                    if isinstance(r.get("vms_peak_mb"), (int, float)) or isinstance(r.get("vms_mb"), (int, float))]
        if rss_vals:
            print(f"\n  Memory Usage:")
            print(f"    Avg RSS/Proc    : {statistics.mean(rss_vals):.1f} MB")
            print(f"    Max RSS/Proc    : {max(rss_vals):.1f} MB")
            print(f"    Total RSS       : {sum(rss_vals):.1f} MB")
            if vms_vals:
                print(f"    Avg VMS/Proc    : {statistics.mean(vms_vals):.1f} MB")
                print(f"    Total VMS       : {sum(vms_vals):.1f} MB")

        hot_vals = [r.get("mem_hot_kb", 0) for r in results if isinstance(r.get("mem_hot_kb"), (int, float))]
        warm_vals = [r.get("mem_warm_kb", 0) for r in results if isinstance(r.get("mem_warm_kb"), (int, float))]
        cold_vals = [r.get("mem_cold_kb", 0) for r in results if isinstance(r.get("mem_cold_kb"), (int, float))]
        if hot_vals and sum(hot_vals) > 0:
            total_hot = sum(hot_vals)
            total_warm = sum(warm_vals) if warm_vals else 0
            total_cold = sum(cold_vals) if cold_vals else 0
            total_all = total_hot + total_warm + total_cold
            print(f"    Memory Hot/Warm/Cold (per-process /proc/PID/smaps):")
            print(f"      Hot  (private_dirty)   : {total_hot:>10,} KB ({total_hot/max(1,total_all)*100:5.1f}%)")
            print(f"      Warm (private_clean)   : {total_warm:>10,} KB ({total_warm/max(1,total_all)*100:5.1f}%)")
            print(f"      Cold (shared_*)        : {total_cold:>10,} KB ({total_cold/max(1,total_all)*100:5.1f}%)")

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
                "mock_llm": self.mock_llm,
                "mock_delay_ms": self.mock_delay_ms if self.mock_llm else 0,
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
                "cache_perf": perf_cache if perf_cache else {},
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
    parser.add_argument("--mock-llm", action="store_true", help="Mock LLM (no network I/O, isolate CPU overhead)")
    parser.add_argument("--mock-delay", type=int, default=50, help="Mock LLM delay per turn in ms (default: 50)")
    args = parser.parse_args()

    task_types = [t.strip() for t in args.types.split(",") if t.strip()]
    test = OversubscriptionTest(
        core_id=args.core,
        concurrency=args.concurrency,
        tasks_per_type=args.tasks_per_type,
        task_types=task_types,
        mock_llm=args.mock_llm,
        mock_delay_ms=args.mock_delay,
    )
    test.run()


if __name__ == "__main__":
    main()
