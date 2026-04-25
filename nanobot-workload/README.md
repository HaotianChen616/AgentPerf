# CPU 超分负载测试工具

基于 **nanobot-ai** 的 CPU 超分场景负载分析测试框架。在单个 CPU 核心上启动多个 AI Agent 实例，执行真实任务，观测 CPU 利用率和时延变化，支持每轮时延分解（LLM / 工具 / 框架）。

## 目录结构

```
nanobot-workload/
├── cpu_oversub.py          # 主测试脚本
├── analyze_results.py      # 结果分析 & 可视化脚本
├── requirements.txt        # Python 依赖
└── README.md               # 本文档
```

运行后自动生成：

```
nanobot-workload/
├── .nanobot/
│   └── config.json         # nanobot 配置（模型、API Key、工具开关）
├── workspace/              # Agent 工作目录
└── results/                # 测试结果 JSON 输出目录
```

## 快速开始

### 1. 环境要求

- Python 3.10+
- pip
- 网络访问（调用 LLM API、DuckDuckGo 搜索）

### 2. 安装

```bash
cd nanobot-workload
pip install -r requirements.txt
```

> 如果你的 Python 环境有 pydantic v2 冲突，可以使用 venv：
> ```bash
> python3 -m venv .venv && source .venv/bin/activate
> pip install -r requirements.txt
> ```

### 3. 配置

首次运行会自动生成 `.nanobot/config.json`，请修改其中的模型和 API 配置：

```json
{
  "agents": {
    "defaults": {
      "model": "<your-model>",
      "provider": "custom",
      "max_tokens": 2048,
      "max_tool_iterations": 10,
      "temperature": 0.3
    }
  },
  "providers": {
    "custom": {
      "api_key": "<your-api-key>",
      "api_base": "<your-api-base-url>"
    }
  }
}
```

- `provider` 设为 `custom` 即可使用任何 OpenAI 兼容 API
- `max_tool_iterations` 控制 Agent 每任务最大迭代轮数
- `tools.web.enable` / `tools.exec.enable` 控制工具是否可用

### 4. 运行测试

```bash
# 默认：4种类型各3个任务，共12个实例，绑定核心0，最大并发10
python3 cpu_oversub.py

# 仅测试 io 类型，2个任务，绑定核心3
python3 cpu_oversub.py -c 3 -t 2 --types io

# 高并发压测：8个并发，每种类型5个任务
python3 cpu_oversub.py -n 8 -t 5
```

### 5. 分析结果

```bash
# 分析最新一次结果
python3 analyze_results.py

# 分析指定文件
python3 analyze_results.py results/oversub_agent_20260423_211320.json

# 对比多次运行结果
python3 analyze_results.py --all --compare
```

## 参数说明

### cpu_oversub.py

| 参数 | 缩写 | 默认值 | 说明 |
|------|------|--------|------|
| `--core` | `-c` | 0 | 绑定的 CPU 核心编号 |
| `--concurrency` | `-n` | 10 | 最大并发 Agent 数 |
| `--tasks-per-type` | `-t` | 3 | 每种负载类型的任务数 |
| `--types` | | `cpu,io,network,mixed` | 负载类型（逗号分隔） |

### analyze_results.py

| 参数 | 缩写 | 说明 |
|------|------|------|
| `files` | 位置参数 | 指定结果 JSON 文件 |
| `--all` | `-a` | 分析 results/ 下所有结果 |
| `--latest` | `-l` | 仅分析最新结果（默认行为） |
| `--compare` | | 多文件对比模式 |

## 负载类型

不同负载类型通过**提示词引导** Agent 使用不同工具，LLM 自主决定实际调用：

| 类型 | 任务特征 | 预期触发的工具 |
|------|----------|---------------|
| `cpu` | 代码生成、算法实现、数学计算 | write_file, exec |
| `io` | 文件读写、目录扫描、内容搜索 | read_file, write_file, grep, glob, list_dir |
| `network` | 网络搜索、信息获取 | web_search, web_fetch, write_file |
| `mixed` | 混合多种工具 | 以上全部 |

## 结果解读

### 终端输出模块

分析脚本输出包含以下可视化模块：

#### 1. CPU 利用率时间线

```
▌ CPU Utilization Timeline (Core 0)
  Avg: 15.9%  Max: 89.7%  Min: 7.1%  StdDev: 20.4%
  █▂▁▁▁▁▁▁▁▁▂▁▁▁▁▁
  0%                                100%
```

Sparkline 展示 CPU 使用率随时间变化，高负载时应观察到持续高位。

#### 2. 时延分解堆叠图（核心功能）

```
▌ Latency Breakdown — LLM / Tool / Framework
  ██=LLM  ██=Tool  ██=Framework
  # 0 io       │████████████████████████████████████████████░░│  15.1s   93%/  1%/  6%
```

- 紫色=LLM 时延，黄色=工具时延，蓝色=框架时延
- 右侧百分比显示三者占比
- **LLM 时延**：模型 API 往返时间（通常占比最大）
- **工具时延**：工具执行时间（exec 较慢，read_file/write_file 很快）
- **框架时延**：上下文治理、消息构建、checkpoint 等框架开销

#### 3. 每轮时延分解表

```
▌ Per-Turn Latency Decomposition
  # 0 io
    Turn   Total     LLM    Tool     Fwk  LLM% Tool% Tools Used
       0    7.62    7.62    0.00    0.00 100.0   0.0 write_file
       1    2.13    1.96    0.17    0.00  92.1   7.9 exec
       2    1.90    1.89    0.00    0.00  99.8   0.2 read_file
       3    2.52    2.52    0.00    0.00 100.0   0.0 -
```

- 每一轮迭代的精确时延分解（单位：秒）
- `Tools Used` 列显示该轮调用的工具名
- 最后一轮通常无工具调用（Agent 给出最终回复）

#### 4. 按负载类型聚合

```
▌ Latency Breakdown by Workload Type
  Type       AvgLLM  AvgTool   AvgFwk AvgTotal   LLM%  Tool%   Fwk%
  io          14.00     0.17     0.96    15.13  92.5%   1.2%   6.3%
```

#### 5. Summary

```
▌ Summary
  Total Tokens       : 42,732
  Throughput         : 0.07 tasks/sec
  Avg Task Latency   : 15.1s
  Latency Breakdown:
    Avg LLM       :   14.0s (92.5%)
    Avg Tool      :    0.2s ( 1.2%)
    Avg Framework :    1.0s ( 6.3%)
```

### JSON 结果文件

结果保存在 `results/oversub_agent_<timestamp>.json`，结构如下：

```json
{
  "test_config": { ... },
  "global": {
    "cpu_avg": 15.91,
    "cpu_max": 89.7,
    "cpu_samples": [89.7, 23.8, ...],
    "ctx_switches_delta": 53
  },
  "per_instance": [
    {
      "instance_id": 0,
      "workload_type": "io",
      "total_latency_ms": 15131.02,
      "total_llm_latency_ms": 13997.55,
      "total_tool_latency_ms": 174.07,
      "total_framework_latency_ms": 959.40,
      "turns": [
        {
          "turn_index": 0,
          "latency_ms": 7624.88,
          "llm_latency_ms": 7623.83,
          "tool_latency_ms": 1.05,
          "tool_calls": 1,
          "tool_names": ["write_file"],
          "prompt_tokens": 10003,
          "completion_tokens": 409
        }
      ]
    }
  ]
}
```

### 时延分解打点原理

利用 nanobot `AgentHook` 的三个生命周期回调实现：

```
迭代开始 → before_iteration(T0) → LLM 调用 → before_execute_tools(T1) → 工具执行 → after_iteration(T2)
            |                                     |                                      |
            └──── LLM 时延 = T1-T0 ──────────────└───── 工具时延 = T2-T1 ─────────────┘

框架时延 = 总墙钟时间 - Σ(LLM 时延 + 工具时延)
         （含上下文治理、消息构建、checkpoint 持久化、session 管理等）
```

## 典型分析场景

### 场景1：对比不同并发数

```bash
python3 cpu_oversub.py -n 1 -t 2
python3 cpu_oversub.py -n 5 -t 2
python3 cpu_oversub.py -n 10 -t 2
python3 analyze_results.py --all --compare
```

观察 LLM 时延在排队等待时是否增加（nanobot 默认限制 API 并发为 3）。

### 场景2：对比不同负载类型

```bash
python3 cpu_oversub.py --types cpu -t 5
python3 cpu_oversub.py --types io -t 5
python3 cpu_oversub.py --types network -t 5
python3 analyze_results.py --all --compare
```

网络类型任务的工具时延（web_search/web_fetch）通常显著高于文件 I/O。

### 场景3：多核对比

```bash
python3 cpu_oversub.py -c 0 -t 3
python3 cpu_oversub.py -c 1 -t 3
python3 analyze_results.py --all --compare
```

## 故障排除

| 问题 | 原因 | 解决方法 |
|------|------|----------|
| `ModuleNotFoundError: No module named 'xxx'` | 依赖未安装 | `pip install -r requirements.txt` |
| `ImportError: cannot import name ... from 'pydantic'` | pydantic 版本 < 2.0 | `pip install 'pydantic>=2.0'` |
| Agent 任务全部 ERR | API Key 无效或网络不通 | 检查 `.nanobot/config.json` 中的 `api_key` 和 `api_base` |
| CPU 利用率始终很低 | 并发数不够或 LLM API 响应慢 | 增大 `-n` 参数，或检查 API 延迟 |
| `web_search` 工具失败 | DuckDuckGo 搜索被限流 | 降低并发数或等待后重试 |
