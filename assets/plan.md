# OctoScout 项目计划

## 整体架构

```
┌─────────────────────────────────────────────────────────┐
│                    用户入口层                             │
│   CLI (octoscout diagnose)  │  MCP Server  │  Web UI    │
└────────────────────┬────────────────────────────────────┘
                     │
┌────────────────────▼────────────────────────────────────┐
│              Troubleshooter Agent Core                    │
│                                                          │
│  ┌──────────┐  ┌───────────┐  ┌────────────────────┐    │
│  │ 环境感知  │→│ 本地诊断   │→│ 置信度分流          │    │
│  │ Module   │  │ Module    │  │ (启发式规则+LLM)   │    │
│  └──────────┘  └───────────┘  └──┬─────────┬───────┘    │
│                                  │         │             │
│                     本地可解 ◄───┘         └──► 需检索   │
│                     直接修复                    │         │
│                                                │         │
│  ┌─────────────────────────────────────────────▼──────┐  │
│  │            Issues 检索 & 方案验证                    │  │
│  │  实时 GitHub API ──┐                               │  │
│  │                    ├──► 合并 ──► 版本感知过滤/排序  │  │
│  │  本地索引 ─────────┘        ──► 方案适用性验证     │  │
│  └───────────────────────────────────────────────────┘  │
│                                                          │
│  ┌──────────────┐  ┌──────────────┐                     │
│  │ Issue 起草    │  │ 社区回馈      │                     │
│  │ (找不到方案)  │  │ (解决后提示)  │                     │
│  └──────────────┘  └──────────────┘                     │
└─────────────────────────────────────────────────────────┘
                     │
┌────────────────────▼────────────────────────────────────┐
│              数据基础设施层                                │
│                                                          │
│  ┌─────────────────────────────────────────────────┐    │
│  │           Compatibility Matrix                    │    │
│  │                                                   │    │
│  │  数据采集 ──► LLM 结构化提取 ──► 矩阵聚合       │    │
│  │  (GitHub Issues)                  (版本×版本)     │    │
│  │                                                   │    │
│  │  输出: JSON DB + 热力图可视化                     │    │
│  └─────────────────────────────────────────────────┘    │
│                                                          │
│  ┌─────────────────────────────────────────────────┐    │
│  │           本地向量索引 (可选)                      │    │
│  │  预爬取的 issues ──► embedding ──► FAISS/Chroma  │    │
│  └─────────────────────────────────────────────────┘    │
└─────────────────────────────────────────────────────────┘
                     │
┌────────────────────▼────────────────────────────────────┐
│                  LLM Provider 层                         │
│  统一接口 ──► Claude API / OpenAI API (可切换)          │
│  Tool Use / Function Calling 统一抽象                    │
└─────────────────────────────────────────────────────────┘
```

---

## Phase 1: Agent 核心链路 (第 1-4 周)

**目标：** 跑通"输入 traceback → 输出诊断结果 + 相关 issues"的端到端链路。这个阶段结束时，应该能对场景 A（明确异常 + traceback）给出有用的诊断。

### 1.1 项目脚手架 (第 1 周)

**任务：**

- 初始化项目结构（见下方目录规划）
- 搭建 LLM Provider 抽象层：统一 Claude API 和 OpenAI API 的 tool use 接口
- 实现基础 CLI 框架（`octoscout diagnose`）
- GitHub API 封装（认证、rate limit 处理、基础 issues/search 接口）

**项目目录结构：**

```
octoscout/
├── pyproject.toml
├── README.md
├── src/
│   └── octoscout/
│       ├── __init__.py
│       ├── cli.py                    # CLI 入口 (click/typer)
│       ├── agent/
│       │   ├── __init__.py
│       │   ├── core.py               # Agent 主循环（orchestrator）
│       │   ├── tools.py              # Agent 可调用的 tool 定义
│       │   └── prompts.py            # System prompt & few-shot examples
│       ├── providers/
│       │   ├── __init__.py
│       │   ├── base.py               # LLM Provider 抽象接口
│       │   ├── claude.py             # Claude API 实现
│       │   └── openai.py             # OpenAI API 实现
│       ├── diagnosis/
│       │   ├── __init__.py
│       │   ├── env_snapshot.py       # 环境感知：读取依赖、版本
│       │   ├── traceback_parser.py   # Traceback 解析与结构化
│       │   ├── local_checker.py      # 本地诊断：API 签名检测等
│       │   └── triage.py             # 分流逻辑（启发式规则）
│       ├── search/
│       │   ├── __init__.py
│       │   ├── github_client.py      # GitHub API 封装
│       │   ├── realtime.py           # 实时检索策略
│       │   ├── local_index.py        # 本地向量索引
│       │   └── version_filter.py     # 版本感知过滤与排序
│       ├── matrix/
│       │   ├── __init__.py
│       │   ├── crawler.py            # Issues 数据采集
│       │   ├── extractor.py          # LLM 结构化提取
│       │   ├── aggregator.py         # 矩阵聚合
│       │   └── visualizer.py         # 热力图生成
│       ├── community/
│       │   ├── __init__.py
│       │   ├── issue_drafter.py      # Issue 起草
│       │   └── reply_suggester.py    # 社区回馈建议
│       └── mcp/
│           ├── __init__.py
│           └── server.py             # MCP Server 实现
├── tests/
│   ├── test_env_snapshot.py
│   ├── test_traceback_parser.py
│   ├── test_github_client.py
│   ├── test_version_filter.py
│   └── fixtures/                     # 真实 traceback 样本
│       ├── transformers_type_error.txt
│       ├── vllm_cuda_mismatch.txt
│       └── ...
├── eval/
│   ├── benchmark.py                  # 评估脚本
│   ├── cases/                        # 评估用例集
│   └── baselines/                    # Baseline 对比结果
└── data/
    └── matrix/                       # Compatibility Matrix 数据
```

**LLM Provider 抽象层设计要点：**

```python
# providers/base.py 核心接口
class LLMProvider(ABC):
    @abstractmethod
    async def chat_with_tools(
        self,
        messages: list[Message],
        tools: list[ToolDefinition],
        system: str = "",
    ) -> AgentResponse:
        """统一的 tool use 接口"""
        ...

# 关键设计决策：
# 1. Tool 定义用内部格式，各 provider 自行转换为 Claude/OpenAI 格式
# 2. AgentResponse 统一包含 text + tool_calls，屏蔽格式差异
# 3. Agent 主循环只依赖 LLMProvider 接口，不感知具体 provider
```

**验收标准：** `octoscout diagnose --help` 可运行；能通过 Claude/OpenAI 完成一次简单的 tool use 调用。

### 1.2 环境感知模块 (第 1-2 周)

**任务：**

- 实现 `env_snapshot.py`：自动检测并收集运行环境信息
- 支持多种来源：`pip list`、`requirements.txt`、`pyproject.toml`、`conda list`
- 输出结构化的 `EnvSnapshot` 对象

**需要采集的信息：**

| 信息 | 来源 | 优先级 |
|------|------|--------|
| Python 版本 | `sys.version` | P0 |
| 已安装包及版本 | `pip list --format=json` | P0 |
| CUDA 版本 | `nvidia-smi` / `torch.version.cuda` | P0 |
| cuDNN 版本 | `torch.backends.cudnn.version()` | P1 |
| OS 信息 | `platform.platform()` | P1 |
| 声明依赖 | `requirements.txt` / `pyproject.toml` | P1 |
| GPU 型号 | `nvidia-smi` | P2 |

**设计注意：**
- 采集必须容错——任何一项失败不应阻断整体流程
- 输出格式要方便 LLM 阅读（不是给人看的 pretty print，而是结构化的 key-value）
- 要考虑虚拟环境 vs 系统环境的区分

**验收标准：** 在一个典型的 ML 项目目录下运行，能正确输出 Python、torch、transformers、CUDA 版本。

### 1.3 Traceback 解析 + 本地诊断 (第 2-3 周)

**任务：**

- `traceback_parser.py`：解析 Python traceback，提取关键信息
- `local_checker.py`：本地诊断逻辑
- `triage.py`：分流规则

**Traceback 解析需要提取：**

```python
@dataclass
class ParsedTraceback:
    exception_type: str          # e.g., "TypeError"
    exception_message: str       # e.g., "unexpected keyword argument 'trust_remote_code'"
    frames: list[StackFrame]     # 调用栈各帧
    root_package: str | None     # 异常发生在哪个包里
    is_user_code: bool           # 最内层帧是否在用户代码中
    involved_packages: set[str]  # 调用栈涉及的所有第三方包
```

**分流规则（启发式，不依赖 LLM）：**

以下信号提升"上游问题"的可能性：
1. 异常发生在第三方库的代码路径中（非用户代码）
2. 错误类型是 `TypeError`（参数签名变更）、`AttributeError`（API 移除）、`ImportError`（模块重构）
3. 调用栈涉及多个第三方库的交叉
4. 用户安装的版本是近期发布的（可能引入了 breaking change）

以下信号提升"本地问题"的可能性：
1. 异常发生在用户代码中
2. 错误类型是 `NameError`、`SyntaxError`、`IndentationError`
3. 只涉及单个包且用法明显不对

**本地诊断——API 签名检测（场景 A 的核心能力）：**

```python
# 思路：用 inspect 模块对比用户调用的参数和实际函数签名
# 例如 traceback 显示:
#   TypeError: __init__() got an unexpected keyword argument 'trust_remote_code'
# 可以 inspect 当前安装版本的该函数，确认是否确实没有这个参数
# 再检查 pip 版本历史，判断是参数名变了还是被移除了

import inspect

def check_api_signature(func_path: str, called_args: dict) -> SignatureCheckResult:
    """对比用户调用参数和实际函数签名"""
    ...
```

**验收标准：** 给定一个 `TypeError: unexpected keyword argument` 的 traceback，能正确判断为"API 签名变更"并定位到具体的包和函数。

### 1.4 实时 GitHub Issues 检索 (第 3-4 周)

**任务：**

- `github_client.py`：封装 GitHub Search API 和 Issues API
- `realtime.py`：检索策略——如何从 traceback 构造搜索 query
- `version_filter.py`：L1 版本过滤

**检索策略设计（关键难点）：**

从一个 traceback 构造搜索 query 不是简单的关键词拼接。需要 LLM 参与生成多组 query：

```
输入: TypeError: Qwen2VLForConditionalGeneration.__init__() got an 
      unexpected keyword argument 'trust_remote_code'
      
Agent 应生成的搜索策略:
1. 精确搜索: repo:QwenLM/Qwen2-VL "trust_remote_code" TypeError
2. 语义搜索: repo:huggingface/transformers Qwen2VL init argument error
3. 版本相关: repo:huggingface/transformers "4.55" breaking change Qwen
```

这个过程应该建模为 Agent 的 tool use——LLM 决定搜什么、搜哪个仓库、搜几次。

**仓库推断逻辑：**
- 从 traceback 的包名推断主仓库（transformers → huggingface/transformers）
- 维护一个常见包名→仓库的映射表
- 对于不在映射表中的包，用 PyPI API 查 homepage/repository URL

**L1 版本过滤：**
- 检索到 issues 后，提取 issue 中提到的版本号
- 与用户环境对比，过滤掉明显不相关的（如 issue 说的是 v3.x 的问题，用户在 v4.x）
- 这一步用正则 + 简单规则即可，不需要 LLM

**验收标准：** 给定选题报告中的 Qwen2.5-VL 案例的 traceback，能在 transformers 和 Qwen 仓库中检索到相关 issues，并通过版本过滤排除不相关结果。

### 1.5 Agent 编排 + CLI 串联 (第 4 周)

**任务：**

- `agent/core.py`：Agent 主循环——串联环境感知、本地诊断、分流、检索、输出
- `agent/tools.py`：定义 Agent 可调用的工具集
- `agent/prompts.py`：精心设计 system prompt 和 few-shot examples
- CLI 完整交互流程

**Agent 的 Tool 定义：**

```python
TOOLS = [
    # 环境感知
    Tool("get_env_snapshot", "获取当前 Python 环境的完整信息"),
    
    # 本地诊断
    Tool("check_api_signature", "检查函数签名是否与调用匹配", 
         params={"function_path": str, "called_args": dict}),
    Tool("get_package_changelog", "获取指定包的近期版本变更",
         params={"package": str, "from_version": str}),
    
    # GitHub 检索
    Tool("search_github_issues", "在 GitHub 仓库中搜索 issues",
         params={"query": str, "repo": str, "state": str}),
    Tool("get_issue_detail", "获取 issue 的完整内容和评论",
         params={"repo": str, "issue_number": int}),
    
    # Compatibility Matrix 查询 (Phase 2 加入)
    Tool("check_compatibility", "查询版本组合的已知兼容性问题",
         params={"packages": dict}),
]
```

**System Prompt 设计原则：**
1. 明确角色：你是一个专业的 ML 框架兼容性诊断专家
2. 明确流程：先环境感知 → 再本地诊断 → 必要时才检索
3. 版本意识：始终带着用户的版本信息做判断
4. 输出格式：结构化的诊断报告（问题定位、可能原因、建议方案、置信度）

**验收标准：** 端到端演示——粘贴一个真实的 ML 框架 traceback，Agent 能完成完整的诊断流程并输出有用的结果。

---

## Phase 2: 版本语义 + Compatibility Matrix (第 5-9 周)

**目标：** 实现 L2 版本语义理解；构建 Compatibility Matrix 数据管道；本地索引增强检索质量。

### 2.1 Compatibility Matrix 数据采集 (第 5-6 周)

**任务：**

- `matrix/crawler.py`：批量爬取目标仓库的 issues
- `matrix/extractor.py`：LLM 驱动的结构化信息提取
- 数据存储格式设计

**目标仓库与爬取范围：**

| 仓库 | 预估 issue 量 | 爬取策略 |
|------|--------------|---------|
| huggingface/transformers | ~25k closed | 按 label 过滤 (bug, compatibility) + 关键词 |
| vllm-project/vllm | ~5k closed | 全量爬取 closed issues |
| huggingface/peft | ~2k closed | 全量 |
| microsoft/DeepSpeed | ~5k closed | 按 label + 关键词 |
| QwenLM/Qwen2-VL, QwenLM/Qwen3 等 | ~2k | 全量 |
| pytorch/pytorch | 太大，只爬 CUDA 和兼容性相关 | 关键词过滤 |

**LLM 结构化提取的 schema：**

```python
@dataclass
class ExtractedIssueInfo:
    issue_id: str                          # repo#number
    title: str
    
    # 版本信息
    reported_versions: dict[str, str]      # {"transformers": "4.55.0", "torch": "2.3.0"}
    python_version: str | None
    cuda_version: str | None
    
    # 问题分类
    problem_type: Literal["crash", "wrong_output", "performance", "install", "other"]
    error_type: str | None                 # TypeError, AttributeError, ...
    error_message_summary: str
    
    # 解决方案
    has_solution: bool
    solution_type: Literal["version_change", "code_fix", "config_change", "workaround", "none"]
    solution_detail: str | None            # "downgrade transformers to 4.52.3"
    fix_version: str | None                # 如果是"在某版本修复了"
    
    # 影响范围
    affected_version_range: str | None     # ">= 4.53, < 4.56"
    related_issues: list[str]              # 关联 issue IDs
```

**成本控制：**
- 对大量 issues 做 LLM 提取，成本可能很高
- 策略：先用简单规则做初筛（是否包含版本号、是否有 traceback），再对高质量候选做 LLM 提取
- 使用便宜的模型（Claude Haiku / GPT-4o-mini）做提取，只对复杂 case 升级

**验收标准：** 从 transformers 仓库提取 500+ 条结构化兼容性记录，抽样检查准确率 > 85%。

### 2.2 矩阵聚合与存储 (第 6-7 周)

**任务：**

- `matrix/aggregator.py`：将提取结果聚合成兼容性矩阵
- 设计 Matrix 的查询接口（供 Agent 调用）

**矩阵结构：**

```python
# 核心数据结构：包A版本 × 包B版本 → 兼容性评分 + 详情
{
    "transformers==4.55.0 + torch==2.3.0": {
        "score": 0.3,          # 0=高风险, 1=安全
        "issue_count": 12,     # 相关 issue 数量
        "known_problems": [
            {
                "summary": "padding_side='left' 导致 generation 输出异常",
                "severity": "high",
                "solution": "降级到 transformers==4.52.3",
                "source_issues": ["QwenLM/Qwen3-VL#759", ...]
            }
        ]
    }
}
```

**查询接口设计：**

```python
class CompatibilityMatrix:
    def check(self, env: EnvSnapshot) -> list[CompatibilityWarning]:
        """给定环境快照，返回所有已知的兼容性风险"""
        ...
    
    def query_pair(self, pkg_a: str, ver_a: str, pkg_b: str, ver_b: str) -> PairResult:
        """查询两个包的特定版本组合"""
        ...
```

**验收标准：** 能查询 `transformers==4.55 + torch==2.3` 并返回已知问题列表。

### 2.3 L2 版本语义理解 (第 7-8 周)

**任务：**

- 增强 `version_filter.py`：从简单的版本号匹配升级到语义理解
- 解析 release notes 和 PR 信息，理解 bug 的引入/修复时间线

**L2 的核心能力：**

```
场景：用户在 transformers==4.54，搜到一个 issue 说 "4.53 有 bug，4.56 已修复"
L1 只看版本号匹配 → 可能不会返回这个 issue（因为没提到 4.54）
L2 理解 "4.53 引入，4.56 修复" → 判断 4.54 在影响范围内 → 高度相关
```

**实现方式：**
1. 从 GitHub Releases API 获取版本时间线
2. 从关联 PR 的 milestone/tag 获取修复版本
3. LLM 辅助理解 issue 讨论中的版本范围描述（"this was introduced in..." "fixed by PR #xxx which is in v..."）
4. 将理解结果结构化为 `AffectedVersionRange`

**验收标准：** 给定用户版本 4.54，能正确判断"4.53 引入、4.56 修复"的 issue 与用户相关。

### 2.4 本地向量索引 (第 8-9 周)

**任务：**

- `search/local_index.py`：基于已爬取的 issues 建立向量索引
- 与实时检索结果合并去重

**索引构建：**
- 复用 Matrix 数据采集管道的 issues 数据
- Embedding 模型：BGE-small（本地运行）或 text-embedding-3-small（API）
- 向量库：FAISS（轻量、无需服务）
- 索引单元：每个 issue 的标题 + 首条评论 + LLM 提取的摘要

**检索合并策略：**

```
1. 实时 GitHub API 搜索 → 结果集 A
2. 本地向量检索 → 结果集 B
3. 合并去重（按 issue URL 去重）
4. 统一经过版本过滤 + 排序
```

**验收标准：** 对评估集中的 case，本地索引+实时检索的召回率比纯实时检索提升 15%+。

---

## Phase 3: 社区互助 + MCP + 评估 (第 10-13 周)

### 3.1 Issue 起草 (第 10 周)

**任务：**

- `community/issue_drafter.py`：自动起草高质量 GitHub issue
- 包含：格式化的环境信息、traceback、最小复现描述、关联已有 issue

**Issue 模板：**

```markdown
## Environment
- Python: {version}
- OS: {os}
- CUDA: {cuda_version}
- Key packages:
  - transformers: {version}
  - torch: {version}
  - ...

## Description
{LLM 生成的问题描述，基于 traceback 和诊断结果}

## Steps to Reproduce
{LLM 辅助整理的复现步骤}

## Error Traceback
```
{原始 traceback}
```

## Related Issues
- #{related_1}: {简述}
- #{related_2}: {简述}

## Additional Context
{环境快照中可能相关的其他信息}
```

### 3.2 社区回馈 (第 11 周)

**任务：**

- `community/reply_suggester.py`：检测可回馈的 issue，草拟回复
- 交互流程设计

**触发条件：**
1. 诊断过程中检索到了相关 issue
2. 该 issue 仍处于 open 状态
3. 用户的问题已经通过其他途径解决
4. 用户的解决方案与该 issue 的问题匹配

**交互流程：**

```
Agent: 诊断完成。你的问题是 transformers 4.55 的 padding_side bug，
       降级到 4.52.3 可以解决。

       另外，发现 issue QwenLM/Qwen3-VL#759 有 3 人遇到了类似问题且未解决。
       是否帮你草拟一条评论分享你的解决方案？

User: 好的

Agent: [生成评论草稿]
       Hi, I encountered the same issue. The root cause is...
       
       是否发布？[Y/n/edit]
```

### 3.3 MCP Server (第 11-12 周)

**任务：**

- `mcp/server.py`：实现 MCP 协议的 Server
- 暴露 `diagnose` 和 `check_compatibility` 工具

**MCP Tool 定义：**

```python
# diagnose: 核心诊断工具
{
    "name": "octoscout_diagnose",
    "description": "诊断 Python/ML 框架的运行时错误，特别是版本兼容性问题",
    "parameters": {
        "traceback": "完整的错误 traceback",
        "env_info": "可选，手动提供的环境信息",
        "auto_detect_env": "是否自动检测环境，默认 true"
    }
}

# check_compatibility: 兼容性查询
{
    "name": "octoscout_check_compatibility",
    "description": "查询当前环境的已知兼容性问题",
    "parameters": {
        "packages": "要查询的包和版本，如 {'transformers': '4.55.0', 'torch': '2.3.0'}"
    }
}
```

### 3.4 评估基准构建 (第 12-13 周)

**任务：**

- 收集 30-50 个真实已解决的 issue 作为评估用例
- 实现自动化评估脚本
- 对比 baseline

**评估用例来源：**

| 仓库 | 目标数量 | 筛选标准 |
|------|---------|---------|
| huggingface/transformers | 15-20 | closed + has solution + 版本相关 |
| vllm-project/vllm | 8-10 | 同上 |
| huggingface/peft | 5-8 | 同上 |
| QwenLM/* | 5-8 | 同上 |

**评估指标：**

| 指标 | 定义 | 目标 |
|------|------|------|
| Issue 召回率 | 在 top-5 结果中找到对应 issue 的比例 | > 60% |
| 方案准确率 | 返回的解决方案确实能解决问题的比例 | > 50% |
| 版本匹配准确率 | 正确判断 issue 与用户版本是否相关 | > 80% |
| API 签名检测准确率 | 本地诊断准确识别签名变更 | > 90% |

**Baseline 对比：**

1. **GitHub 搜索：** 用 traceback 中的错误信息直接搜索
2. **直接问 LLM：** 把 traceback + 环境信息发给 Claude/GPT，不给任何工具
3. **OctoScout：** 完整 Agent 流程

---

## Phase 4: 可视化 + 打磨 + 发布 (第 14-16 周)

### 4.1 Matrix 热力图 Web UI (第 14 周)

**技术方案：** 简单的静态页面 + Plotly.js / D3.js

**功能：**
- 选择包对（如 transformers × torch）查看兼容性热力图
- 点击单元格查看具体问题列表
- 输入环境信息，高亮当前组合的风险

### 4.2 CLI 体验打磨 (第 15 周)

- Rich 库做终端美化（进度条、彩色输出、表格）
- 错误处理和 edge case 覆盖
- 配置文件支持（GitHub token、LLM API key、默认仓库列表）
- `octoscout matrix` 子命令：查询兼容性矩阵

### 4.3 文档 + 开源发布 (第 16 周)

- README 文档（安装、快速开始、配置、架构说明）
- PyPI 发布
- GitHub Actions CI
- 演示视频 / GIF

---

## 关键技术决策备忘

### 1. Agent 编排模式

**推荐：ReAct 模式（Reasoning + Acting）**

让 LLM 在每一步先"思考"再"行动"，而不是一次性规划所有步骤。理由：
- 诊断过程天然是迭代的——本地诊断的结果影响是否需要检索，检索的结果影响下一步搜什么
- Tool use 的 ReAct 循环是 Claude 和 OpenAI 都原生支持的模式
- 比预定义的 DAG/Pipeline 更灵活，能处理意外情况

### 2. 分流逻辑：启发式规则 vs LLM

**推荐：启发式规则为主，LLM 为辅**

报告中提到"置信度评估"，但 LLM 的概率校准不太可靠。建议：
- 用明确的规则做初步分流（如 traceback 是否在第三方库代码中）
- 边界模糊的 case 才交给 LLM 判断
- 这样既快又稳定，且可调试

### 3. GitHub API Rate Limit

- 未认证：60 次/小时
- Token 认证：5000 次/小时
- Search API：30 次/分钟

**应对策略：**
- 必须要求用户配置 GitHub token
- 实现请求级缓存（相同 query 不重复请求）
- Agent 的搜索次数做上限控制（单次诊断不超过 10-15 次 API 调用）

### 4. LLM 成本控制

- 环境感知、traceback 解析：不需要 LLM，纯代码实现
- 搜索 query 生成、方案验证、issue 起草：需要 LLM
- Matrix 数据提取：大量调用，用便宜模型
- 预估单次诊断 LLM 成本：3-5 次 API 调用，约 $0.01-0.05

---

## 风险与应对

| 风险 | 影响 | 应对 |
|------|------|------|
| GitHub API rate limit 限制检索深度 | 单次诊断能搜的 issue 数量有限 | 本地索引补充；优化 query 质量减少无效请求 |
| LLM 结构化提取准确率不够 | Matrix 数据质量差 | 设计验证规则；人工抽检；迭代 prompt |
| 检索召回率低（搜不到相关 issue） | 核心价值打折扣 | 多策略检索；本地语义索引；用户反馈闭环 |
| 跨 provider 的 tool use 行为不一致 | Claude 和 OpenAI 表现差异大 | 充分测试；对齐 prompt；必要时做 provider 特定适配 |