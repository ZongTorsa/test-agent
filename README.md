# Python Agent & RAG 学习工程

一个用于学习 LangChain、RAG 与多智能体协同的 Python 项目。项目提供基础工具 Agent、RAG-Agent，以及带短期/长期记忆的 Planner–Executor 多智能体工作流。

## 项目架构

```text
用户需求
  │
  ▼
multi_agent.py
  │
  ├─ MultiAgentMemory
  │   ├─ 短期记忆：当前会话最近几轮上下文
  │   └─ 长期记忆：Chroma / project_memory 中的项目事实
  │
  ├─ Planner Agent（planner_agent.py）
  │   └─ PlanMessage / TaskMessage
  │
  └─ Executor Agent（executor_agent.py）
      ├─ calculator
      ├─ file_reader
      └─ rag_retriever → Chroma / knowledge
          └─ ExecutionResultMessage → 汇总答案
```

## 环境准备

建议使用项目内虚拟环境：

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
```

复制 `.env.example` 为 `.env`，并配置密钥：

```text
DEEPSEEK_API_KEY=你的 DeepSeek 密钥
DASHSCOPE_API_KEY=你的 DashScope 密钥
```

`.env` 不应提交到版本控制。运行环境变量优先于 `.env` 中的同名配置。

## 启动命令

### 多智能体主程序

```powershell
.\.venv\Scripts\python.exe multi_agent.py "整理知识库中所有关于 RAG 优化的内容并总结优缺点"
```

主入口会依次执行：需求规划、任务执行、结果汇总、短期记忆更新和长期记忆沉淀。

### Streamlit 测试网页

```powershell
.\.venv\Scripts\python.exe -m streamlit run streamlit_app.py
```

启动后打开 `http://localhost:8501`，可在网页中测试多智能体、RAG 检索和双层记忆。

### 基础工具 Agent

```powershell
.\.venv\Scripts\python.exe run_agent.py "请计算 (12 + 8) * 3 / 2"
```

### 单 RAG-Agent

```powershell
.\.venv\Scripts\python.exe rag_agent.py "RAG 的基本流程是什么？"
```

### 构建与查询向量库

```powershell
.\.venv\Scripts\python.exe vector_store.py --rebuild
.\.venv\Scripts\python.exe vector_store.py --query "RAG 的基本流程是什么？" --k 3
```

## 模块说明

| 模块 | 作用 |
| --- | --- |
| `multi_agent.py` | 多智能体统一入口，协调 Planner、Executor 和记忆系统。 |
| `streamlit_app.py` | Streamlit 交互测试页面，展示多智能体、RAG 和双层记忆。 |
| `planner_agent.py` | 将复杂需求拆分为有序 `PlanMessage` 和 `TaskMessage`。 |
| `executor_agent.py` | 逐条执行任务，可使用 RAG 与本地工具，并汇总执行结果。 |
| `memory.py` | 短期会话记忆与 Chroma 长期项目记忆；新会话会检索历史事实。 |
| `rag_agent.py` | 单 RAG-Agent，支持检索后反思并按需补检索，最多 3 轮。 |
| `agent.py` | DeepSeek 对话模型初始化与基础工具 Agent。 |
| `tools.py` | `calculator`、`file_reader`、`rag_retriever` 三个工具。 |
| `rag_loader.py` | 递归加载 `knowledge/` 中的 UTF-8 `.md` / `.txt` 文档并切分。 |
| `vector_store.py` | DashScope `text-embedding-v4` 嵌入、Chroma 持久化和语义检索。 |
| `config.py` | 集中读取 `.env` 与环境变量，校验项目参数。 |
| `logging_config.py` | 统一项目日志格式与日志级别。 |
| `file_report.py` | 统计当前目录文件的行数和大小。 |

## 知识库与记忆

- `knowledge/`：放置需要 RAG 检索的 `.md` 和 `.txt` 文档。
- `chroma_db/`：Chroma 本地持久化目录，包含知识库集合和 `project_memory` 长期记忆集合。
- 知识库不存在相关内容时，RAG 工具会返回未命中信息；Agent 被约束为如实说明，不编造资料。

## 常用配置

配置项均可在 `.env` 或系统环境变量中设置，完整列表见 `.env.example`。

- `DEEPSEEK_MODEL`：默认 `deepseek-v4-pro`。
- `EMBEDDING_MODEL`：默认 `text-embedding-v4`。
- `RAG_CHUNK_SIZE` / `RAG_CHUNK_OVERLAP`：文本切分参数。
- `RAG_RETRIEVAL_K` / `RAG_MAX_DISTANCE`：知识库检索数量和距离阈值。
- `SHORT_TERM_MEMORY_TURNS`：当前会话保留的最近轮数。
- `MEMORY_RETRIEVAL_K` / `MEMORY_MAX_DISTANCE`：长期记忆检索参数。
- `LOG_LEVEL`：支持 `DEBUG`、`INFO`、`WARNING`、`ERROR`。
