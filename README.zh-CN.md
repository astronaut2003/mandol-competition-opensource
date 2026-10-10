# Mandol Competition

[English](README.md) | [简体中文](README.zh-CN.md)

Mandol Competition 是面向 Agent Memory Leaderboard（AML）的统一、自托管智能体
记忆系统，通过共享的持久化存储与检索基础设施支持文本、代码和多模态记忆。

## 项目概览

项目提供三个独立的 Memory API：**Textual T0、Coding C0、Multimodal M0**。
三个赛道各自维护记忆表示与检索策略，共用 SQLite 持久化、向量发布、请求幂等和
恢复机制。Add 写入记忆，Search 返回已存储的证据，不直接生成最终答案。

实现基于 [Mandol](https://github.com/AgentCombo/Mandol)。英文 README 是主要维护文档；
[冻结方法说明](docs/method/Frozen-Pipelines.md)完整描述本项目的参赛实现。

## 主要特性

- **三赛道隔离：**独立 API、SQLite 存储根目录与 Qdrant collection，隔离用户和赛道数据。
- **可靠写入与恢复：**请求重放、规范化内容冲突检测、向量发布校验和 reconciliation 支持持久化恢复。
- **赛道专用检索：**文本和多模态使用词法、Dense、原生 Sparse 混合检索；代码使用词法证据与可选 Dense。
- **多模态记忆：**验证文本/图像输入，通过 VLM 构建表示，并由内容寻址 MediaStore 保存和恢复原始图像。
- **外部推理服务：**CPU HTTP 客户端与桥接服务连接模型端点，API 环境不加载 GPU 模型。
- **可复现测试：**确定性推理 fixture 配合真实 SQLite、临时 Qdrant 和 HTTP API，覆盖隔离、故障与恢复。

## 架构与三赛道差异

三个 API 共用 `mandol_substrate`。SQLite 保存规范记忆状态；Qdrant 保存可校验、
可重建的向量投影与精简元数据；Multimodal 使用独立 MediaStore 保存图像。
每个赛道运行一个 API worker，外部 Qdrant 的生命周期由部署者管理。

| 赛道 | 冻结检索 | Add 成功条件 | 返回策略 |
|---|---|---|---|
| Textual T0 | BM25 + v4 Dense/原生 Sparse，RRF60，Querit B1 重排 | 完成必要的向量发布校验 | 最多 10 条来源/上下文记录，24,000 字符预算 |
| Coding C0 | 原始历史与工程 episode；词法 + 可选完整 frontier 的 v4 Dense，Dot 相似度 | 规范记忆与词法持久化 | 最多 8 张证据卡，总计 16,000 字符，每来源 1,200 字符 |
| Multimodal M0 | DM + SGM；BM25 + v4 Dense/原生 Sparse，RRF60，BGE BF16 重排 | 完成 prepare 后必要的向量发布 | 按来源顺序返回文本/图像，遵守上下文和媒体预算 |

返回数量还受请求 `top_k` 限制。T0/M0 使用 1024 维 `text-embedding-v4` 和同一
Provider 的原生 Sparse；历史 `splade` 字段名不表示需要启动 SPLADE 服务。
**C0 不使用原生 Sparse 或神经 reranker。** 详细架构图见
[英文 README](README.md#architecture)，模型及数值约定见
[推理部署](docs/deployment/Inference.md)。

## 快速安装

使用 Linux、Python 3.12 和 [uv](https://docs.astral.sh/uv/)。在仓库根目录执行：

```bash
uv sync --frozen --python 3.12.13 --extra all --extra dev
uv pip install --python .venv/bin/python --no-deps -e services/inference
python3 deployment/cluster/frozen.py
.venv/bin/python -m pytest tests/common/test_neutral_contracts.py tests/common/test_retrieval_primitives.py -q
```

这会安装应用、仓库内的 substrate 包和 CPU HTTP 桥接服务，不会安装 GPU 模型
服务器或下载权重。单赛道安装可将 `--extra all` 换成 `--extra textual`、
`--extra code` 或 `--extra multimodal`；完整测试需要三赛道 extras 与 `--extra dev`。

## 部署与配置

按照 [本地部署指南](docs/deployment/Local-Reproduction.md)准备 Qdrant **1.19.1**、
v4 Provider 桥接、Querit/BGE reranker 和授权的 GPT-4o-mini 兼容服务。
API 环境锁定的 Qdrant 客户端为 **1.19.0**，reranker 的冻结精度为 **BF16**。
C0 的 Dense 通道可降级为词法检索。

复制并编辑 [部署模板](configs/deployment/frozen-bf16.example.toml) 和
[模型绑定模板](configs/deployment/serving-bindings.example.json)，通过环境变量或
支持的 secret mount 注入凭据。`.env.example` 仅列出变量名，不会自动加载。
TOML 不展开 shell 变量；模板中的 VLM URL 必须替换为自己的授权端点。

配置完成后，在不同终端启动三个赛道：

```bash
.venv/bin/python scripts/start_api.py textual --config configs/local/deployment.toml
.venv/bin/python scripts/start_api.py code --config configs/local/deployment.toml
.venv/bin/python scripts/start_api.py multimodal --config configs/local/deployment.toml
```

默认仅监听 loopback，端口分别为 **18600 / 18601 / 18602**。远程访问使用 HTTPS，
Qdrant、推理和运维端点保持私有。凭据处理和存储保护见 [安全说明](SECURITY.md)。

## API 使用

三个赛道提供相同的 `GET /health`、`POST /add` 和 `POST /search` 入口，通过端口
选择赛道。认证支持 `Authorization: Bearer`、`Authorization: Token` 或 `X-Api-Key`。

在已导出自己的 `MANDOL_MEMORY_SYSTEM_KEY` 后，可调用 Textual API：

```bash
curl -sS http://127.0.0.1:18600/health
curl -sS http://127.0.0.1:18600/add \
  -H "Authorization: Bearer ${MANDOL_MEMORY_SYSTEM_KEY}" \
  -H 'Content-Type: application/json' \
  -d '{"request_id":"example-1","user_id":"example-user","session_id":"session-1","messages":[{"role":"user","content":"I prefer coffee without sugar."}]}'
curl -sS http://127.0.0.1:18600/search \
  -H "Authorization: Bearer ${MANDOL_MEMORY_SYSTEM_KEY}" \
  -H 'Content-Type: application/json' \
  -d '{"query":"What is my coffee preference?","user_id":"example-user","top_k":100}'
```

Add 成功返回请求、用户与 session 标识；相同请求和规范内容会重放同一逻辑结果，
内容改变则冲突。Search 返回 `{"data": [...]}`，允许为空，按用户跨 session 检索，
不同赛道物理隔离。多模态支持有序文本/图像片段。
完整 Schema、响应示例和图像限制见 [API 文档](docs/API.md)。

## 测试与复现

上面的快速测试无需模型凭据。完整存储与 HTTP 回归使用带所有权标签的全新临时
Qdrant，通过 `scripts/run_tests.py` 运行；命令见
[英文 README](README.md#testing-and-reproduction) 和 [复现指南](docs/Reproduction.md)。
不要把测试连接到生产数据库。

合成推理测试验证接口、存储和故障行为，不代表真实模型端到端质量或官方 AML
成绩。真实模型测试需要自己的合法模型服务、硬件、凭据和新存储，并显式启用。
离线 QA 可单独获取校验固定 Commit 的上游评测脚本；脚本、数据集与模型权重不随
本项目分发。具体依赖与评测边界见 [复现指南](docs/Reproduction.md)。

## 文档导航

- [冻结 T0/C0/M0 方法](docs/method/Frozen-Pipelines.md)
- [本地部署与故障排查](docs/deployment/Local-Reproduction.md)
- [推理服务与模型配置](docs/deployment/Inference.md)
- [API 文档](docs/API.md)
- [测试与复现](docs/Reproduction.md)
- [来源署名](docs/ATTRIBUTION.md) 与 [依赖许可证](docs/Dependency-Licenses.md)
- [安全说明](SECURITY.md) 与 [贡献指南](CONTRIBUTING.md)

## 许可证与来源

Mandol Competition 及其统一参赛实现按 [Apache-2.0](LICENSE) 分发。
原始 Mandol 的版权与署名保留在 [NOTICE](NOTICE) 和
[Attribution](docs/ATTRIBUTION.md)。独立第三方依赖、模型、服务、数据集及自行获取
的评测脚本适用各自条款；本项目不分发模型权重或数据集。

原始设计参考论文
[Mandol: An Agglomerative Agent Memory System for Long-Term Conversations](https://arxiv.org/abs/2606.29778)，
作者为 Yuhan Zhang、Zhiyuan Guo、Ziheng Zeng、Wei Wang、Wentao Wu、Lijie Xu。
本仓库描述统一参赛实现，论文和官方榜单结果分别对应各自评测版本。
