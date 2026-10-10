"""Machine-readable exports for inspecting a real SemanticCode graph.

JSONL is the authoritative representation. CSV and GraphML are convenience
views built from the same records; embeddings are represented only by flags.
"""

from __future__ import annotations

import csv
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable, TYPE_CHECKING

import networkx as nx

if TYPE_CHECKING:
    from .facade import SemanticCode


_HIGH_LEVEL_TYPES = {
    "file_memory",
    "symbol_memory",
    "repair_memory",
    "module_memory",
    "architecture_memory",
}


def _json_default(value: Any) -> Any:
    if hasattr(value, "item"):
        return value.item()
    return str(value)


def _json_text(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, default=_json_default)


def _write_json(path: Path, value: Any) -> None:
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, default=_json_default) + "\n",
        encoding="utf-8",
    )


def _write_jsonl(path: Path, records: Iterable[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(_json_text(record) + "\n")


def _node_record(unit: Any, spaces: list[str]) -> dict[str, Any]:
    software_type = str(unit.metadata.get("software_type", "unknown"))
    raw_data = dict(unit.raw_data)
    metadata = dict(unit.metadata)
    if software_type == "semantic_block":
        software_type = "code_segment"
        raw_data["software_type"] = software_type
        raw_data.setdefault("segment_id", raw_data.pop("block_id", ""))
        raw_data.setdefault(
            "segment_type", raw_data.pop("statement_type", "statement_group")
        )
        raw_data.setdefault("parent_uid", raw_data.get("function_uid", ""))
        metadata["software_type"] = software_type
    return {
        "uid": unit.uid,
        "software_type": software_type,
        "memory_spaces": spaces,
        "raw_data": raw_data,
        "metadata": metadata,
        "has_dense_embedding": unit.embedding is not None,
        "has_sparse_embedding": unit.sparse_embedding is not None,
    }


def _edge_records(
    semantic_code: SemanticCode, unit_uids: set[str]
) -> list[dict[str, Any]]:
    graph = semantic_code.semantic_graph
    records: list[dict[str, Any]] = []
    for edge_index in graph.rx_graph.edge_indices():
        source_index, target_index = graph.rx_graph.get_edge_endpoints_by_index(
            edge_index
        )
        source_uid = graph._index_to_uid.get(source_index, "")
        target_uid = graph._index_to_uid.get(target_index, "")
        if source_uid not in unit_uids or target_uid not in unit_uids:
            continue
        edge_data = dict(graph.rx_graph.get_edge_data_by_index(edge_index) or {})
        relation = str(edge_data.pop("type", "unknown"))
        records.append(
            {
                "edge_index": int(edge_index),
                "source_uid": source_uid,
                "target_uid": target_uid,
                "relation": relation,
                "properties": edge_data,
            }
        )
    return sorted(
        records,
        key=lambda edge: (
            edge["source_uid"],
            edge["target_uid"],
            edge["relation"],
            edge["properties"].get("source_order", -1),
            edge["edge_index"],
        ),
    )


def _write_graphml(
    path: Path,
    nodes: Iterable[dict[str, Any]],
    edges: Iterable[dict[str, Any]],
) -> None:
    graph = nx.MultiDiGraph()
    for node in nodes:
        graph.add_node(
            node["uid"],
            software_type=node["software_type"],
            memory_spaces_json=_json_text(node["memory_spaces"]),
            raw_data_json=_json_text(node["raw_data"]),
            metadata_json=_json_text(node["metadata"]),
            has_dense_embedding=str(node["has_dense_embedding"]).lower(),
            has_sparse_embedding=str(node["has_sparse_embedding"]).lower(),
        )
    for edge in edges:
        properties = edge["properties"]
        attributes = {
            "relation": edge["relation"],
            "properties_json": _json_text(properties),
        }
        for key in (
            "source_order",
            "start_line",
            "end_line",
            "start_column",
            "end_column",
            "expression",
            "enclosing_segment_uid",
            "call_site_id",
        ):
            if key in properties:
                attributes[key] = (
                    "" if properties[key] is None else str(properties[key])
                )
        graph.add_edge(
            edge["source_uid"],
            edge["target_uid"],
            key=str(edge["edge_index"]),
            **attributes,
        )
    nx.write_graphml(graph, path, infer_numeric_types=False)


def _local_subgraph(
    target_uid: str,
    nodes_by_uid: dict[str, dict[str, Any]],
    edges: list[dict[str, Any]],
) -> dict[str, Any]:
    direct_relations = {
        "CONTAINS",
        "CONTAINED_IN",
        "HAS_METHOD",
        "CALLS",
        "TESTED_BY",
        "AFFECTS_BEHAVIOR_OF",
        "MODIFIES",
        "HAS_EVOLUTION",
        "IMPACTS",
        "MENTIONS",
        "DERIVED_FROM",
        "SUPPORTED_BY",
    }
    selected = {target_uid}
    for edge in edges:
        if edge["relation"] not in direct_relations:
            continue
        if edge["source_uid"] == target_uid:
            selected.add(edge["target_uid"])
        elif edge["target_uid"] == target_uid:
            selected.add(edge["source_uid"])

    # Segments are owned by the function even though the durable edge points
    # from CodeSegment to its parent.
    selected.update(
        uid
        for uid, node in nodes_by_uid.items()
        if node["software_type"] == "code_segment"
        and node["raw_data"].get("parent_uid", node["raw_data"].get("function_uid"))
        == target_uid
    )
    parent_classes = {
        uid for uid in selected if nodes_by_uid[uid]["software_type"] == "class"
    }
    for edge in edges:
        if edge["relation"] not in {"CONTAINS", "CONTAINED_IN"}:
            continue
        if (
            edge["target_uid"] in parent_classes
            and nodes_by_uid[edge["source_uid"]]["software_type"] == "file"
        ):
            selected.add(edge["source_uid"])
        elif (
            edge["source_uid"] in parent_classes
            and nodes_by_uid[edge["target_uid"]]["software_type"] == "file"
        ):
            selected.add(edge["target_uid"])

    # Follow concrete evolution evidence and the high-level hierarchy, but do
    # not expand a ModuleMemory to every member in a large repository.
    for _ in range(3):
        additions: set[str] = set()
        for edge in edges:
            source = edge["source_uid"]
            target = edge["target_uid"]
            relation = edge["relation"]
            source_type = nodes_by_uid[source]["software_type"]
            target_type = nodes_by_uid[target]["software_type"]
            if (
                source in selected
                and source_type == "evolution"
                and relation
                in {
                    "CAUSED_BY_EVIDENCE",
                    "EVOLVES_FROM",
                }
            ):
                additions.add(target)
            if (
                target in selected
                and target_type in _HIGH_LEVEL_TYPES
                and relation
                in {
                    "DERIVED_FROM",
                    "SUPPORTED_BY",
                }
            ):
                additions.add(source)
            if target in selected and source_type in _HIGH_LEVEL_TYPES:
                if target == target_uid or target_type in _HIGH_LEVEL_TYPES:
                    additions.add(source)
        if not additions - selected:
            break
        selected.update(additions)

    if len(selected) > 50:
        preferred = {target_uid}
        preferred.update(
            uid
            for uid in selected
            if nodes_by_uid[uid]["software_type"]
            in {
                "file",
                "class",
                "code_segment",
                "commit",
                "evolution",
                "issue",
                *_HIGH_LEVEL_TYPES,
            }
        )
        selected = preferred | set(
            sorted(selected - preferred)[: max(0, 50 - len(preferred))]
        )

    local_nodes = [nodes_by_uid[uid] for uid in sorted(selected)]
    local_edges = [
        edge
        for edge in edges
        if edge["source_uid"] in selected and edge["target_uid"] in selected
    ]
    return {"target_uid": target_uid, "nodes": local_nodes, "edges": local_edges}


def _provenance_subgraph(local: dict[str, Any]) -> dict[str, Any]:
    node_map = {node["uid"]: node for node in local["nodes"]}
    target_uid = local["target_uid"]
    target_symbols = {
        uid
        for uid, node in node_map.items()
        if node["software_type"] == "symbol_memory"
        and node["raw_data"].get("subject_uid") == target_uid
    }
    selected = {target_uid, *target_symbols}
    for edge in local["edges"]:
        if edge["source_uid"] in target_symbols and edge["relation"] in {
            "DERIVED_FROM",
            "SUPPORTED_BY",
        }:
            selected.add(edge["target_uid"])
    # Add actual Module/Architecture parents without expanding their complete
    # repository-wide evidence fan-out.
    for _ in range(3):
        additions = {
            edge["source_uid"]
            for edge in local["edges"]
            if edge["target_uid"] in selected
            and edge["relation"] in {"DERIVED_FROM", "SUPPORTED_BY"}
            and node_map[edge["source_uid"]]["software_type"] in _HIGH_LEVEL_TYPES
        }
        if not additions - selected:
            break
        selected.update(additions)
    provenance_edges = [
        edge
        for edge in local["edges"]
        if edge["source_uid"] in selected
        and edge["target_uid"] in selected
        and edge["relation"]
        in {
            "DERIVED_FROM",
            "SUPPORTED_BY",
            "CALLS",
            "IMPACTS",
            "HAS_EVOLUTION",
            "CAUSED_BY_EVIDENCE",
            "EVOLVES_FROM",
            "MODIFIES",
            "TESTED_BY",
        }
    ]
    return {
        "target_uid": local["target_uid"],
        "nodes": [node_map[uid] for uid in sorted(selected)],
        "edges": provenance_edges,
    }


def _write_readme(
    path: Path,
    summary: dict[str, Any],
    target: dict[str, Any],
    segments: list[dict[str, Any]],
    calls: list[dict[str, Any]],
    local: dict[str, Any],
) -> None:
    function_uid = target["uid"]
    symbol_uid = f"symbol_memory:{function_uid.split(':', 1)[-1]}"
    local_type_counts = Counter(node["software_type"] for node in local["nodes"])
    repeated = summary.get("repeated_call_example")
    repeated_text = (
        f"例如 `{repeated['source_uid']}` 在不同调用点 {repeated['source_orders']} "
        f"重复调用 `{repeated['target_uid']}`。"
        if repeated
        else "当前导出中没有解析到重复的已解析调用对。"
    )
    call_example = calls[0] if calls else None
    call_text = (
        f"本例真实边为 `{function_uid} --CALLS(source_order="
        f"{call_example['properties'].get('source_order')}, line="
        f"{call_example['properties'].get('start_line')})--> "
        f"{call_example['target_uid']}`。"
        if call_example
        else "该函数没有可解析到仓库内部 UID 的 CALLS 边。"
    )
    path.write_text(
        f"""# Kombu 真实 Coda 代码图

本目录是 `{summary["repository"]}` 在 revision `{summary["revision"]}` 上实际构建出的
SemanticGraph 导出。`full_graph/nodes.jsonl` 与 `full_graph/edges.jsonl` 是权威数据；CSV
用于快速查看，GraphML 用于 Gephi/Cytoscape。embedding 数组未导出，只保留存在性标记。

## A. 整体层次

```text
Repository → Directory → File → Class / Function
                                  ↑          ↑
                         HAS_METHOD   CodeSegment --CONTAINED_IN

Commit --MODIFIES--> Function --HAS_EVOLUTION--> Evolution
                                 Evolution --CAUSED_BY_EVIDENCE--> Commit

ArchitectureMemory → ModuleMemory → SymbolMemory
                                      └─ DERIVED_FROM / SUPPORTED_BY → 原始证据
```

本图共 {summary["node_count"]:,} 个节点、{summary["edge_count"]:,} 条边；局部图含
{len(local["nodes"])} 个节点、{len(local["edges"])} 条真实边，类型分布为
`{dict(sorted(local_type_counts.items()))}`。

## B. 节点保存什么

- `repository/directory/file`：仓库身份、目录与文件属性、语言、import、hash。
- `class/function`：统一符号身份；Function 额外保存签名、源码、span、hash、静态行为和 call sites。
- `code_segment`：保存有检索价值的分支、循环、异常处理和 statement group，不持久化完整 AST。
- `commit/evolution`：Git 证据与符号变化节点。
- `config/issue`：配置项和外部任务证据。
- `symbol_memory/repair_memory/module_memory/architecture_memory`：可回溯的证据聚合。

建议先打开 `nodes_by_type/function.jsonl`、`nodes_by_type/code_segment.jsonl`、
`nodes_by_type/evolution.jsonl` 和 `nodes_by_type/symbol_memory.jsonl`。

## C. 主要 Relation

- `CONTAINS / CONTAINED_IN / HAS_METHOD`：仓库和符号层次。
- `IMPORTS / CALLS / TESTED_BY`：静态结构与测试联系。
- `MODIFIES / HAS_EVOLUTION / EVOLVES_FROM / CAUSED_BY_EVIDENCE`：Git 演化证据。
- `DERIVED_FROM / SUPPORTED_BY`：高阶记忆 provenance。

## D. Function 如何进入图

选定函数 `{function_uid}` 来自
`{target["raw_data"].get("file_path")}:{target["raw_data"].get("start_line")}-`
`{target["raw_data"].get("end_line")}`。Python AST 先提取确定性事实，再经 FunctionSchema
归一化为标准 MemoryUnit；原始源码、签名、span、content hash 与 call_sites 都保存在
`selected_case/target_function.json`。

## E. Function 与 CodeSegment

该函数实际生成 {len(segments)} 个 CodeSegment。持久边方向是
`CodeSegment --CONTAINED_IN--> Function`；多个片段按源码顺序用 `NEXT_SEGMENT` 连接。
详见 `selected_case/target_function_segments.json`。

## F. CALLS 顺序属性

`source_order` 是词法/源码顺序，不是运行时执行顺序。每条边同时保留行列、原表达式、
`call_site_id` 与 `enclosing_segment_uid`，因此同一 caller 在两个位置调用同一 callee 时不会去重。
目标函数提取出 {len(target["raw_data"].get("call_sites", []))} 个 ParsedCall，其中
{len(calls)} 个能保守解析为仓库内部 CALLS 边。{call_text} {repeated_text}

## G. Evolution 与 Git

Commit 通过 `MODIFIES` 指向 File/Function；Function 通过 `HAS_EVOLUTION` 指向独立
Evolution，Evolution 再通过 `CAUSED_BY_EVIDENCE` 回到 Commit，并可用 `EVOLVES_FROM`
串成历史链。本导出有 {summary["commit_count"]} 个 Commit、{summary["evolution_count"]}
个 Evolution。

## H. 高阶记忆 provenance

目标 SymbolMemory 是 `{symbol_uid}`。其 `derived_from` 和显式 `DERIVED_FROM` 边均指向
真实底层 UID。`selected_case/provenance_subgraph.json` 只收录真实节点/边，不添加示意节点。
目标 task 在 base revision 尚无 fix commit 证据，因此没有为该目标伪造 RepairMemory。
""",
        encoding="utf-8",
    )


def _write_repository_readme(path: Path, summary: dict[str, Any]) -> None:
    """Write a repository-level guide without selecting a task-specific symbol."""
    path.write_text(
        f"""# 真实 Coda 代码图

本目录是 `{summary["repository"]}` 在 revision `{summary["revision"]}` 上实际构建出的
SemanticGraph 导出。`full_graph/nodes.jsonl` 与 `full_graph/edges.jsonl` 是权威数据；CSV
用于快速检查，GraphML 可由 Gephi/Cytoscape 打开。embedding 数组未导出，只保留存在性标记。

本次导出刻意没有预先选择任务目标函数：初始代码图只由仓库源码、Git 历史与确定性
高阶聚合构建，避免将事后诊断信息泄漏给后续 Agent。

## 图的组织

```text
Repository → Directory → File → Class / Function
                                  ↑          ↑
                         HAS_METHOD   CodeSegment --CONTAINED_IN

Commit --MODIFIES--> Function --HAS_EVOLUTION--> Evolution
                                 Evolution --CAUSED_BY_EVIDENCE--> Commit

ArchitectureMemory → ModuleMemory / FileMemory / SymbolMemory
                                      └─ DERIVED_FROM / SUPPORTED_BY → 原始证据
```

本图共 {summary["node_count"]:,} 个节点、{summary["edge_count"]:,} 条真实边，包含
{summary["function_count"]:,} 个 Function、{summary["code_segment_count"]:,} 个
CodeSegment、{summary["commit_count"]:,} 个 Commit 与 {summary["evolution_count"]:,} 个
Evolution。

## 权威数据与推荐入口

- `full_graph/nodes.jsonl`：标准 MemoryUnit 的 UID、raw_data、metadata、MemorySpace。
- `full_graph/edges.jsonl`：SemanticGraph typed edge 与完整 edge properties。
- `nodes_by_type/function.jsonl`：函数签名、源码 span、静态事实及调用点。
- `nodes_by_type/code_segment.jsonl`：分支、循环、异常处理等选择性片段。
- `edges_by_type/CALLS.jsonl`：含 `source_order`、`call_site_id`、源码位置和
  `enclosing_segment_uid` 的调用边；顺序仅表示词法/源码顺序。
- `nodes_by_type/evolution.jsonl`：独立的 Git/符号演化证据。
- `nodes_by_type/file_memory.jsonl` 与 `symbol_memory.jsonl`：可回溯的确定性高阶聚合。

## 完整性

- duplicate UID：{summary["duplicate_uid_count"]}
- dangling edge：{summary["dangling_edge_count"]}
- CALLS：{summary["CALLS_count"]}（带 source_order：{summary["ordered_CALLS_count"]}）
- 高阶记忆：{summary["high_level_memory_count"]}
""",
        encoding="utf-8",
    )


def export_real_code_graph(
    semantic_code: SemanticCode,
    output_dir: str | Path,
    *,
    selected_function_uid: str | None = None,
    repository_uid: str | None = None,
) -> dict[str, Any]:
    """Export one repository's real MemoryUnits and SemanticGraph edges.

    ``selected_function_uid`` is optional so a pre-experiment export can avoid
    encoding task-specific target knowledge. When supplied, the historical
    selected-case subgraph artifacts remain available for compatibility.
    """
    output = Path(output_dir)
    full_dir = output / "full_graph"
    nodes_dir = output / "nodes_by_type"
    edges_dir = output / "edges_by_type"
    selected_dir = output / "selected_case"
    figures_dir = output / "figures"
    directories = [full_dir, nodes_dir, edges_dir, figures_dir]
    if selected_function_uid is not None:
        directories.append(selected_dir)
    for directory in directories:
        directory.mkdir(parents=True, exist_ok=True)

    repo_uid = repository_uid or getattr(semantic_code, "repository_uid", None)
    units = [
        unit
        for unit in semantic_code.semantic_graph.semantic_map.get_all_units()
        if repo_uid is None or unit.metadata.get("repo_uid") == repo_uid
    ]
    unit_uids = {unit.uid for unit in units}
    if selected_function_uid is not None and selected_function_uid not in unit_uids:
        raise ValueError(
            f"Selected function is not in exported repository: {selected_function_uid}"
        )

    semantic_map = semantic_code.semantic_graph.semantic_map
    membership: dict[str, list[str]] = defaultdict(list)
    for name, space in semantic_map.memory_spaces.items():
        for uid in space.get_unit_uids():
            if uid in unit_uids:
                membership[uid].append(name)
    nodes = sorted(
        (_node_record(unit, sorted(membership.get(unit.uid, []))) for unit in units),
        key=lambda node: node["uid"],
    )
    nodes_by_uid = {node["uid"]: node for node in nodes}
    edges = _edge_records(semantic_code, unit_uids)

    _write_jsonl(full_dir / "nodes.jsonl", nodes)
    _write_jsonl(full_dir / "edges.jsonl", edges)
    with (full_dir / "nodes.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "uid",
                "software_type",
                "memory_spaces",
                "raw_data",
                "metadata",
                "has_dense_embedding",
                "has_sparse_embedding",
            ],
        )
        writer.writeheader()
        for node in nodes:
            writer.writerow(
                {
                    **node,
                    "memory_spaces": _json_text(node["memory_spaces"]),
                    "raw_data": _json_text(node["raw_data"]),
                    "metadata": _json_text(node["metadata"]),
                }
            )
    with (full_dir / "edges.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "edge_index",
                "source_uid",
                "target_uid",
                "relation",
                "properties",
            ],
        )
        writer.writeheader()
        for edge in edges:
            writer.writerow({**edge, "properties": _json_text(edge["properties"])})
    _write_graphml(full_dir / "graph.graphml", nodes, edges)

    nodes_by_type: dict[str, list[dict[str, Any]]] = defaultdict(list)
    edges_by_type: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for node in nodes:
        nodes_by_type[node["software_type"]].append(node)
    for edge in edges:
        edges_by_type[edge["relation"]].append(edge)
    for unit_type, records in sorted(nodes_by_type.items()):
        _write_jsonl(nodes_dir / f"{unit_type}.jsonl", records)
    for relation, records in sorted(edges_by_type.items()):
        _write_jsonl(edges_dir / f"{relation}.jsonl", records)

    target: dict[str, Any] | None = None
    segments: list[dict[str, Any]] = []
    calls: list[dict[str, Any]] = []
    local: dict[str, Any] = {"target_uid": None, "nodes": [], "edges": []}
    if selected_function_uid is not None:
        target = nodes_by_uid[selected_function_uid]
        segments = [
            node
            for node in nodes
            if node["software_type"] == "code_segment"
            and node["raw_data"].get("parent_uid", node["raw_data"].get("function_uid"))
            == selected_function_uid
        ]
        calls = [
            edge
            for edge in edges
            if edge["source_uid"] == selected_function_uid
            and edge["relation"] == "CALLS"
        ]
        calls.sort(
            key=lambda edge: (
                edge["properties"].get("source_order", 0),
                edge["edge_index"],
            )
        )
        local = _local_subgraph(selected_function_uid, nodes_by_uid, edges)
        provenance = _provenance_subgraph(local)
        _write_json(selected_dir / "target_function.json", target)
        _write_json(selected_dir / "target_function_segments.json", segments)
        _write_json(selected_dir / "ordered_calls.json", calls)
        _write_json(selected_dir / "local_subgraph.json", local)
        _write_graphml(
            selected_dir / "local_subgraph.graphml", local["nodes"], local["edges"]
        )
        _write_json(selected_dir / "provenance_subgraph.json", provenance)

    node_counts = Counter(node["software_type"] for node in nodes)
    edge_counts = Counter(edge["relation"] for edge in edges)
    repeated_groups: dict[tuple[str, str], list[int]] = defaultdict(list)
    for edge in edges_by_type.get("CALLS", []):
        order = edge["properties"].get("source_order")
        if order is not None:
            repeated_groups[(edge["source_uid"], edge["target_uid"])].append(int(order))
    repeated = [
        (pair, sorted(orders))
        for pair, orders in repeated_groups.items()
        if len(orders) > 1
    ]
    repeated.sort(key=lambda item: (-len(item[1]), item[0]))
    repo = nodes_by_uid.get(str(repo_uid), {})
    high_count = sum(node_counts.get(kind, 0) for kind in _HIGH_LEVEL_TYPES)
    all_uids = [unit.uid for unit in units]
    summary = {
        "repository": repo.get("raw_data", {}).get("name", str(repo_uid)),
        "repository_uid": repo_uid,
        "revision": repo.get("raw_data", {}).get("revision", ""),
        "node_count": len(nodes),
        "edge_count": len(edges),
        "nodes_by_type": dict(sorted(node_counts.items())),
        "edges_by_type": dict(sorted(edge_counts.items())),
        "memory_spaces": {
            name: {
                "direct_unit_count": len(space.get_unit_uids() & unit_uids),
                "child_spaces": sorted(space.get_child_space_names()),
            }
            for name, space in sorted(semantic_map.memory_spaces.items())
            if space.get_unit_uids() & unit_uids or name.startswith("software")
        },
        "function_count": node_counts.get("function", 0),
        "code_segment_count": node_counts.get("code_segment", 0),
        "commit_count": node_counts.get("commit", 0),
        "evolution_count": node_counts.get("evolution", 0),
        "high_level_memory_count": high_count,
        "CALLS_count": edge_counts.get("CALLS", 0),
        "ordered_CALLS_count": sum(
            "source_order" in edge["properties"]
            for edge in edges_by_type.get("CALLS", [])
        ),
        "repeated_call_pair_count": len(repeated),
        "repeated_call_example": (
            {
                "source_uid": repeated[0][0][0],
                "target_uid": repeated[0][0][1],
                "source_orders": repeated[0][1],
            }
            if repeated
            else None
        ),
        "dangling_edge_count": sum(
            edge["source_uid"] not in unit_uids or edge["target_uid"] not in unit_uids
            for edge in edges
        ),
        "duplicate_uid_count": len(all_uids) - len(set(all_uids)),
        "selected_function_uid": selected_function_uid,
        "selected_function_code_segments": len(segments),
        "selected_function_parsed_call_sites": (
            len(target["raw_data"].get("call_sites", [])) if target else 0
        ),
        "selected_function_ordered_calls": len(calls),
        "selected_local_node_count": len(local["nodes"]),
        "selected_local_edge_count": len(local["edges"]),
    }
    _write_json(output / "graph_summary.json", summary)
    if target is None:
        _write_repository_readme(output / "README.md", summary)
    else:
        _write_readme(output / "README.md", summary, target, segments, calls, local)
    return summary


__all__ = ["export_real_code_graph"]
