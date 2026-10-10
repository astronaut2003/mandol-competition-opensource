"""Execute archived algorithm definitions without importing their model runtime.

Only test oracles use this loader. Production never imports legacy files.
Function bodies are compiled unchanged; deferred annotations avoid pulling
unrelated SemanticMap/model classes into pure structural graph comparisons.
"""

import __future__
import ast
from collections import Counter, defaultdict, deque
from dataclasses import dataclass
from datetime import datetime
import logging
from pathlib import Path
import sys
from types import ModuleType
import uuid

import numpy as np
import rustworkx as rx

from mandol_substrate.representation import MemoryUnit


def graph_oracles():
    module = ModuleType("_archived_graph_oracle")
    sys.modules[module.__name__] = module
    module.__dict__.update(np=np, rx=rx, Counter=Counter, defaultdict=defaultdict, deque=deque,
        dataclass=dataclass, datetime=datetime, uuid=uuid, logger=logging.getLogger(module.__name__),
        MemoryUnit=MemoryUnit)
    root = Path(__file__).resolve().parents[1] / "tests/oracles/mandol"
    for relative in ("core/semantic_graph.py", "retrieval/graph_context_expander.py"):
        tree = ast.parse((root / relative).read_text())
        tree.body = [node for node in tree.body if isinstance(node, ast.ClassDef)]
        exec(compile(tree, str(root / relative), "exec", flags=__future__.annotations.compiler_flag), module.__dict__)
    return module.SemanticGraph, module.GraphContextExpander
