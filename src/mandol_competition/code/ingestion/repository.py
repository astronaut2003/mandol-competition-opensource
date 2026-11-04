"""Transient builder view. SQLite remains the only semantic persistence.

The frozen builder mutates this disposable representation; its output is an
ordinary PreparedMutation. No embeddings, FAISS, pickle, or private database.
"""
from copy import deepcopy
from dataclasses import asdict, replace
import json

import rustworkx as rx
from mandol_substrate.contracts import PreparedMutation
from mandol_substrate.representation import MemoryUnit
from mandol_substrate.state import CanonicalMemory, Mutation
from mandol_substrate.state.models import Edge, digest


class RepositoryPreparation:
    def __init__(self, state=None, scope=None):
        self.semantic_map = self
        self.units, self.memberships, self.spaces, self.space_edges = {}, {}, set(), set()
        self.rx_graph = rx.PyDiGraph(multigraph=True)
        self._uid_to_index, self._index_to_uid = {}, {}
        self.baseline, self.baseline_edges = {}, {}
        if state is not None:
            for batch in state.iter_memories(scope):
                for memory in batch:
                    self.baseline[memory.logical_uid] = memory
                    self.add_unit(MemoryUnit(memory.logical_uid, deepcopy(memory.raw_data), deepcopy(memory.metadata)), space_names=memory.spaces)
            for batch in state.graph_rows(scope, 'edges'):
                for row in batch:
                    if row['source_kind'] == row['target_kind'] == 'memory':
                        edge = Edge(row['edge_id'], row['source_uid'], row['target_uid'], row['relation_type'],
                                    json.loads(row['properties']), json.loads(row['provenance']), version=row['version'])
                        self.baseline_edges[edge.edge_id] = edge
                        self.add_relationship(edge.source_uid, edge.target_uid, edge.relation_type, **edge.properties)

    def create_memory_space(self, name):
        self.spaces.add(name)

    def add_space_to_space(self, child, parent):
        self.space_edges.add((parent, child))

    def add_unit(self, unit, *, space_names=(), **_):
        if unit.uid not in self.units:
            index = self.rx_graph.add_node(unit.uid)
            self._uid_to_index[unit.uid] = index
            self._index_to_uid[index] = unit.uid
        self.units[unit.uid] = unit
        self.memberships[unit.uid] = set(space_names)
        self.spaces.update(space_names)

    def batch_add_units(self, units, *, per_unit_space_names, **_):
        for unit, spaces in zip(units, per_unit_space_names, strict=True):
            self.add_unit(unit, space_names=spaces)
        return {'embedding_generated': 0, 'embedding_batches': 0}

    def get_unit(self, uid):
        return self.units.get(uid)

    def get_all_units(self):
        return list(self.units.values())

    def delete_unit(self, uid):
        if uid not in self.units:
            return False
        del self.units[uid]
        del self.memberships[uid]
        index = self._uid_to_index.pop(uid)
        del self._index_to_uid[index]
        self.rx_graph.remove_node(index)
        return True

    def get_units_in_memory_space(self, space, recursive=False):
        spaces = {space}
        if recursive:
            while True:
                new = spaces | {child for parent, child in self.space_edges if parent in spaces}
                if new == spaces:
                    break
                spaces = new
        return [unit for uid, unit in self.units.items() if self.memberships[uid] & spaces]

    def add_relationship(self, source, target, relationship_type, **properties):
        if source not in self.units or target not in self.units:
            return False
        self.rx_graph.add_edge(self._uid_to_index[source], self._uid_to_index[target], dict(properties, type=relationship_type))
        return True

    def get_edge_data(self, source, target):
        if source not in self.units or target not in self.units:
            return []
        return list(self.rx_graph.get_all_edge_data(self._uid_to_index[source], self._uid_to_index[target])) if self.rx_graph.has_edge(self._uid_to_index[source], self._uid_to_index[target]) else []

    def get_relationship(self, source, target, relation):
        return next((data for data in self.get_edge_data(source, target) if data.get('type') == relation), None)

    def delete_relationship(self, source, target, relation):
        for index in self.rx_graph.edge_indices():
            a, b = self.rx_graph.get_edge_endpoints_by_index(index)
            if (self._index_to_uid[a], self._index_to_uid[b]) == (source, target) and self.rx_graph.get_edge_data_by_index(index)['type'] == relation:
                self.rx_graph.remove_edge_from_index(index)
                return True
        return False

    def get_successors(self, uid):
        return [self._index_to_uid[i] for i in self.rx_graph.successor_indices(self._uid_to_index[uid])]

    def get_predecessors(self, uid):
        return [self._index_to_uid[i] for i in self.rx_graph.predecessor_indices(self._uid_to_index[uid])]

    def traverse_explicit_nodes(self, uid, relationship_type=None, direction='all'):
        result = {}
        for index in self.rx_graph.edge_indices():
            a, b = self.rx_graph.get_edge_endpoints_by_index(index)
            source, target = self._index_to_uid[a], self._index_to_uid[b]
            data = self.rx_graph.get_edge_data_by_index(index)
            if relationship_type is not None and data['type'] != relationship_type:
                continue
            if source == uid and direction in ('all', 'successors'):
                result[target] = self.units[target]
            if target == uid and direction in ('all', 'predecessors'):
                result[source] = self.units[source]
        return list(result.values())

    def prepare(self, request_id, payload_digest):
        memories = []
        for uid, unit in sorted(self.units.items()):
            old = self.baseline.get(uid)
            spaces = tuple(sorted(self.memberships[uid]))
            if old and old.raw_data == unit.raw_data and old.metadata == unit.metadata and set(old.spaces) == set(spaces):
                continue
            memory = CanonicalMemory(uid, unit.text_cached, raw_data=unit.raw_data, metadata=unit.metadata,
                memory_tower='software', memory_level='repository', memory_type=unit.metadata.get('software_type', 'code'),
                spaces=spaces, version=old.version + 1 if old else 1)
            memories.append(memory)
        edges = {}
        for index in self.rx_graph.edge_indices():
            a, b = self.rx_graph.get_edge_endpoints_by_index(index)
            source, target = self._index_to_uid[a], self._index_to_uid[b]
            properties = dict(self.rx_graph.get_edge_data_by_index(index))
            relation = properties.pop('type')
            edge_id = 'relation_' + digest([source, target, relation, properties])
            edges[edge_id] = Edge(edge_id, source, target, relation, properties,
                                  provenance={'adapter': 'repository-builder-v6'})
        mutation = Mutation(memories=tuple(memories), delete_uids=tuple(sorted(self.baseline.keys() - self.units.keys())),
            spaces=tuple(sorted(self.spaces)), space_edges=tuple(sorted(self.space_edges)),
            edges=tuple(edge for key, edge in edges.items() if key not in self.baseline_edges),
            delete_edge_ids=tuple(sorted(self.baseline_edges.keys() - edges.keys())), kind='repository')
        return PreparedMutation(request_id, payload_digest, mutation, {'memories': len(memories), 'relations': len(edges)})


def prepare_repository(path, *, state, scope, request_id, payload_digest, cache_path, **options):
    """Caller holds scope write admission and publishes returned mutation.

    Cache must live in the fresh runtime root, never inside an input repository.
    Indexing is the common projection coordinator's job, after canonical prepare.
    """
    from ..software.builder.repository_builder import RepositoryMemoryBuilder
    from ..software.facade import SemanticCode
    view = RepositoryPreparation(state, scope)
    facade = SemanticCode(view)
    RepositoryMemoryBuilder(path, semantic_code=facade, cache_path=cache_path,
                            build_indexes=False, **options).build()
    return view.prepare(request_id, payload_digest)
