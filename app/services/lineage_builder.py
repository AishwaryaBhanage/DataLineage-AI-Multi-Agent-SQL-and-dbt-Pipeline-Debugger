"""
Lineage Builder — Phase 4
Uses networkx to build a DAG from ManifestLoader data.
Provides graph traversal, path finding, and ASCII/dict visualisation.
"""

import networkx as nx
from app.services.manifest_loader import ManifestLoader


class LineageGraph:
    """
    Directed Acyclic Graph where edges point FROM upstream TO downstream.
    e.g.  raw_orders --> stg_orders --> customer_revenue
    """

    def __init__(self, loader: ManifestLoader):
        self._loader = loader
        self._graph: nx.DiGraph = self._build()

    # ── build ─────────────────────────────────────────────────────────────────

    def _build(self) -> nx.DiGraph:
        G = nx.DiGraph()
        for name in self._loader.all_model_names():
            G.add_node(name)
            for parent in self._loader.get_upstream(name):
                G.add_edge(parent, name)   # parent → child
        return G

    # ── traversal ─────────────────────────────────────────────────────────────

    def upstream(self, model: str) -> list[str]:
        """Direct parents of model."""
        return list(self._graph.predecessors(model))

    def downstream(self, model: str) -> list[str]:
        """Direct children of model."""
        return list(self._graph.successors(model))

    def all_upstream(self, model: str) -> list[str]:
        """All ancestors in topological order (closest first)."""
        if model not in self._graph:
            return []
        ancestors = nx.ancestors(self._graph, model)
        subgraph = self._graph.subgraph(ancestors | {model})
        order = list(nx.topological_sort(subgraph))
        order = [n for n in order if n != model]
        return list(reversed(order))   # closest ancestor first

    def all_downstream(self, model: str) -> list[str]:
        """All descendants in topological order (closest first)."""
        if model not in self._graph:
            return []
        descendants = nx.descendants(self._graph, model)
        subgraph = self._graph.subgraph(descendants | {model})
        order = list(nx.topological_sort(subgraph))
        return [n for n in order if n != model]

    def path_to_root(self, model: str) -> list[list[str]]:
        """
        Return all paths from model back to root source nodes.
        Useful for explaining the full ancestry chain.
        """
        roots = [n for n in self._graph.nodes if self._graph.in_degree(n) == 0]
        paths = []
        for root in roots:
            if nx.has_path(self._graph, root, model):
                for path in nx.all_simple_paths(self._graph, root, model):
                    paths.append(path)
        return paths

    def impacted_models(self, broken_model: str) -> list[str]:
        """All downstream models broken if this model fails."""
        return self.all_downstream(broken_model)

    # ── inspection ────────────────────────────────────────────────────────────

    def exists(self, model: str) -> bool:
        return model in self._graph

    def root_models(self) -> list[str]:
        """Models with no upstream dependencies (source layer)."""
        return [n for n in self._graph.nodes if self._graph.in_degree(n) == 0]

    def leaf_models(self) -> list[str]:
        """Models with no downstream dependents (final layer)."""
        return [n for n in self._graph.nodes if self._graph.out_degree(n) == 0]

    # ── output formats ────────────────────────────────────────────────────────

    def to_dict(self, model: str | None = None) -> dict:
        """
        Return lineage as a serialisable dict.
        If model is given, scope it to that model's context.
        """
        if model:
            relevant = (
                set(self.all_upstream(model))
                | {model}
                | set(self.all_downstream(model))
            )
            nodes = list(relevant)
            edges = [
                {"from": u, "to": v}
                for u, v in self._graph.edges
                if u in relevant and v in relevant
            ]
        else:
            nodes = list(self._graph.nodes)
            edges = [{"from": u, "to": v} for u, v in self._graph.edges]

        return {
            "nodes": nodes,
            "edges": edges,
            "upstream": self.all_upstream(model) if model else [],
            "downstream": self.all_downstream(model) if model else [],
            "impacted": self.impacted_models(model) if model else [],
            "paths_to_root": self.path_to_root(model) if model else [],
        }

    def ascii(self) -> str:
        """
        Print the full lineage as a simple ASCII tree rooted at source models.
        e.g.
          raw_orders
            └── stg_orders
                  └── customer_revenue
        """
        lines = []
        visited = set()

        def draw(node: str, indent: int):
            if node in visited:
                return
            visited.add(node)
            prefix = "  " * indent + ("└── " if indent > 0 else "")
            lines.append(f"{prefix}{node}")
            for child in self._graph.successors(node):
                draw(child, indent + 1)

        for root in self.root_models():
            draw(root, 0)

        return "\n".join(lines)
