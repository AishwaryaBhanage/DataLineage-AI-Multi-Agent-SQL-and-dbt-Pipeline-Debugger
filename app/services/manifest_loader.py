"""
Manifest Loader — Phase 4
Reads dbt's target/manifest.json and exposes structured model metadata
and dependency helpers used by the lineage builder and rule engine.
"""

import json
from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class ModelNode:
    unique_id: str          # e.g. "model.dbt_demo.stg_orders"
    name: str               # e.g. "stg_orders"
    package: str            # e.g. "dbt_demo"
    schema: str             # e.g. "main"
    path: str               # e.g. "models/stg_orders.sql"
    materialized: str       # "table" | "view" | "incremental" | "ephemeral"
    raw_sql: str            # original SQL source code
    depends_on: list[str]   # list of unique_ids this model depends on
    columns: dict[str, str] # column_name -> description (empty if not documented)
    refs: list[str]         # friendly ref names e.g. ["raw_orders"]


class ManifestLoader:
    """
    Loads a dbt manifest.json and provides lookup helpers.
    All methods operate on model names (short names like "stg_orders"),
    not full unique_ids.
    """

    def __init__(self, manifest_path: str | Path):
        self._path = Path(manifest_path)
        self._raw: dict = self._load()
        self._nodes: dict[str, ModelNode] = self._parse_models()
        # name -> unique_id reverse index
        self._name_index: dict[str, str] = {
            n.name: n.unique_id for n in self._nodes.values()
        }

    # ── loading ──────────────────────────────────────────────────────────────

    def _load(self) -> dict:
        if not self._path.exists():
            raise FileNotFoundError(
                f"manifest.json not found at {self._path}\n"
                "Run `dbt compile` or `dbt run` inside your dbt project first."
            )
        with open(self._path) as f:
            return json.load(f)

    def _parse_models(self) -> dict[str, ModelNode]:
        nodes = {}
        for uid, raw in self._raw.get("nodes", {}).items():
            if raw.get("resource_type") != "model":
                continue
            nodes[uid] = ModelNode(
                unique_id=uid,
                name=raw["name"],
                package=raw.get("package_name", ""),
                schema=raw.get("schema", ""),
                path=raw.get("original_file_path", ""),
                materialized=raw.get("config", {}).get("materialized", "table"),
                raw_sql=raw.get("raw_code", ""),
                depends_on=raw.get("depends_on", {}).get("nodes", []),
                columns={
                    col: meta.get("description", "")
                    for col, meta in raw.get("columns", {}).items()
                },
                refs=[r["name"] for r in raw.get("refs", [])],
            )
        return nodes

    # ── public helpers ────────────────────────────────────────────────────────

    def all_model_names(self) -> list[str]:
        """Return all model names in the project."""
        return [n.name for n in self._nodes.values()]

    def get_model(self, name: str) -> ModelNode | None:
        """Look up a ModelNode by short name (e.g. 'stg_orders')."""
        uid = self._name_index.get(name)
        return self._nodes.get(uid) if uid else None

    def get_upstream(self, name: str) -> list[str]:
        """
        Return the direct upstream dependencies of a model (one level up).
        e.g. get_upstream('stg_orders') -> ['raw_orders']
        """
        node = self.get_model(name)
        if not node:
            return []
        result = []
        for dep_uid in node.depends_on:
            dep = self._nodes.get(dep_uid)
            if dep:
                result.append(dep.name)
        return result

    def get_downstream(self, name: str) -> list[str]:
        """
        Return all models that directly depend on this model (one level down).
        e.g. get_downstream('stg_orders') -> ['customer_revenue']
        """
        uid = self._name_index.get(name)
        if not uid:
            return []
        result = []
        for node in self._nodes.values():
            if uid in node.depends_on:
                result.append(node.name)
        return result

    def get_all_upstream(self, name: str, visited: set | None = None) -> list[str]:
        """
        Recursively return ALL ancestors of a model (full upstream chain).
        e.g. get_all_upstream('customer_revenue') -> ['stg_orders', 'raw_orders']
        """
        if visited is None:
            visited = set()
        direct = self.get_upstream(name)
        result = []
        for parent in direct:
            if parent not in visited:
                visited.add(parent)
                result.append(parent)
                result.extend(self.get_all_upstream(parent, visited))
        return result

    def get_all_downstream(self, name: str, visited: set | None = None) -> list[str]:
        """
        Recursively return ALL descendants of a model (full downstream chain).
        e.g. get_all_downstream('raw_orders') -> ['stg_orders', 'customer_revenue']
        """
        if visited is None:
            visited = set()
        direct = self.get_downstream(name)
        result = []
        for child in direct:
            if child not in visited:
                visited.add(child)
                result.append(child)
                result.extend(self.get_all_downstream(child, visited))
        return result

    def get_columns(self, name: str) -> dict[str, str]:
        """
        Return documented columns for a model {col_name: description}.
        Note: columns are only present if documented in schema.yml.
        For runtime columns use the SQL parser on raw_sql.
        """
        node = self.get_model(name)
        return node.columns if node else {}

    def get_sql(self, name: str) -> str:
        """Return the raw SQL source code of a model."""
        node = self.get_model(name)
        return node.raw_sql if node else ""

    def exists(self, name: str) -> bool:
        """Check if a model with this name exists in the manifest."""
        return name in self._name_index

    def get_impacted_by(self, name: str) -> list[str]:
        """
        Given a broken model, return all downstream models that are also impacted.
        Useful for blast-radius analysis.
        """
        return self.get_all_downstream(name)

    def summary(self) -> dict:
        """Return a compact summary of the entire project."""
        return {
            "total_models": len(self._nodes),
            "models": [
                {
                    "name": n.name,
                    "upstream": self.get_upstream(n.name),
                    "downstream": self.get_downstream(n.name),
                    "materialized": n.materialized,
                }
                for n in self._nodes.values()
            ],
        }
