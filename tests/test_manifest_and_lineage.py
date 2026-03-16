"""
Tests for manifest_loader.py and lineage_builder.py
Run with: python -m pytest tests/test_manifest_and_lineage.py -v
"""

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pytest
from app.services.manifest_loader import ManifestLoader
from app.services.lineage_builder import LineageGraph

MANIFEST = "dbt_demo/target/manifest.json"


@pytest.fixture
def loader():
    return ManifestLoader(MANIFEST)

@pytest.fixture
def graph(loader):
    return LineageGraph(loader)


# ── ManifestLoader tests ──────────────────────────────────────────────────────

def test_all_models_found(loader):
    names = loader.all_model_names()
    assert "raw_orders" in names
    assert "stg_orders" in names
    assert "customer_revenue" in names

def test_get_model_returns_node(loader):
    node = loader.get_model("stg_orders")
    assert node is not None
    assert node.name == "stg_orders"
    assert node.materialized == "table"

def test_get_model_unknown_returns_none(loader):
    assert loader.get_model("does_not_exist") is None

def test_get_upstream_direct(loader):
    assert loader.get_upstream("stg_orders") == ["raw_orders"]
    assert loader.get_upstream("customer_revenue") == ["stg_orders"]
    assert loader.get_upstream("raw_orders") == []

def test_get_downstream_direct(loader):
    assert loader.get_downstream("stg_orders") == ["customer_revenue"]
    assert loader.get_downstream("raw_orders") == ["stg_orders"]
    assert loader.get_downstream("customer_revenue") == []

def test_get_all_upstream(loader):
    chain = loader.get_all_upstream("customer_revenue")
    assert "stg_orders" in chain
    assert "raw_orders" in chain

def test_get_all_downstream(loader):
    chain = loader.get_all_downstream("raw_orders")
    assert "stg_orders" in chain
    assert "customer_revenue" in chain

def test_raw_sql_present(loader):
    sql = loader.get_sql("customer_revenue")
    assert "sum(amount)" in sql.lower()

def test_exists(loader):
    assert loader.exists("stg_orders") is True
    assert loader.exists("nonexistent") is False

def test_impacted_by(loader):
    impacted = loader.get_impacted_by("stg_orders")
    assert "customer_revenue" in impacted

def test_summary_shape(loader):
    s = loader.summary()
    assert "total_models" in s
    assert s["total_models"] == 3


# ── LineageGraph tests ────────────────────────────────────────────────────────

def test_graph_upstream(graph):
    assert graph.upstream("customer_revenue") == ["stg_orders"]
    assert graph.upstream("stg_orders") == ["raw_orders"]
    assert graph.upstream("raw_orders") == []

def test_graph_downstream(graph):
    assert graph.downstream("raw_orders") == ["stg_orders"]
    assert graph.downstream("stg_orders") == ["customer_revenue"]
    assert graph.downstream("customer_revenue") == []

def test_graph_all_upstream_order(graph):
    chain = graph.all_upstream("customer_revenue")
    # stg_orders should appear before raw_orders (closest first)
    assert chain.index("stg_orders") < chain.index("raw_orders")

def test_path_to_root(graph):
    paths = graph.path_to_root("customer_revenue")
    assert len(paths) == 1
    assert paths[0] == ["raw_orders", "stg_orders", "customer_revenue"]

def test_root_and_leaf(graph):
    assert "raw_orders" in graph.root_models()
    assert "customer_revenue" in graph.leaf_models()

def test_impacted_models(graph):
    assert "customer_revenue" in graph.impacted_models("stg_orders")

def test_to_dict_scoped(graph):
    result = graph.to_dict("stg_orders")
    assert "stg_orders" in result["nodes"]
    assert "raw_orders" in result["nodes"]
    assert "customer_revenue" in result["nodes"]
    assert len(result["edges"]) == 2
    assert result["upstream"] == ["stg_orders", "raw_orders"] or "raw_orders" in result["upstream"]

def test_ascii_output(graph):
    tree = graph.ascii()
    assert "raw_orders" in tree
    assert "stg_orders" in tree
    assert "customer_revenue" in tree
    # raw_orders should appear before its children
    assert tree.index("raw_orders") < tree.index("stg_orders")
