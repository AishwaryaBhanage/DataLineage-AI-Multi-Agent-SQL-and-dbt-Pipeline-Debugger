"""
Tests for error_parser.py and rule_engine.py
Run with: python -m pytest tests/test_error_and_rules.py -v
"""

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pytest
from app.services.error_parser import parse_error, ErrorType
from app.services.sql_parser import parse_sql
from app.services.manifest_loader import ManifestLoader
from app.services.rule_engine import run_rules

MANIFEST = "dbt_demo/target/manifest.json"


# ══════════════════════════════════════════════════════════════════════════════
# Error Parser Tests
# ══════════════════════════════════════════════════════════════════════════════

def test_duckdb_missing_column():
    err = """
    Runtime Error in model customer_revenue (models/customer_revenue.sql)
    Binder Error: Referenced column "amount" not found in FROM clause!
    Candidate bindings: "amount_total", "status", "order_date", "customer_id"
    LINE 15:     sum(amount) as total_revenue,
    """
    result = parse_error(err)
    assert result.error_type == ErrorType.MISSING_COLUMN
    assert result.column == "amount"
    assert result.model == "customer_revenue"
    assert result.line_number == 15
    assert "amount_total" in result.candidates


def test_postgres_missing_column():
    err = 'ERROR: column "total_amount" does not exist at character 42'
    result = parse_error(err)
    assert result.error_type == ErrorType.MISSING_COLUMN
    assert result.column == "total_amount"


def test_snowflake_invalid_identifier():
    err = "SQL compilation error: invalid identifier 'AMOUNT'"
    result = parse_error(err)
    assert result.error_type == ErrorType.MISSING_COLUMN
    assert result.column == "AMOUNT"


def test_bigquery_unrecognized_name():
    err = "Unrecognized name: revenue at [3:5]"
    result = parse_error(err)
    assert result.error_type == ErrorType.MISSING_COLUMN
    assert result.column == "revenue"


def test_missing_relation_postgres():
    err = 'ERROR: relation "stg_orders" does not exist'
    result = parse_error(err)
    assert result.error_type == ErrorType.MISSING_RELATION
    assert result.relation == "stg_orders"


def test_missing_ref_dbt():
    err = "Compilation Error: depends on a node named 'raw_orders' which was not found"
    result = parse_error(err)
    assert result.error_type == ErrorType.MISSING_RELATION


def test_ambiguous_column_postgres():
    err = 'ERROR: column reference "id" is ambiguous'
    result = parse_error(err)
    assert result.error_type == ErrorType.AMBIGUOUS_COLUMN
    assert result.column == "id"


def test_ambiguous_column_duckdb():
    err = 'Binder Error: Ambiguous reference to column name "customer_id"'
    result = parse_error(err)
    assert result.error_type == ErrorType.AMBIGUOUS_COLUMN
    assert result.column == "customer_id"


def test_type_mismatch():
    err = "operator does not exist: text = integer"
    result = parse_error(err)
    assert result.error_type == ErrorType.TYPE_MISMATCH


def test_null_violation():
    err = "null value in column violates not-null constraint"
    result = parse_error(err)
    assert result.error_type == ErrorType.NULL_VIOLATION


def test_syntax_error():
    err = "syntax error at or near 'FROM'"
    result = parse_error(err)
    assert result.error_type == ErrorType.SYNTAX_ERROR


def test_unknown_error():
    result = parse_error("something completely unexpected happened")
    assert result.error_type == ErrorType.UNKNOWN


def test_to_dict():
    result = parse_error('column "x" does not exist')
    d = result.to_dict()
    assert "error_type" in d
    assert "column" in d
    assert d["column"] == "x"


# ══════════════════════════════════════════════════════════════════════════════
# Rule Engine Tests
# ══════════════════════════════════════════════════════════════════════════════

@pytest.fixture
def loader():
    return ManifestLoader(MANIFEST)


def test_rename_rule_fires(loader):
    """Our exact scenario: amount used, amount_total exists upstream."""
    sql = open("dbt_demo/models/customer_revenue.sql").read()
    err = open("data/sample_errors/customer_revenue_error.txt").read()

    parsed_sql   = parse_sql(sql)
    parsed_error = parse_error(err)
    hits = run_rules(parsed_sql, parsed_error, loader, "customer_revenue")

    assert len(hits) > 0
    top = hits[0]
    assert top.cause == "column_renamed_upstream"
    assert top.confidence >= 0.85
    assert top.suggested_column == "amount_total"
    assert top.upstream_model == "stg_orders"


def test_rename_rule_top_confidence(loader):
    """Rename rule should be the highest confidence hit."""
    sql = open("dbt_demo/models/customer_revenue.sql").read()
    err = open("data/sample_errors/customer_revenue_error.txt").read()
    hits = run_rules(parse_sql(sql), parse_error(err), loader, "customer_revenue")
    assert hits[0].confidence == max(h.confidence for h in hits)


def test_missing_relation_rule(loader):
    sql = "select id from {{ ref('nonexistent_model') }}"
    err = "relation \"nonexistent_model\" does not exist"
    hits = run_rules(parse_sql(sql), parse_error(err), loader, "customer_revenue")
    causes = [h.cause for h in hits]
    assert "missing_ref_model" in causes or "missing_relation" in causes


def test_ambiguous_column_rule(loader):
    sql = """
    select id, name
    from orders o
    join customers c on o.customer_id = c.id
    """
    err = 'column reference "id" is ambiguous'
    hits = run_rules(parse_sql(sql), parse_error(err), loader, "customer_revenue")
    causes = [h.cause for h in hits]
    assert "ambiguous_column" in causes


def test_type_mismatch_rule(loader):
    sql = "select id from orders where amount = 'abc'"
    err = "operator does not exist: integer = text"
    hits = run_rules(parse_sql(sql), parse_error(err), loader, "customer_revenue")
    causes = [h.cause for h in hits]
    assert "type_mismatch" in causes


def test_hits_sorted_by_confidence(loader):
    sql = open("dbt_demo/models/customer_revenue.sql").read()
    err = open("data/sample_errors/customer_revenue_error.txt").read()
    hits = run_rules(parse_sql(sql), parse_error(err), loader, "customer_revenue")
    confs = [h.confidence for h in hits]
    assert confs == sorted(confs, reverse=True)


def test_hit_to_dict(loader):
    sql = open("dbt_demo/models/customer_revenue.sql").read()
    err = open("data/sample_errors/customer_revenue_error.txt").read()
    hits = run_rules(parse_sql(sql), parse_error(err), loader, "customer_revenue")
    d = hits[0].to_dict()
    for key in ("cause", "title", "evidence", "confidence", "fix_hint"):
        assert key in d
