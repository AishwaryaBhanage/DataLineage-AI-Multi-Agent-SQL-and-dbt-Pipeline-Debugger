"""
Tests for debug_pipeline.py and llm_service.py (no real API key needed).
Run with: python -m pytest tests/test_pipeline.py -v
"""

import sys, os, json
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pytest
from unittest.mock import patch, MagicMock

from app.workflows.debug_pipeline import run_pipeline, PipelineResult
from app.services.llm_service import _parse_response, LLMResult

MANIFEST = "dbt_demo/target/manifest.json"
SQL      = open("dbt_demo/models/customer_revenue.sql").read()
ERROR    = open("data/sample_errors/customer_revenue_error.txt").read()


# ── pipeline without LLM ──────────────────────────────────────────────────────

def test_pipeline_runs_without_llm():
    result = run_pipeline(
        sql=SQL,
        error_text=ERROR,
        manifest_path=MANIFEST,
        broken_model="customer_revenue",
        use_llm=False,
    )
    assert isinstance(result, PipelineResult)
    assert result.broken_model == "customer_revenue"
    assert result.llm_result is None
    assert result.error is None


def test_pipeline_parsed_sql_populated(  ):
    result = run_pipeline(SQL, ERROR, MANIFEST, "customer_revenue", use_llm=False)
    assert "amount" in result.parsed_sql.columns
    assert "stg_orders" in result.parsed_sql.dbt_refs


def test_pipeline_parsed_error_populated():
    result = run_pipeline(SQL, ERROR, MANIFEST, "customer_revenue", use_llm=False)
    assert result.parsed_error.column == "amount"
    assert result.parsed_error.model == "customer_revenue"


def test_pipeline_lineage_populated():
    result = run_pipeline(SQL, ERROR, MANIFEST, "customer_revenue", use_llm=False)
    assert "stg_orders" in result.lineage["nodes"]
    assert "raw_orders" in result.lineage["nodes"]
    assert "raw_orders" in result.lineage_ascii


def test_pipeline_rule_hits_present():
    result = run_pipeline(SQL, ERROR, MANIFEST, "customer_revenue", use_llm=False)
    assert len(result.rule_hits) > 0
    assert result.rule_hits[0].cause == "column_renamed_upstream"
    assert result.rule_hits[0].suggested_column == "amount_total"


def test_pipeline_upstream_columns_present():
    result = run_pipeline(SQL, ERROR, MANIFEST, "customer_revenue", use_llm=False)
    assert "stg_orders" in result.upstream_columns
    assert "amount_total" in result.upstream_columns["stg_orders"]


def test_pipeline_to_dict_serialisable():
    result = run_pipeline(SQL, ERROR, MANIFEST, "customer_revenue", use_llm=False)
    d = result.to_dict()
    # Must be JSON-serialisable
    json.dumps(d)
    assert "broken_model" in d
    assert "rule_hits" in d
    assert "lineage" in d


def test_pipeline_broken_model_auto_detected():
    # Don't pass broken_model — it should be read from the error text
    result = run_pipeline(SQL, ERROR, MANIFEST, use_llm=False)
    assert result.broken_model == "customer_revenue"


def test_pipeline_missing_api_key_graceful():
    """Pipeline should not crash when API key is missing — it sets error field."""
    result = run_pipeline(SQL, ERROR, MANIFEST, "customer_revenue", use_llm=True)
    # With a dummy key in .env it should set error, not raise
    if result.llm_result is None:
        assert result.error is not None


# ── LLM response parser (no API call) ────────────────────────────────────────

SAMPLE_CLAUDE_RESPONSE = json.dumps({
    "root_cause": "Column 'amount' was renamed to 'amount_total' in stg_orders",
    "explanation": (
        "The model customer_revenue references column 'amount' but the upstream "
        "model stg_orders renamed it to 'amount_total'. This is a classic schema "
        "drift issue where a downstream model was not updated after an upstream rename."
    ),
    "corrected_sql": (
        "select\n"
        "    customer_id,\n"
        "    sum(amount_total) as total_revenue,\n"
        "    count(order_id) as order_count,\n"
        "    max(order_date) as last_order_date\n"
        "from {{ ref('stg_orders') }}\n"
        "group by customer_id"
    ),
    "confidence_score": 0.95,
    "ranked_causes": [
        {"cause": "column_renamed_upstream", "title": "Column renamed in stg_orders", "confidence": 0.95},
        {"cause": "schema_drift", "title": "Schema drift detected", "confidence": 0.65},
    ],
    "validation_steps": [
        "Run: SELECT amount_total FROM stg_orders LIMIT 5 to confirm column exists",
        "Run dbt run --select customer_revenue after applying the fix",
        "Run dbt test --select customer_revenue to validate downstream",
    ],
})


def test_parse_valid_response():
    result = _parse_response(SAMPLE_CLAUDE_RESPONSE, SQL)
    assert isinstance(result, LLMResult)
    assert result.root_cause != ""
    assert "amount_total" in result.corrected_sql
    assert result.confidence_score == 0.95
    assert len(result.validation_steps) == 3
    assert len(result.ranked_causes) == 2


def test_parse_response_with_markdown_fences():
    wrapped = f"```json\n{SAMPLE_CLAUDE_RESPONSE}\n```"
    result = _parse_response(wrapped, SQL)
    assert result.confidence_score == 0.95


def test_parse_invalid_json_fallback():
    result = _parse_response("this is not json at all", SQL)
    assert result.root_cause == "Claude response could not be parsed"
    assert result.corrected_sql == SQL


def test_parse_response_to_dict():
    result = _parse_response(SAMPLE_CLAUDE_RESPONSE, SQL)
    d = result.to_dict()
    for key in ("root_cause", "explanation", "corrected_sql", "confidence_score",
                "validation_steps", "ranked_causes"):
        assert key in d


# ── pipeline with mocked Claude ───────────────────────────────────────────────

def test_pipeline_with_mocked_claude():
    """Full pipeline run with Claude mocked — validates end-to-end wiring."""
    mock_llm = LLMResult(
        root_cause="Column 'amount' was renamed to 'amount_total' in stg_orders",
        explanation="Schema drift: upstream renamed the column.",
        corrected_sql="select customer_id, sum(amount_total) as total_revenue from {{ ref('stg_orders') }} group by customer_id",
        confidence_score=0.95,
        validation_steps=["Run dbt run --select customer_revenue"],
        ranked_causes=[{"cause": "column_renamed_upstream", "confidence": 0.95}],
    )

    with patch("app.workflows.debug_pipeline.call_claude", return_value=mock_llm):
        result = run_pipeline(SQL, ERROR, MANIFEST, "customer_revenue", use_llm=True)

    assert result.llm_result is not None
    assert result.llm_result.confidence_score == 0.95
    assert "amount_total" in result.llm_result.corrected_sql
    assert len(result.llm_result.validation_steps) == 1
