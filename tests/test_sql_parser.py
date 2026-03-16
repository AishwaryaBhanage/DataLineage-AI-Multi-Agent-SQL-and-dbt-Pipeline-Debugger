"""
Tests for sql_parser.py — run with: python -m pytest tests/test_sql_parser.py -v
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app.services.sql_parser import parse_sql


# ── Test 1: our actual broken dbt model ──────────────────────────────────────

def test_broken_customer_revenue():
    sql = """
    select
        customer_id,
        sum(amount) as total_revenue,
        count(order_id) as order_count,
        max(order_date) as last_order_date
    from {{ ref('stg_orders') }}
    group by customer_id
    """
    result = parse_sql(sql)

    assert "stg_orders" in result.tables
    assert "stg_orders" in result.dbt_refs
    assert "customer_id" in result.columns
    assert "amount" in result.columns           # the missing column we need to catch
    assert any("sum" in a.lower() for a in result.aggregations)
    assert "total_revenue" in result.aliases


# ── Test 2: dbt ref extraction ────────────────────────────────────────────────

def test_dbt_ref_extraction():
    sql = """
    select o.order_id, c.name
    from {{ ref('orders') }} o
    join {{ ref('customers') }} c on o.customer_id = c.id
    """
    result = parse_sql(sql)

    assert "orders" in result.dbt_refs
    assert "customers" in result.dbt_refs
    assert len(result.joins) == 1


# ── Test 3: CTE detection ─────────────────────────────────────────────────────

def test_cte_detection():
    sql = """
    with base as (
        select order_id, amount from orders
    ),
    filtered as (
        select * from base where amount > 100
    )
    select * from filtered
    """
    result = parse_sql(sql)

    assert "base" in result.ctes
    assert "filtered" in result.ctes


# ── Test 4: filters (WHERE clause) ───────────────────────────────────────────

def test_filter_extraction():
    sql = """
    select customer_id, total_amount
    from stg_orders
    where order_date >= '2024-01-01'
      and status = 'completed'
    """
    result = parse_sql(sql)

    assert len(result.filters) > 0
    combined = " ".join(result.filters)
    assert "order_date" in combined


# ── Test 5: multiple aggregations ────────────────────────────────────────────

def test_multiple_aggregations():
    sql = """
    select
        region,
        sum(revenue)  as total_revenue,
        avg(revenue)  as avg_revenue,
        count(*)      as row_count,
        max(sale_date) as latest_sale
    from sales
    group by region
    """
    result = parse_sql(sql)

    agg_text = " ".join(result.aggregations).lower()
    assert "sum" in agg_text
    assert "avg" in agg_text
    assert "count" in agg_text
    assert "max" in agg_text


# ── Test 6: column alias mapping ─────────────────────────────────────────────

def test_alias_mapping():
    sql = """
    select
        customer_id,
        amount_total as revenue
    from stg_orders
    """
    result = parse_sql(sql)

    assert "revenue" in result.aliases
    assert "amount_total" in result.aliases["revenue"]


# ── Test 7: to_dict output ────────────────────────────────────────────────────

def test_to_dict():
    sql = "select id, name from customers"
    result = parse_sql(sql).to_dict()

    assert isinstance(result, dict)
    assert "tables" in result
    assert "columns" in result
    assert "customers" in result["tables"]
