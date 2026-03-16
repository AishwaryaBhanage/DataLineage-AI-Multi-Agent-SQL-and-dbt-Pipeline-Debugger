"""
SQL Parser — Phase 3
Uses sqlglot to extract structured information from any SQL query.
Handles raw SQL and dbt-style SQL ({{ ref(...) }} jinja refs).
"""

import re
import sqlglot
from sqlglot import exp
from dataclasses import dataclass, field


@dataclass
class ParsedSQL:
    tables: list[str] = field(default_factory=list)
    columns: list[str] = field(default_factory=list)
    joins: list[dict] = field(default_factory=list)
    filters: list[str] = field(default_factory=list)
    ctes: list[str] = field(default_factory=list)
    aggregations: list[str] = field(default_factory=list)
    dbt_refs: list[str] = field(default_factory=list)   # models referenced via {{ ref(...) }}
    aliases: dict[str, str] = field(default_factory=dict)  # alias -> real name
    group_by: list[str] = field(default_factory=list)    # columns in GROUP BY

    def to_dict(self) -> dict:
        return {
            "tables": self.tables,
            "columns": self.columns,
            "joins": self.joins,
            "filters": self.filters,
            "ctes": self.ctes,
            "aggregations": self.aggregations,
            "dbt_refs": self.dbt_refs,
            "group_by": self.group_by,
            "aliases": self.aliases,
        }


def _strip_dbt_jinja(sql: str) -> tuple[str, list[str]]:
    """
    Replace dbt {{ ref('model') }} and {{ source('schema', 'table') }} with
    plain table names so sqlglot can parse them cleanly.
    Returns (cleaned_sql, list_of_ref_names).
    """
    refs: list[str] = []

    # Match {{ ref('model_name') }} or {{ ref("model_name") }}
    def replace_ref(match):
        model_name = match.group(1).strip("'\"")
        refs.append(model_name)
        return model_name   # replace jinja with plain table name

    sql = re.sub(r"\{\{\s*ref\(\s*['\"](\w+)['\"]\s*\)\s*\}\}", replace_ref, sql)

    # Match {{ source('schema', 'table') }}
    def replace_source(match):
        table_name = match.group(2).strip("'\"")
        refs.append(table_name)
        return table_name

    sql = re.sub(
        r"\{\{\s*source\(\s*['\"](\w+)['\"]\s*,\s*['\"](\w+)['\"]\s*\)\s*\}\}",
        replace_source,
        sql,
    )

    # Strip any remaining jinja blocks (config, macros, etc.)
    sql = re.sub(r"\{\{.*?\}\}", "", sql, flags=re.DOTALL)
    sql = re.sub(r"\{%.*?%\}", "", sql, flags=re.DOTALL)

    return sql.strip(), refs


def _extract_tables(ast: exp.Expression) -> list[str]:
    tables = []
    for table in ast.find_all(exp.Table):
        name = table.name
        if name and name not in tables:
            tables.append(name)
    return tables


def _extract_columns(ast: exp.Expression) -> list[str]:
    columns = []
    for col in ast.find_all(exp.Column):
        name = col.name
        if name and name not in columns:
            columns.append(name)
    return columns


def _extract_joins(ast: exp.Expression) -> list[dict]:
    joins = []
    for join in ast.find_all(exp.Join):
        join_info = {
            "table": join.this.name if hasattr(join.this, "name") else str(join.this),
            "type": join.args.get("kind", "INNER"),
            "on": str(join.args.get("on", "")) or None,
        }
        joins.append(join_info)
    return joins


def _extract_filters(ast: exp.Expression) -> list[str]:
    filters = []
    for where in ast.find_all(exp.Where):
        filters.append(str(where.this))
    return filters


def _extract_ctes(ast: exp.Expression) -> list[str]:
    ctes = []
    for with_ in ast.find_all(exp.With):
        for cte in with_.expressions:
            ctes.append(cte.alias)
    return ctes


def _extract_group_by(ast: exp.Expression) -> list[str]:
    cols = []
    for group in ast.find_all(exp.Group):
        for expr in group.expressions:
            name = expr.name if hasattr(expr, "name") else str(expr)
            if name and name not in cols:
                cols.append(name)
    return cols


def _extract_aggregations(ast: exp.Expression) -> list[str]:
    agg_types = (exp.Sum, exp.Avg, exp.Count, exp.Max, exp.Min, exp.Stddev, exp.Variance)
    seen = []
    for node in ast.find_all(*agg_types):
        text = str(node)
        if text not in seen:
            seen.append(text)
    return seen


def _extract_aliases(ast: exp.Expression) -> dict[str, str]:
    aliases = {}
    for alias_node in ast.find_all(exp.Alias):
        alias_name = alias_node.alias
        real = str(alias_node.this)
        if alias_name:
            aliases[alias_name] = real
    return aliases


def parse_sql(sql: str) -> ParsedSQL:
    """
    Main entry point.
    Accepts raw SQL or dbt-style SQL with {{ ref(...) }} syntax.
    Returns a ParsedSQL dataclass with all extracted entities.
    """
    cleaned_sql, dbt_refs = _strip_dbt_jinja(sql)

    try:
        ast = sqlglot.parse_one(cleaned_sql, dialect="duckdb")
    except Exception:
        # Fallback: try without dialect
        try:
            ast = sqlglot.parse_one(cleaned_sql)
        except Exception as e:
            raise ValueError(f"Failed to parse SQL: {e}\n\nSQL was:\n{cleaned_sql}")

    return ParsedSQL(
        tables=_extract_tables(ast),
        columns=_extract_columns(ast),
        joins=_extract_joins(ast),
        filters=_extract_filters(ast),
        ctes=_extract_ctes(ast),
        aggregations=_extract_aggregations(ast),
        dbt_refs=dbt_refs,
        aliases=_extract_aliases(ast),
        group_by=_extract_group_by(ast),
    )
