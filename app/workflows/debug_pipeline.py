"""
Debug Pipeline — Phase 6
Single entry point that runs all modules in order:
  parse SQL → load manifest → build lineage → parse error
  → run rules → call Claude → return unified result
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

from app.services.sql_parser import ParsedSQL, parse_sql
from app.services.error_parser import ParsedError, parse_error, parse_all_errors
from app.services.manifest_loader import ManifestLoader
from app.services.lineage_builder import LineageGraph
from app.services.rule_engine import RuleHit, run_rules, check_query_validity, generate_corrected_sql
from app.services.llm_service import LLMResult, call_claude


@dataclass
class PipelineResult:
    broken_model: str
    parsed_sql: ParsedSQL
    parsed_error: ParsedError
    lineage: dict
    lineage_ascii: str
    rule_hits: list[RuleHit]
    upstream_columns: dict[str, list[str]]
    llm_result: LLMResult | None = None
    error: str | None = None           # pipeline-level error message if something failed
    query_is_valid: bool = False       # True = SQL is clean; error msg is stale
    validity_reason: str = ""          # explanation of why valid/invalid
    all_errors: list = field(default_factory=list)      # all sub-errors parsed from message
    corrected_sql_from_rules: str | None = None         # auto-fix from rule engine (no LLM needed)

    def to_dict(self) -> dict:
        return {
            "broken_model": self.broken_model,
            "parsed_sql": self.parsed_sql.to_dict(),
            "parsed_error": self.parsed_error.to_dict(),
            "lineage": self.lineage,
            "lineage_ascii": self.lineage_ascii,
            "rule_hits": [h.to_dict() for h in self.rule_hits],
            "upstream_columns": self.upstream_columns,
            "llm_result": self.llm_result.to_dict() if self.llm_result else None,
            "error": self.error,
            "query_is_valid": self.query_is_valid,
            "validity_reason": self.validity_reason,
        }


def _collect_upstream_columns(
    broken_model: str,
    loader: ManifestLoader,
) -> dict[str, list[str]]:
    """
    For each upstream model, parse its raw SQL and collect output column names
    (aliases take priority — they are what downstream models actually see).
    """
    result: dict[str, list[str]] = {}
    for name in loader.get_all_upstream(broken_model):
        raw_sql = loader.get_sql(name)
        if not raw_sql:
            continue
        try:
            p = parse_sql(raw_sql)
            # Aliases are the "published" columns; fall back to plain columns
            cols = list(p.aliases.keys()) if p.aliases else p.columns
            result[name] = list(dict.fromkeys(cols))
        except Exception:
            result[name] = []
    return result


def run_pipeline(
    sql: str,
    error_text: str,
    manifest_path: str | Path,
    broken_model: str | None = None,
    use_llm: bool = True,
) -> PipelineResult:
    """
    Run the full debug pipeline.

    Args:
        sql           : broken dbt model SQL (may contain {{ ref(...) }})
        error_text    : raw dbt / warehouse error message
        manifest_path : path to dbt's target/manifest.json
        broken_model  : model name to analyse (auto-detected from error if None)
        use_llm       : set False to skip Claude (useful for offline tests)

    Returns:
        PipelineResult with all intermediate and final outputs.
    """
    # ── Step 1: parse SQL ─────────────────────────────────────────────────────
    parsed_sql = parse_sql(sql)

    # ── Step 2: parse error (primary + all sub-errors) ───────────────────────
    parsed_error = parse_error(error_text)
    all_errors   = parse_all_errors(error_text)

    # ── Step 3: resolve broken model name ─────────────────────────────────────
    # Load manifest early so we can use it for model detection
    loader_early = ManifestLoader(manifest_path)

    if not broken_model:
        broken_model = parsed_error.model

    if not broken_model:
        # The broken model is the one that CONTAINS the SQL (has these refs as deps).
        # It is DOWNSTREAM of the ref'd models, not the refs themselves.
        for ref in parsed_sql.dbt_refs:
            downstream = loader_early.get_downstream(ref)
            if downstream:
                broken_model = downstream[0]
                break

    if not broken_model:
        # Last resort: use the first ref (at least something)
        broken_model = parsed_sql.dbt_refs[0] if parsed_sql.dbt_refs else "unknown"

    # ── Step 4: load manifest + build lineage ─────────────────────────────────
    loader = loader_early          # already loaded above
    graph  = LineageGraph(loader)

    lineage       = graph.to_dict(broken_model)
    lineage_ascii = graph.ascii()

    # ── Step 5: collect upstream column info ──────────────────────────────────
    upstream_columns = _collect_upstream_columns(broken_model, loader)

    # ── Step 6: check if SQL is already valid (error may be stale) ───────────
    query_is_valid, validity_reason = check_query_validity(
        parsed_sql, parsed_error, loader, broken_model
    )

    # ── Step 6b: run deterministic rule engine ────────────────────────────────
    rule_hits = run_rules(parsed_sql, parsed_error, loader, broken_model)
    corrected_sql_from_rules = generate_corrected_sql(sql, rule_hits, parsed_sql) if rule_hits else None

    # ── Step 7: Claude reasoning (skip if query is already valid) ─────────────
    llm_result = None
    llm_error  = None

    if use_llm and not query_is_valid:
        try:
            llm_result = call_claude(
                broken_model=broken_model,
                sql=sql,
                parsed_error=parsed_error,
                lineage=lineage,
                upstream_columns=upstream_columns,
                rule_hits=rule_hits,
            )
        except ValueError as e:
            # Missing API key — surface as a friendly message, don't crash
            llm_error = str(e)
        except Exception as e:
            llm_error = f"Claude API error: {e}"

    return PipelineResult(
        broken_model=broken_model,
        parsed_sql=parsed_sql,
        parsed_error=parsed_error,
        lineage=lineage,
        lineage_ascii=lineage_ascii,
        rule_hits=rule_hits,
        upstream_columns=upstream_columns,
        llm_result=llm_result,
        error=llm_error,
        query_is_valid=query_is_valid,
        validity_reason=validity_reason,
        all_errors=all_errors,
        corrected_sql_from_rules=corrected_sql_from_rules,
    )
