"""
Rule Engine — Phase 5
Deterministic root-cause analysis. Runs BEFORE Claude.
Produces ranked hypotheses from structured evidence alone.
Each rule takes (parsed_sql, parsed_error, lineage, loader) and returns
zero or more RuleHit objects.
"""

from __future__ import annotations

import difflib
from dataclasses import dataclass, field

from app.services.sql_parser import ParsedSQL, parse_sql
from app.services.error_parser import ParsedError, ErrorType, parse_all_errors
from app.services.manifest_loader import ManifestLoader
from app.services.lineage_builder import LineageGraph


@dataclass
class RuleHit:
    cause: str            # short label, e.g. "column_renamed_upstream"
    title: str            # human sentence
    evidence: str         # what we found
    confidence: float     # 0.0 – 1.0
    fix_hint: str         # plain-English fix suggestion
    upstream_model: str | None = None    # where the fix lives
    suggested_column: str | None = None  # correct column name if known
    original_column: str | None = None   # the bad column that needs replacing

    def to_dict(self) -> dict:
        return {
            "cause": self.cause,
            "title": self.title,
            "evidence": self.evidence,
            "confidence": self.confidence,
            "fix_hint": self.fix_hint,
            "upstream_model": self.upstream_model,
            "suggested_column": self.suggested_column,
        }


# ── helpers ───────────────────────────────────────────────────────────────────

def _fuzzy_matches(name: str, candidates: list[str], cutoff: float = 0.6) -> list[str]:
    """Return candidates similar to name using sequence matching."""
    return difflib.get_close_matches(name, candidates, n=5, cutoff=cutoff)


def _all_upstream_columns(model: str, loader: ManifestLoader) -> dict[str, list[str]]:
    """
    For every upstream model, parse its raw SQL and collect column aliases/names.
    Returns {upstream_model_name: [col1, col2, ...]}
    """
    result: dict[str, list[str]] = {}
    for upstream_name in loader.get_all_upstream(model):
        raw_sql = loader.get_sql(upstream_name)
        if not raw_sql:
            continue
        try:
            parsed = parse_sql(raw_sql)
            # aliases are the "output" columns of a model (SELECT x AS alias)
            cols = list(parsed.aliases.keys()) + parsed.columns
            result[upstream_name] = list(dict.fromkeys(cols))  # dedupe, keep order
        except Exception:
            result[upstream_name] = []
    return result


# ── individual rules ──────────────────────────────────────────────────────────

def rule_missing_column_renamed(
    parsed_sql: ParsedSQL,
    parsed_error: ParsedError,
    loader: ManifestLoader,
    broken_model: str,
) -> list[RuleHit]:
    """
    RULE: A column used in the broken model doesn't exist in the upstream model,
    but a similar-sounding column does → likely a rename.
    """
    if parsed_error.error_type != ErrorType.MISSING_COLUMN:
        return []

    missing = parsed_error.column
    if not missing:
        return []

    hits: list[RuleHit] = []
    upstream_cols = _all_upstream_columns(broken_model, loader)

    for upstream_name, cols in upstream_cols.items():
        if missing in cols:
            continue   # column IS present — not a rename in this model
        similar = _fuzzy_matches(missing, cols)
        if similar:
            hits.append(RuleHit(
                cause="column_renamed_upstream",
                title=f"Column '{missing}' was renamed in upstream model '{upstream_name}'",
                evidence=(
                    f"'{missing}' is used in {broken_model} but not found in {upstream_name}. "
                    f"Similar columns: {similar}"
                ),
                confidence=0.95 if len(similar) == 1 else 0.80,
                fix_hint=f"Replace '{missing}' with '{similar[0]}' in {broken_model}",
                upstream_model=upstream_name,
                suggested_column=similar[0],
                original_column=missing,
            ))

    return hits


def rule_missing_column_never_existed(
    parsed_sql: ParsedSQL,
    parsed_error: ParsedError,
    loader: ManifestLoader,
    broken_model: str,
) -> list[RuleHit]:
    """
    RULE: The missing column doesn't exist anywhere in the upstream chain
    and has no fuzzy match → likely a typo or stale reference.
    """
    if parsed_error.error_type != ErrorType.MISSING_COLUMN:
        return []

    missing = parsed_error.column
    if not missing:
        return []

    upstream_cols = _all_upstream_columns(broken_model, loader)
    all_cols = [c for cols in upstream_cols.values() for c in cols]

    if _fuzzy_matches(missing, all_cols):
        return []   # covered by rename rule

    return [RuleHit(
        cause="column_never_existed",
        title=f"Column '{missing}' does not exist anywhere in the upstream lineage",
        evidence=(
            f"'{missing}' is used in {broken_model} but was not found in any upstream model. "
            f"Available columns: {all_cols[:10]}"
        ),
        confidence=0.70,
        fix_hint=(
            f"Check if '{missing}' was removed from an upstream model "
            f"or if it's a typo. Available columns: {all_cols[:5]}"
        ),
    )]


def rule_missing_relation(
    parsed_sql: ParsedSQL,
    parsed_error: ParsedError,
    loader: ManifestLoader,
    broken_model: str,
) -> list[RuleHit]:
    """
    RULE: A ref()'d model doesn't exist in the manifest.
    """
    if parsed_error.error_type != ErrorType.MISSING_RELATION:
        return []

    missing_rel = parsed_error.relation or ""
    all_models = loader.all_model_names()

    hits: list[RuleHit] = []

    # Check refs in parsed SQL
    for ref in parsed_sql.dbt_refs:
        if not loader.exists(ref):
            similar = _fuzzy_matches(ref, all_models)
            hits.append(RuleHit(
                cause="missing_ref_model",
                title=f"dbt ref('{ref}') points to a model that does not exist",
                evidence=(
                    f"ref('{ref}') used in {broken_model} but '{ref}' "
                    f"not found in manifest. Known models: {all_models}"
                ),
                confidence=0.85,
                fix_hint=(
                    f"Check the model name in ref(). "
                    + (f"Did you mean ref('{similar[0]}')?" if similar else "")
                ),
                suggested_column=similar[0] if similar else None,
            ))

    # Also check via error text if parsed_sql refs didn't catch it
    if not hits and missing_rel:
        similar = _fuzzy_matches(missing_rel, all_models)
        hits.append(RuleHit(
            cause="missing_relation",
            title=f"Relation '{missing_rel}' does not exist",
            evidence=f"Table/view '{missing_rel}' referenced but not found.",
            confidence=0.80,
            fix_hint=(
                f"Verify the table name. "
                + (f"Did you mean '{similar[0]}'?" if similar else "Check manifest.")
            ),
        ))

    return hits


def rule_ambiguous_column(
    parsed_sql: ParsedSQL,
    parsed_error: ParsedError,
    loader: ManifestLoader,
    broken_model: str,
) -> list[RuleHit]:
    """
    RULE: A column name appears in multiple joined tables without qualification.
    """
    if parsed_error.error_type != ErrorType.AMBIGUOUS_COLUMN:
        return []

    col = parsed_error.column or "unknown"
    join_tables = [j["table"] for j in parsed_sql.joins]

    return [RuleHit(
        cause="ambiguous_column",
        title=f"Column '{col}' is ambiguous across joined tables",
        evidence=(
            f"'{col}' exists in multiple joined relations: {join_tables or parsed_sql.tables}. "
            "No table qualifier specified."
        ),
        confidence=0.88,
        fix_hint=(
            f"Qualify the column with its table alias, "
            f"e.g. 'table_alias.{col}'"
        ),
    )]


def rule_type_mismatch(
    parsed_sql: ParsedSQL,
    parsed_error: ParsedError,
    loader: ManifestLoader,
    broken_model: str,
) -> list[RuleHit]:
    """
    RULE: A type cast or comparison is invalid.
    """
    if parsed_error.error_type != ErrorType.TYPE_MISMATCH:
        return []

    return [RuleHit(
        cause="type_mismatch",
        title="Type mismatch or invalid cast in query",
        evidence=parsed_error.raw_text[:200],
        confidence=0.75,
        fix_hint=(
            "Check column types in the upstream model schema. "
            "Add explicit CAST(...) or use a type-safe comparison."
        ),
    )]


def rule_schema_drift(
    parsed_sql: ParsedSQL,
    parsed_error: ParsedError,
    loader: ManifestLoader,
    broken_model: str,
) -> list[RuleHit]:
    """
    RULE: Cross-check ALL columns used in the broken model against
    what is actually available upstream. Catches schema drift broadly.
    """
    if parsed_error.error_type not in (ErrorType.MISSING_COLUMN, ErrorType.UNKNOWN):
        return []

    upstream_cols = _all_upstream_columns(broken_model, loader)
    all_available = {c for cols in upstream_cols.values() for c in cols}

    missing_cols = [
        col for col in parsed_sql.columns
        if col not in all_available
        and col not in parsed_sql.ctes           # CTE-defined columns are fine
        and col not in parsed_sql.aliases        # aliases defined in this query
    ]

    if not missing_cols:
        return []

    return [RuleHit(
        cause="schema_drift",
        title=f"Schema drift: {len(missing_cols)} column(s) used but not available upstream",
        evidence=(
            f"Columns used but not found upstream: {missing_cols}. "
            f"Available upstream: {sorted(all_available)[:15]}"
        ),
        confidence=0.65,
        fix_hint=(
            f"Columns {missing_cols} are not present in the upstream models. "
            "The upstream schema may have changed. Check recent model edits."
        ),
    )]


# ── orchestrator ──────────────────────────────────────────────────────────────

def rule_missing_group_by(
    parsed_sql: ParsedSQL,
    parsed_error: ParsedError,
    loader: ManifestLoader,
    broken_model: str,
) -> list[RuleHit]:
    """
    RULE: Aggregation functions (SUM/COUNT/etc.) used in SELECT but no GROUP BY,
    or a non-aggregated column is selected alongside aggregations without GROUP BY.
    """
    if parsed_error.error_type not in (ErrorType.MISSING_GROUP_BY, ErrorType.UNKNOWN):
        return []

    has_aggs = bool(parsed_sql.aggregations)
    has_filters = bool(parsed_sql.filters)

    # Check SQL structurally: aggregations present but no GROUP BY in raw SQL
    # sqlglot doesn't give us group_by directly, so check the aggregations + columns
    if not has_aggs:
        return []

    # Non-aggregated columns = in SELECT but not in any aggregation and not in GROUP BY
    agg_text = " ".join(parsed_sql.aggregations).lower()
    non_agg_cols = [
        c for c in parsed_sql.columns
        if c.lower() not in agg_text
        and c not in parsed_sql.aliases
        and c not in parsed_sql.group_by   # already grouped → fine
    ]

    if not non_agg_cols:
        return []

    return [RuleHit(
        cause="missing_group_by",
        title=f"Non-aggregated column(s) selected without GROUP BY",
        evidence=(
            f"Columns {non_agg_cols} are selected alongside aggregations "
            f"{parsed_sql.aggregations} but no GROUP BY is specified for them."
        ),
        confidence=0.92,
        fix_hint=(
            f"Add GROUP BY {', '.join(non_agg_cols)} at the end of the query, "
            f"or remove {non_agg_cols} from SELECT if they should be aggregated."
        ),
    )]


def rule_invalid_agg_column(
    parsed_sql: ParsedSQL,
    parsed_error: ParsedError,
    loader: ManifestLoader,
    broken_model: str,
) -> list[RuleHit]:
    """
    RULE: A column used inside an aggregation function (COUNT/SUM/etc.) does not
    exist in the upstream model. E.g. count(order) when order_id is the real column.
    """
    if parsed_error.error_type not in (
        ErrorType.INVALID_AGG_COLUMN, ErrorType.MISSING_COLUMN, ErrorType.UNKNOWN
    ):
        return []

    upstream_cols = _all_upstream_columns(broken_model, loader)
    all_available = {c for cols in upstream_cols.values() for c in cols}

    hits: list[RuleHit] = []
    import re as _re
    # Extract column names from inside aggregation calls like SUM(x), COUNT(x)
    agg_col_re = _re.compile(r'\b(?:sum|count|avg|max|min|stddev)\s*\(\s*(\w+)\s*\)', _re.IGNORECASE)

    for agg in parsed_sql.aggregations:
        m = agg_col_re.search(agg)
        if not m:
            continue
        col_in_agg = m.group(1)
        if col_in_agg == "*" or col_in_agg.lower() in ("1",):
            continue
        if col_in_agg in all_available:
            continue
        similar = _fuzzy_matches(col_in_agg, list(all_available))
        hits.append(RuleHit(
            cause="invalid_column_in_aggregation",
            title=f"Column '{col_in_agg}' inside {agg} does not exist upstream",
            evidence=(
                f"'{col_in_agg}' used in aggregation '{agg}' but not found in "
                f"upstream models. "
                + (f"Similar column: {similar}" if similar else f"Available: {sorted(all_available)[:8]}")
            ),
            confidence=0.91,
            fix_hint=(
                f"Replace '{col_in_agg}' with '{similar[0]}' inside {agg}."
                if similar else
                f"Check upstream model columns. Available: {sorted(all_available)[:5]}"
            ),
            suggested_column=similar[0] if similar else None,
            original_column=col_in_agg,
        ))

    return hits


ALL_RULES = [
    rule_missing_column_renamed,
    rule_missing_column_never_existed,
    rule_missing_relation,
    rule_ambiguous_column,
    rule_type_mismatch,
    rule_schema_drift,
    rule_missing_group_by,
    rule_invalid_agg_column,
]


def check_query_validity(
    parsed_sql: ParsedSQL,
    parsed_error: ParsedError,
    loader: ManifestLoader,
    broken_model: str,
) -> tuple[bool, str]:
    """
    Cross-check the SQL against ALL errors in the error message.
    Returns (is_valid, reason).

    is_valid = True only when:
      1. All columns used in SQL exist upstream, AND
      2. No structural issues (missing GROUP BY, invalid agg columns)
    """
    import re as _re
    upstream_cols = _all_upstream_columns(broken_model, loader)
    all_available = {c for cols in upstream_cols.values() for c in cols}
    internal = set(parsed_sql.ctes) | set(parsed_sql.aliases.keys())

    # ── Check 1: all plain columns exist upstream ────────────────────────────
    missing_cols = [
        col for col in parsed_sql.columns
        if col not in all_available and col not in internal
    ]

    # ── Check 2: all columns inside aggregations exist upstream ──────────────
    agg_col_re = _re.compile(r'\b(?:sum|count|avg|max|min)\s*\(\s*(\w+)\s*\)', _re.IGNORECASE)
    bad_agg_cols = []
    for agg in parsed_sql.aggregations:
        m = agg_col_re.search(agg)
        if m:
            c = m.group(1)
            if c != "*" and c not in all_available and c not in internal:
                bad_agg_cols.append(f"{c} (in {agg})")

    # ── Check 3: GROUP BY present when needed ────────────────────────────────
    # Check if error message mentions GROUP BY issue
    all_parsed = parse_all_errors(parsed_error.raw_text)
    has_group_by_error = any(
        e.error_type == ErrorType.MISSING_GROUP_BY for e in all_parsed
    )
    # Non-aggregated columns = in SELECT but not inside an agg and not in GROUP BY
    agg_text = " ".join(parsed_sql.aggregations).lower()
    non_agg_cols = [
        c for c in parsed_sql.columns
        if c.lower() not in agg_text
        and c not in internal
        and c not in parsed_sql.group_by   # present in GROUP BY → valid
    ]
    # Missing GROUP BY only if:
    #   - error message mentions it, AND
    #   - SQL has aggregations, AND
    #   - there are non-aggregated cols NOT covered by GROUP BY
    missing_group_by = (
        has_group_by_error
        and bool(parsed_sql.aggregations)
        and bool(non_agg_cols)
    )

    problems = []
    if missing_cols:
        problems.append(f"Missing columns: {missing_cols}")
    if bad_agg_cols:
        problems.append(f"Invalid aggregation columns: {bad_agg_cols}")
    if missing_group_by:
        problems.append(f"GROUP BY required for: {non_agg_cols}")

    if not problems:
        return True, (
            f"All columns exist upstream and SQL structure is valid. "
            f"Available upstream: {sorted(all_available)}"
        )

    return False, " | ".join(problems)


def run_rules(
    parsed_sql: ParsedSQL,
    parsed_error: ParsedError,
    loader: ManifestLoader,
    broken_model: str,
) -> list[RuleHit]:
    """
    Run all rules against every sub-error in the error message.
    Returns hits sorted by confidence descending, deduplicated by cause.
    """
    # Expand the error message into individual issues
    all_errors = parse_all_errors(parsed_error.raw_text)

    hits: list[RuleHit] = []
    seen_causes: set[str] = set()

    for err in all_errors:
        for rule_fn in ALL_RULES:
            for hit in rule_fn(parsed_sql, err, loader, broken_model):
                if hit.cause not in seen_causes:
                    hits.append(hit)
                    seen_causes.add(hit.cause)

    return sorted(hits, key=lambda h: h.confidence, reverse=True)


def generate_corrected_sql(sql: str, rule_hits: list[RuleHit], parsed_sql: "ParsedSQL") -> str | None:
    """
    Apply rule hits to the SQL and return a corrected version.
    Returns None if no automatic fix can be determined.
    Handles:
      - column renames (column_renamed_upstream)
      - invalid aggregation columns (invalid_column_in_aggregation)
      - missing GROUP BY (missing_group_by)
    """
    import re as _re
    corrected = sql
    changed = False

    for hit in rule_hits:
        if hit.original_column and hit.suggested_column:
            # Replace exact word boundary matches to avoid partial replacements
            pattern = r'\b' + _re.escape(hit.original_column) + r'\b'
            new_sql = _re.sub(pattern, hit.suggested_column, corrected)
            if new_sql != corrected:
                corrected = new_sql
                changed = True

        elif hit.cause == "missing_group_by":
            # Find non-aggregated columns to add to GROUP BY
            agg_text = " ".join(parsed_sql.aggregations).lower()
            internal = set(parsed_sql.ctes) | set(parsed_sql.aliases.keys())
            group_cols = [
                c for c in parsed_sql.columns
                if c.lower() not in agg_text and c not in internal
                and c not in parsed_sql.group_by
            ]
            if group_cols and "group by" not in corrected.lower():
                corrected = corrected.rstrip().rstrip(";")
                corrected += f"\ngroup by {', '.join(group_cols)}"
                changed = True

    return corrected if changed else None
