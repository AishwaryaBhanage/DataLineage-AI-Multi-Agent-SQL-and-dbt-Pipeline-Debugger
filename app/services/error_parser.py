"""
Error Parser — Phase 5
Normalises raw dbt/warehouse error text into a structured object.
Supports the 7 most common dbt/SQL failure patterns.
"""

import re
from dataclasses import dataclass, field
from enum import Enum


class ErrorType(str, Enum):
    MISSING_COLUMN      = "missing_column"
    MISSING_RELATION    = "missing_relation"
    AMBIGUOUS_COLUMN    = "ambiguous_column"
    TYPE_MISMATCH       = "type_mismatch"
    BROKEN_REF          = "broken_ref"
    NULL_VIOLATION      = "null_violation"
    SYNTAX_ERROR        = "syntax_error"
    MISSING_GROUP_BY    = "missing_group_by"    # aggregation without GROUP BY
    INVALID_AGG_COLUMN  = "invalid_agg_column"  # bad column inside COUNT/SUM/etc.
    UNKNOWN             = "unknown"


@dataclass
class ParsedError:
    raw_text: str
    error_type: ErrorType
    # extracted entities (populated depending on error_type)
    column: str | None = None        # missing/ambiguous column name
    relation: str | None = None      # missing table/view name
    model: str | None = None         # dbt model mentioned in error
    line_number: int | None = None   # SQL line number if present
    hint: str | None = None          # warehouse-provided hint / candidates
    candidates: list[str] = field(default_factory=list)  # suggested alternatives

    def to_dict(self) -> dict:
        return {
            "raw_text": self.raw_text,
            "error_type": self.error_type.value,
            "column": self.column,
            "relation": self.relation,
            "model": self.model,
            "line_number": self.line_number,
            "hint": self.hint,
            "candidates": self.candidates,
        }


# ── patterns (order matters — more specific first) ────────────────────────────

_PATTERNS: list[tuple[ErrorType, list[re.Pattern]]] = [

    (ErrorType.MISSING_COLUMN, [
        # DuckDB: Referenced column "amount" not found in FROM clause!
        # Candidate bindings: "amount_total", "status"
        re.compile(
            r'referenced column ["\']?(?P<col>[\w.]+)["\']? not found',
            re.IGNORECASE
        ),
        # Postgres: column "amount" does not exist
        re.compile(
            r'column ["\']?(?P<col>[\w.]+)["\']? does not exist',
            re.IGNORECASE
        ),
        # Snowflake: invalid identifier 'AMOUNT'
        re.compile(
            r"invalid identifier ['\"]?(?P<col>[\w.]+)['\"]?",
            re.IGNORECASE
        ),
        # BigQuery: Unrecognized name: amount
        re.compile(
            r'unrecognized name:\s*(?P<col>\w+)',
            re.IGNORECASE
        ),
    ]),

    (ErrorType.MISSING_RELATION, [
        # Postgres/DuckDB: relation "stg_orders" does not exist
        re.compile(
            r'relation ["\']?(?P<rel>[\w.]+)["\']? does not exist',
            re.IGNORECASE
        ),
        # Snowflake: Object 'STG_ORDERS' does not exist
        re.compile(
            r"object ['\"]?(?P<rel>[\w.]+)['\"]? does not exist",
            re.IGNORECASE
        ),
        # BigQuery: Table not found: project.dataset.stg_orders
        re.compile(
            r'table not found[:\s]+(?P<rel>[\w.]+)',
            re.IGNORECASE
        ),
        # dbt: Compilation Error — depends on a node named 'X' which was not found
        re.compile(
            r"depends on a node named ['\"]?(?P<rel>\w+)['\"]? which was not found",
            re.IGNORECASE
        ),
    ]),

    (ErrorType.AMBIGUOUS_COLUMN, [
        # Postgres: column reference "id" is ambiguous
        re.compile(
            r'column (?:reference )?["\']?(?P<col>[\w.]+)["\']? is ambiguous',
            re.IGNORECASE
        ),
        # DuckDB: Ambiguous reference to column name "id"
        re.compile(
            r'ambiguous reference to column (?:name )?["\']?(?P<col>\w+)["\']?',
            re.IGNORECASE
        ),
    ]),

    (ErrorType.TYPE_MISMATCH, [
        # generic: cannot cast ... to integer / operator does not exist: text = integer
        re.compile(r'cannot cast', re.IGNORECASE),
        re.compile(r'operator does not exist', re.IGNORECASE),
        re.compile(r'type mismatch', re.IGNORECASE),
        re.compile(r'conversion failed', re.IGNORECASE),
        re.compile(r'incompatible types', re.IGNORECASE),
    ]),

    (ErrorType.BROKEN_REF, [
        # dbt Compilation Error with ref
        re.compile(r'compilation error', re.IGNORECASE),
        re.compile(r'error in ref', re.IGNORECASE),
    ]),

    (ErrorType.NULL_VIOLATION, [
        re.compile(r'null value in column', re.IGNORECASE),
        re.compile(r'not-null constraint', re.IGNORECASE),
        re.compile(r'violates not-null', re.IGNORECASE),
    ]),

    (ErrorType.SYNTAX_ERROR, [
        re.compile(r'syntax error', re.IGNORECASE),
        re.compile(r'parse error', re.IGNORECASE),
        re.compile(r'unexpected token', re.IGNORECASE),
    ]),

    (ErrorType.MISSING_GROUP_BY, [
        re.compile(r'group\s+by\s+clause', re.IGNORECASE),
        re.compile(r'selected without a.*group by', re.IGNORECASE),
        re.compile(r'non.aggregated column', re.IGNORECASE),
        re.compile(r'must appear in the GROUP BY', re.IGNORECASE),
        re.compile(r'not in group by', re.IGNORECASE),
    ]),

    (ErrorType.INVALID_AGG_COLUMN, [
        # e.g. count(order) is invalid because 'order' is not a valid column
        re.compile(r'is invalid because [`\']?(?P<col>\w+)[`\']? is not a valid column', re.IGNORECASE),
        re.compile(r'invalid aggregation', re.IGNORECASE),
    ]),
]

# Extract candidate column names from DuckDB's "Candidate bindings: ..." hint
_CANDIDATES_RE = re.compile(
    r'candidate bindings[:\s]+"?(.+)"?',
    re.IGNORECASE
)

# Extract line number from "LINE 15:" style messages
_LINE_RE = re.compile(r'\bline\s+(\d+)', re.IGNORECASE)

# Extract dbt model name from "in model X" or "(models/X.sql)"
_MODEL_RE = re.compile(
    r'(?:in model\s+(\w+)|models/(\w+)\.sql)',
    re.IGNORECASE
)


def _extract_column(text: str, patterns: list[re.Pattern]) -> str | None:
    for pat in patterns:
        m = pat.search(text)
        if m and "col" in m.groupdict():
            return m.group("col")
    return None


def _extract_relation(text: str, patterns: list[re.Pattern]) -> str | None:
    for pat in patterns:
        m = pat.search(text)
        if m and "rel" in m.groupdict():
            return m.group("rel")
    return None


def _extract_candidates(text: str) -> list[str]:
    m = _CANDIDATES_RE.search(text)
    if not m:
        return []
    raw = m.group(1)
    # split on comma or quote boundaries
    return [c.strip().strip('"\'') for c in re.split(r'[,\s]+', raw) if c.strip().strip('"\'')]


def parse_error(error_text: str) -> ParsedError:
    """
    Main entry point.
    Accepts raw dbt or warehouse error text.
    Returns a structured ParsedError.
    """
    text = error_text.strip()

    # Detect model name
    model_match = _MODEL_RE.search(text)
    model = next((g for g in (model_match.group(1), model_match.group(2)) if g), None) if model_match else None

    # Detect line number
    line_match = _LINE_RE.search(text)
    line_number = int(line_match.group(1)) if line_match else None

    # Detect candidates hint
    candidates = _extract_candidates(text)
    hint = f"Candidate bindings: {', '.join(candidates)}" if candidates else None

    # Match error type
    for error_type, patterns in _PATTERNS:
        for pat in patterns:
            if pat.search(text):
                # type-specific entity extraction
                column = None
                relation = None

                if error_type == ErrorType.MISSING_COLUMN:
                    column = _extract_column(text, patterns)
                elif error_type == ErrorType.MISSING_RELATION:
                    relation = _extract_relation(text, patterns)
                elif error_type == ErrorType.AMBIGUOUS_COLUMN:
                    column = _extract_column(text, patterns)

                return ParsedError(
                    raw_text=text,
                    error_type=error_type,
                    column=column,
                    relation=relation,
                    model=model,
                    line_number=line_number,
                    hint=hint,
                    candidates=candidates,
                )

    return ParsedError(raw_text=text, error_type=ErrorType.UNKNOWN, model=model)


def parse_all_errors(error_text: str) -> list[ParsedError]:
    """
    Split a multi-error message into individual errors and parse each one.

    Handles formats like:
      - bullet points (- error one\n- error two)
      - numbered lists (1. error  2. error)
      - newline-separated sentences
      - single error (falls back to parse_error)

    Returns a deduplicated list of ParsedError objects — one per distinct issue.
    Always returns at least one entry (the full-text parse).
    """
    text = error_text.strip()

    # Split on: newline + bullet/dash/number prefix, or double-newline
    # e.g. "- foo\n- bar" or "1. foo\n2. bar"
    segments = re.split(r'\n\s*[-•*]\s*|\n\s*\d+\.\s*|\n{2,}', text)
    segments = [s.strip() for s in segments if s.strip()]

    # Always parse the full text first
    primary = parse_error(text)
    results: list[ParsedError] = [primary]
    seen_types: set[tuple] = {(primary.error_type, primary.column, primary.relation)}

    for seg in segments:
        if len(seg) < 10:
            continue
        p = parse_error(seg)
        key = (p.error_type, p.column, p.relation)
        if key not in seen_types and p.error_type != ErrorType.UNKNOWN:
            # Carry over model from primary if not detected in segment
            if not p.model:
                p.model = primary.model
            results.append(p)
            seen_types.add(key)

    return results
