"""
LLM Service — Phase 6
Sends structured evidence to Claude and parses a structured response.
Claude's job: rank hypotheses, explain the root cause, write corrected SQL,
and produce a validation checklist. All parsing is done with code; Claude
never returns free-form text that we have to guess the structure of.
"""

import json
import re
from dataclasses import dataclass, field

import anthropic

from app.core.config import ANTHROPIC_API_KEY
from app.services.sql_parser import ParsedSQL
from app.services.error_parser import ParsedError
from app.services.rule_engine import RuleHit


# ── response schema ───────────────────────────────────────────────────────────

@dataclass
class LLMResult:
    root_cause: str                       # one-line summary
    explanation: str                      # plain-English paragraph
    corrected_sql: str                    # fixed SQL (or original if no fix needed)
    confidence_score: float               # 0.0 – 1.0
    validation_steps: list[str] = field(default_factory=list)
    ranked_causes: list[dict] = field(default_factory=list)
    raw_response: str = ""

    def to_dict(self) -> dict:
        return {
            "root_cause": self.root_cause,
            "explanation": self.explanation,
            "corrected_sql": self.corrected_sql,
            "confidence_score": self.confidence_score,
            "validation_steps": self.validation_steps,
            "ranked_causes": self.ranked_causes,
        }


# ── prompts ───────────────────────────────────────────────────────────────────

SYSTEM_PROMPT = """\
You are a senior data engineer specialising in dbt and SQL pipeline debugging.

You receive structured evidence about a broken dbt model:
- the broken SQL
- the error message
- the upstream lineage chain
- columns available in each upstream model
- candidate root causes already identified by a rule engine

Your job is to:
1. Rank the candidate causes from most to least likely
2. Explain the most likely root cause clearly in plain English
3. Write a corrected version of the SQL
4. List concrete validation steps the engineer should run

You MUST respond with valid JSON only — no markdown fences, no prose outside the JSON.
Use exactly this schema:

{
  "root_cause": "<one sentence summary>",
  "explanation": "<2-4 sentence plain-English explanation>",
  "corrected_sql": "<full corrected SQL>",
  "confidence_score": <float 0.0-1.0>,
  "ranked_causes": [
    {"cause": "<label>", "title": "<sentence>", "confidence": <float>}
  ],
  "validation_steps": [
    "<step 1>",
    "<step 2>"
  ]
}
"""


def _build_user_prompt(
    broken_model: str,
    sql: str,
    error: ParsedError,
    lineage: dict,
    upstream_columns: dict[str, list[str]],
    rule_hits: list[RuleHit],
) -> str:
    candidates_block = "\n".join(
        f"  - [{h.confidence:.0%}] {h.title}\n    Evidence: {h.evidence}\n    Fix hint: {h.fix_hint}"
        for h in rule_hits
    ) or "  None detected by rule engine."

    upstream_block = "\n".join(
        f"  {model}: {cols}"
        for model, cols in upstream_columns.items()
    ) or "  No upstream column data available."

    lineage_path = " → ".join(
        lineage.get("paths_to_root", [[broken_model]])[0]
    ) if lineage.get("paths_to_root") else broken_model

    return f"""\
## Broken model
Name: {broken_model}

## SQL
```sql
{sql.strip()}
```

## Error
Type   : {error.error_type.value}
Message: {error.raw_text.strip()}
Column : {error.column or "N/A"}
Line   : {error.line_number or "N/A"}
Candidates from warehouse: {error.candidates or []}

## Lineage chain
{lineage_path}

## Columns available in upstream models
{upstream_block}

## Candidate root causes (from rule engine, sorted by confidence)
{candidates_block}

Respond with JSON only.
"""


# ── main call ─────────────────────────────────────────────────────────────────

def call_claude(
    broken_model: str,
    sql: str,
    parsed_error: ParsedError,
    lineage: dict,
    upstream_columns: dict[str, list[str]],
    rule_hits: list[RuleHit],
    model: str = "claude-sonnet-4-6",
) -> LLMResult:
    """
    Send structured evidence to Claude and return a parsed LLMResult.
    Raises ValueError if the API key is missing or Claude returns unparseable JSON.
    """
    if not ANTHROPIC_API_KEY or ANTHROPIC_API_KEY == "your_api_key_here":
        raise ValueError(
            "ANTHROPIC_API_KEY is not set. "
            "Add your key to .env: ANTHROPIC_API_KEY=sk-ant-..."
        )

    client = anthropic.Anthropic(api_key=ANTHROPIC_API_KEY)

    user_prompt = _build_user_prompt(
        broken_model=broken_model,
        sql=sql,
        error=parsed_error,
        lineage=lineage,
        upstream_columns=upstream_columns,
        rule_hits=rule_hits,
    )

    message = client.messages.create(
        model=model,
        max_tokens=2048,
        system=SYSTEM_PROMPT,
        messages=[{"role": "user", "content": user_prompt}],
    )

    raw = message.content[0].text
    return _parse_response(raw, sql)


def _parse_response(raw: str, original_sql: str) -> LLMResult:
    """Parse Claude's JSON response into an LLMResult, with graceful fallback."""
    # Strip accidental markdown fences if Claude adds them
    clean = re.sub(r"^```(?:json)?\s*", "", raw.strip(), flags=re.MULTILINE)
    clean = re.sub(r"\s*```$", "", clean.strip(), flags=re.MULTILINE)

    try:
        data = json.loads(clean)
    except json.JSONDecodeError as e:
        # Return partial result rather than crashing the whole pipeline
        return LLMResult(
            root_cause="Claude response could not be parsed",
            explanation=raw[:500],
            corrected_sql=original_sql,
            confidence_score=0.0,
            raw_response=raw,
        )

    return LLMResult(
        root_cause=data.get("root_cause", ""),
        explanation=data.get("explanation", ""),
        corrected_sql=data.get("corrected_sql", original_sql),
        confidence_score=float(data.get("confidence_score", 0.0)),
        validation_steps=data.get("validation_steps", []),
        ranked_causes=data.get("ranked_causes", []),
        raw_response=raw,
    )
