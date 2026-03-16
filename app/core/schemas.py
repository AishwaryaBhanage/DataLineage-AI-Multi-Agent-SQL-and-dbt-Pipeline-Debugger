from pydantic import BaseModel
from typing import Optional


class DebugRequest(BaseModel):
    sql: str
    error_message: str
    manifest_path: Optional[str] = None


class ParsedSQL(BaseModel):
    tables: list[str]
    columns: list[str]
    joins: list[str]
    filters: list[str]
    ctes: list[str]
    aggregations: list[str]


class RootCauseHypothesis(BaseModel):
    cause: str
    evidence: str
    confidence: float


class DebugResult(BaseModel):
    parsed_sql: ParsedSQL
    lineage: dict
    candidate_causes: list[RootCauseHypothesis]
    root_cause: Optional[str] = None
    corrected_sql: Optional[str] = None
    explanation: Optional[str] = None
    validation_steps: list[str] = []
