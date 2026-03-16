from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import Optional

from app.core.config import MANIFEST_PATH
from app.workflows.debug_pipeline import run_pipeline

app = FastAPI(title="DataLineage AI", version="0.1.0")


class AnalyzeRequest(BaseModel):
    sql: str
    error_message: str
    manifest_path: Optional[str] = None
    broken_model: Optional[str] = None
    use_llm: bool = True


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/analyze")
def analyze(req: AnalyzeRequest):
    try:
        result = run_pipeline(
            sql=req.sql,
            error_text=req.error_message,
            manifest_path=req.manifest_path or MANIFEST_PATH,
            broken_model=req.broken_model,
            use_llm=req.use_llm,
        )
        return result.to_dict()
    except FileNotFoundError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
