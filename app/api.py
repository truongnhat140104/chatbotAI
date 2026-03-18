from __future__ import annotations

import traceback
from contextlib import asynccontextmanager
from typing import Any

from fastapi import FastAPI
from pydantic import BaseModel, Field

from app.core.loader import HotichLoader, HotichBundle
from app.core.hybrid_retriever import HotichHybridRetriever
from app.core.hill_climbing_reranker import HotichHillClimbingReranker, HillClimbingConfig
from app.core.answer_builder import HotichAnswerBuilder
from app.core.context_builder import HotichContextBuilder
from app.core.llm_answerer import QwenLLMAnswerer
from app.core.router import HotichRouter


bundle: HotichBundle | None = None
retriever: HotichHybridRetriever | None = None
hc_reranker: HotichHillClimbingReranker | None = None
builder: HotichAnswerBuilder | None = None
context_builder: HotichContextBuilder | None = None
llm_answerer: QwenLLMAnswerer | None = None
router: HotichRouter | None = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    global bundle, retriever, hc_reranker, builder, context_builder, llm_answerer, router

    loader = HotichLoader()
    bundle = loader.load_all()

    retriever = HotichHybridRetriever(
        model_name="bkai-foundation-models/vietnamese-bi-encoder",
        use_semantic=True,
        lexical_weight=1.0,
        semantic_weight=20.0,
    )

    hc_reranker = HotichHillClimbingReranker(
        bundle=bundle,
        config=HillClimbingConfig(
            k=10,
            m=6,
            max_iter=20,
            lambda_redundancy=0.80,
            eta_constraint=2.00,
            gamma_coverage=0.50,
        ),
    )

    builder = HotichAnswerBuilder(
        bundle=bundle,
        retriever=retriever,
        hc_reranker=hc_reranker,
    )

    context_builder = HotichContextBuilder(
        max_procedures=1,
        max_legals=4,
        max_templates=1,
        max_cases=1,
        max_authorities=1,
    )

    # Sửa base_url / model_name nếu server Qwen của bạn khác
    llm_answerer = QwenLLMAnswerer(
        base_url="http://127.0.0.1:11434/v1",
        model_name="qwen2.5:3b",
        api_key="ollama",
        temperature=0.1,
        max_tokens=1200,
        timeout=120,
    )

    router = HotichRouter()

    print("=" * 80)
    print("HOTICH API STARTED")
    bundle.print_summary()
    print("=" * 80)

    yield


app = FastAPI(
    title="Hotich Assistant API",
    version="0.3.0",
    lifespan=lifespan,
)


class AskRequest(BaseModel):
    query: str = Field(..., min_length=1, description="Câu hỏi của người dùng")
    per_kind: int = Field(default=3, ge=1, le=10)
    use_llm: bool = Field(default=True, description="Có dùng Qwen2.5 hay không")


class SearchItem(BaseModel):
    kind: str
    item_id: str
    title: str
    score: float
    lexical_score: float
    semantic_score: float
    source_path: str
    snippet: str


class AskResponse(BaseModel):
    query: str
    intent: str
    answer_text: str
    answer_mode: str
    llm_used: bool
    llm_error: str | None = None
    scores: dict[str, float]
    results: dict[str, list[SearchItem]]
    stats: dict[str, Any]
    citation_map: dict[str, str] | None = None
    llm_mode: str | None = None

@app.get("/")
def root() -> dict[str, Any]:
    return {
        "message": "Hotich Assistant API is running",
        "docs": "/docs",
        "health": "/health",
    }


@app.get("/health")
def health() -> dict[str, Any]:
    if bundle is None:
        return {"status": "starting"}

    return {
        "status": "ok",
        "data_root": str(bundle.data_root),
        "stats": bundle.summary(),
    }


def _build_grouped_results(query: str, per_kind: int) -> dict[str, list[Any]]:
    if retriever is None or hc_reranker is None:
        raise RuntimeError("Retriever or reranker chưa khởi tạo")

    decision = router.route(query)


    procedure_k = max(per_kind, 2)
    legal_k = max(per_kind, 2)
    template_k = max(per_kind, 2)
    case_k = max(per_kind, 2)
    authority_k = max(per_kind, 2)

    if decision.primary_intent == "template":
        template_k = 6
        procedure_k = 2
        legal_k = 1
        case_k = 1
        authority_k = 1
    elif decision.primary_intent == "procedure":
        procedure_k = 5
        legal_k = 3
        template_k = 2
        case_k = 2
        authority_k = 2
    elif decision.primary_intent == "legal":
        legal_k = 6
        procedure_k = 2
        template_k = 1
        case_k = 1
        authority_k = 1

    groups = {
        "procedure": retriever.search(query, source_kinds=["procedure"], top_k=procedure_k, min_score=1.0),
        "legal": retriever.search(query, source_kinds=["legal"], top_k=legal_k, min_score=1.0),
        "template": retriever.search(query, source_kinds=["template"], top_k=template_k, min_score=1.0),
        "case": retriever.search(query, source_kinds=["case"], top_k=case_k, min_score=1.0),
        "authority": retriever.search(query, source_kinds=["authority"], top_k=authority_k, min_score=1.0),
    }

    total_k = procedure_k + legal_k + template_k + case_k + authority_k
    groups = hc_reranker.optimize_grouped(
        query=query,
        grouped_results=groups,
        total_k=total_k,
    )
    return groups


def _serialize_results(grouped_results: dict[str, list[Any]]) -> dict[str, list[SearchItem]]:
    result_payload: dict[str, list[SearchItem]] = {}
    for kind, rows in grouped_results.items():
        result_payload[kind] = [
            SearchItem(
                kind=r.kind,
                item_id=r.item_id,
                title=r.title,
                score=r.score,
                lexical_score=r.lexical_score,
                semantic_score=r.semantic_score,
                source_path=r.source_path,
                snippet=r.snippet,
            )
            for r in rows
        ]
    return result_payload


@app.post("/ask", response_model=AskResponse)
def ask(req: AskRequest) -> AskResponse:
    if (
        bundle is None
        or retriever is None
        or hc_reranker is None
        or builder is None
        or context_builder is None
        or llm_answerer is None
        or router is None
    ):
        raise RuntimeError("API chưa khởi tạo xong")

    decision = router.route(req.query)
    grouped_results = _build_grouped_results(req.query, req.per_kind)
    result_payload = _serialize_results(grouped_results)

    llm_error: str | None = None
    citation_map: dict[str, str] | None = None

    # Fallback mặc định
    fallback_answer = builder.answer(req.query, per_kind=req.per_kind).answer_text

    if req.use_llm:
        try:
            built_context = context_builder.build(
                query=req.query,
                grouped_results=grouped_results,
            )
            citation_map = built_context.citation_map

            llm_result = llm_answerer.answer(built_context)

            return AskResponse(
                query=req.query,
                intent=decision.primary_intent,
                answer_text=llm_result.answer_text,
                answer_mode="llm",
                llm_used=True,
                llm_error=None,
                llm_mode=llm_result.mode,
                scores=decision.scores,
                results=result_payload,
                stats=bundle.summary(),
                citation_map=citation_map,
            )
        except Exception as e:
            llm_error = f"{type(e).__name__}: {e}"
            print("\n[LLM FALLBACK]")
            print(llm_error)
            traceback.print_exc()

    return AskResponse(
        query=req.query,
        intent=decision.primary_intent,
        answer_text=fallback_answer,
        answer_mode="fallback",
        llm_used=False,
        llm_error=llm_error,
        llm_mode=None,
        scores=decision.scores,
        results=result_payload,
        stats=bundle.summary(),
        citation_map=citation_map,
    )