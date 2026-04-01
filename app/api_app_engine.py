from __future__ import annotations

import os
from functools import lru_cache
from typing import Any

from fastapi import FastAPI
from pydantic import BaseModel, Field

from app.core.context_builder import HotichContextBuilder
from app.core.hill_climbing_reranker import HotichHillClimbingReranker
from app.core.hybrid_retriever import HotichHybridRetriever, SearchResult
from app.core.llm_answerer import QwenLLMAnswerer
from app.core.loader import HotichLoader
from app.core.router import HotichRouter
from app.engines.case_rag_engine import CaseRAGEngine
from app.engines.legal_engine import LegalLookupEngine
from app.engines.procedure_engine import ProcedureEngine
from app.engines.selector import EngineSelector
from app.engines.template_engine import TemplateEngine


class AskRequest(BaseModel):
    query: str = Field(..., min_length=1, description="Câu hỏi người dùng")
    per_kind: int = Field(default=3, ge=1, le=10, description="Số lượng kết quả mỗi nhóm")
    use_llm: bool = Field(default=True, description="Có gọi LLM hay không")
    mode: str = Field(default="auto", description="Chế độ truy vấn: auto/legal/procedure/template/case")


class HotichEngineQAService:
    def __init__(self) -> None:
        data_root = os.getenv("HOTICH_DATA_ROOT")
        index_dir = os.getenv("HOTICH_INDEX_DIR")
        embeddings_dir = os.getenv("HOTICH_EMBEDDINGS_DIR")
        embed_model_name = os.getenv(
            "HOTICH_EMBED_MODEL",
            "bkai-foundation-models/vietnamese-bi-encoder",
        )
        use_semantic = os.getenv("HOTICH_USE_SEMANTIC", "true").strip().lower() != "false"
        use_hc = os.getenv("HOTICH_USE_HC", "true").strip().lower() != "false"

        llm_base_url = os.getenv("HOTICH_LLM_BASE_URL", "http://127.0.0.1:11434/v1")
        llm_model = os.getenv("HOTICH_LLM_MODEL", "qwen2.5:3b")
        llm_api_key = os.getenv("HOTICH_LLM_API_KEY", "not-needed")

        self.init_warnings: list[str] = []

        self.loader = HotichLoader(data_root=data_root)
        self.bundle = self.loader.load_all()
        self.router = HotichRouter()
        self.retriever = HotichHybridRetriever(
            index_dir=index_dir,
            embeddings_dir=embeddings_dir,
            model_name=embed_model_name,
            use_semantic=use_semantic,
        )

        self.hc_reranker: HotichHillClimbingReranker | None = None
        if use_hc:
            try:
                self.hc_reranker = HotichHillClimbingReranker(
                    bundle=self.bundle,
                    embeddings_dir=embeddings_dir,
                )
            except Exception as exc:
                self.init_warnings.append(f"Hill Climbing disabled: {exc}")

        self.context_builder = HotichContextBuilder()
        self.llm_answerer = QwenLLMAnswerer(
            base_url=llm_base_url,
            model_name=llm_model,
            api_key=llm_api_key,
        )

        self.legal_engine = LegalLookupEngine(
            bundle=self.bundle,
            retriever=self.retriever,
            hc_reranker=self.hc_reranker,
        )
        self.procedure_engine = ProcedureEngine(
            bundle=self.bundle,
            retriever=self.retriever,
            hc_reranker=self.hc_reranker,
        )
        self.template_engine = TemplateEngine(
            bundle=self.bundle,
            retriever=self.retriever,
            hc_reranker=self.hc_reranker,
        )
        self.case_engine = CaseRAGEngine(
            bundle=self.bundle,
            retriever=self.retriever,
            hc_reranker=self.hc_reranker,
        )
        self.selector = EngineSelector(
            legal_engine=self.legal_engine,
            procedure_engine=self.procedure_engine,
            template_engine=self.template_engine,
            case_engine=self.case_engine,
        )

    @staticmethod
    def _serialize_result(item: SearchResult) -> dict[str, Any]:
        return {
            "kind": item.kind,
            "item_id": item.item_id,
            "title": item.title,
            "score": item.score,
            "lexical_score": item.lexical_score,
            "semantic_score": item.semantic_score,
            "source_path": item.source_path,
            "snippet": item.snippet,
            "top_units": item.data.get("top_units", []) if isinstance(item.data, dict) else [],
        }

    def _serialize_groups(self, grouped_results: dict[str, list[SearchResult]]) -> dict[str, list[dict[str, Any]]]:
        out: dict[str, list[dict[str, Any]]] = {
            "procedure": [],
            "legal": [],
            "template": [],
            "case": [],
            "authority": [],
        }
        for kind, rows in grouped_results.items():
            out[kind] = [self._serialize_result(r) for r in rows]
        return out

    def health(self) -> dict[str, Any]:
        stats = self.bundle.summary()
        stats["init_warnings"] = self.init_warnings
        return {"status": "ok", "stats": stats}

    def ask(self, query: str, per_kind: int = 3, use_llm: bool = True, mode: str = "auto") -> dict[str, Any]:
        route = self.router.route(query)
        requested_mode = (mode or "auto").strip().lower()
        if requested_mode not in {"auto", "legal", "procedure", "template", "case"}:
            requested_mode = "auto"

        engine = self.selector.pick(query=query, route=route, requested_mode=requested_mode)
        engine_result = engine.run(query=query, route=route, per_kind=per_kind)

        built_context = self.context_builder.build(
            query=query,
            grouped_results=engine_result.grouped_results,
            forced_mode=engine_result.query_mode,
        )

        answer_text = engine_result.answer_text
        answer_mode = "engine_rule_based"
        llm_mode = built_context.query_mode
        llm_used = False
        llm_error = None

        missing_legal_context = built_context.query_mode in {"legal", "legal_article_lookup"} and not built_context.citation_map

        if use_llm and not missing_legal_context:
            try:
                llm_result = self.llm_answerer.answer(built_context)
                answer_text = llm_result.answer_text
                answer_mode = "engine_llm"
                llm_mode = llm_result.mode
                llm_used = True
            except Exception as exc:
                llm_error = str(exc)
                answer_mode = "engine_fallback_rule_based"
        elif missing_legal_context:
            llm_error = "Skipped LLM: legal engine chưa truy xuất được căn cứ pháp lý đủ tin cậy."

        return {
            "query": query,
            "requested_mode": requested_mode,
            "selected_engine": engine_result.engine_name,
            "intent": route.primary_intent if requested_mode == "auto" else requested_mode,
            "auto_intent": route.primary_intent,
            "scores": route.scores,
            "reasons": route.reasons,
            "sub_intent": route.sub_intent,
            "article_no": route.article_no,
            "clause_no": route.clause_no,
            "law_alias": route.law_alias,
            "query_mode": built_context.query_mode,
            "answer_text": answer_text,
            "answer_mode": answer_mode,
            "llm_mode": llm_mode,
            "llm_used": llm_used,
            "llm_error": llm_error,
            "citation_map": built_context.citation_map,
            "selected_items": built_context.selected_items,
            "engine_debug": engine_result.debug,
            "results": self._serialize_groups(engine_result.grouped_results),
        }


@lru_cache(maxsize=1)
def get_service() -> HotichEngineQAService:
    return HotichEngineQAService()


app = FastAPI(
    title="Hotich Engine QA API",
    version="2.0.0",
    description="API engine-based: router -> engine selector -> exact engine / case RAG engine -> context -> LLM",
)


@app.get("/health")
def health() -> dict[str, Any]:
    return get_service().health()


@app.post("/ask")
def ask(request: AskRequest) -> dict[str, Any]:
    return get_service().ask(
        query=request.query,
        per_kind=request.per_kind,
        use_llm=request.use_llm,
        mode=request.mode,
    )
