from __future__ import annotations

from app.core.router import RouteDecision

from .base_engine import BaseEngine, EngineRunResult


class CaseRAGEngine(BaseEngine):
    name = "case_rag_engine"
    query_mode = "case"

    def can_handle(self, query: str, route: RouteDecision, requested_mode: str = "auto") -> bool:
        return requested_mode == "case" or route.primary_intent == "case"

    def _rule_based_answer(self, grouped_results: dict) -> str:
        parts: list[str] = []
        if grouped_results.get("procedure"):
            top = grouped_results["procedure"][0]
            parts.append(f"Thủ tục gần nhất với tình huống là: {top.title}.")
        if grouped_results.get("case"):
            top = grouped_results["case"][0]
            parts.append(f"Tình huống tham chiếu: {top.title}.")
        if grouped_results.get("legal"):
            top = grouped_results["legal"][0]
            parts.append(f"Căn cứ pháp lý nên đối chiếu thêm: {top.title}.")
        if grouped_results.get("authority"):
            top = grouped_results["authority"][0]
            parts.append(f"Nguồn thẩm quyền liên quan: {top.title}.")
        if not parts:
            return "Chưa truy xuất được ngữ cảnh phù hợp cho câu hỏi tình huống này."
        return "\n".join(parts)

    def run(self, query: str, route: RouteDecision, per_kind: int = 3) -> EngineRunResult:
        groups = self.retriever.grouped_search(query=query, per_kind=per_kind)
        debug = {"strategy": "grouped_search"}

        if self.hc_reranker is not None:
            groups = self.hc_reranker.optimize_grouped(
                query=query,
                grouped_results=groups,
                total_k=max(per_kind * 3, 8),
            )
            debug["reranked"] = True
        else:
            debug["reranked"] = False

        return EngineRunResult(
            engine_name=self.name,
            query_mode=self.query_mode,
            answer_text=self._rule_based_answer(groups),
            grouped_results=groups,
            debug=debug,
        )
