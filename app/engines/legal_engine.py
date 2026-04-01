from __future__ import annotations

from typing import Any

from app.core.router import RouteDecision

from .base_engine import BaseEngine, EngineRunResult
from .common import compact, normalize_text


class LegalLookupEngine(BaseEngine):
    name = "legal_lookup_engine"
    query_mode = "legal"

    def can_handle(self, query: str, route: RouteDecision, requested_mode: str = "auto") -> bool:
        if requested_mode == "legal":
            return True
        return route.primary_intent == "legal" or route.sub_intent == "legal_article_lookup"

    def _build_answer_from_result(self, query: str, result: Any, route: RouteDecision) -> str:
        data = result.data or {}
        best_unit = data.get("best_unit", {}) if isinstance(data, dict) else {}
        title = result.title or data.get("doc_title") or "Văn bản pháp lý"
        citation = best_unit.get("citation_label") or title
        snippet = compact(best_unit.get("text") or result.snippet, limit=1200)

        if route.sub_intent == "legal_article_lookup":
            article = best_unit.get("article_no") or route.article_no
            clause = best_unit.get("clause_key") or route.clause_no
            header = f"Tra cứu Điều {article}"
            if clause:
                header += f", khoản {clause}"
            header += f" của {title}:"
            return f"{header}\n- Trích yếu: {snippet}\n- Căn cứ: {citation}."

        return (
            f"Căn cứ pháp lý phù hợp nhất hiện tại là: {title}.\n"
            f"- Nội dung liên quan: {snippet}\n"
            f"- Dẫn chiếu: {citation}."
        )

    def _fallback_no_result(self, query: str) -> str:
        return (
            "Chưa truy xuất được căn cứ pháp lý đủ tin cậy từ kho dữ liệu hiện có cho câu hỏi này. "
            "Bạn có thể thử nêu rõ tên luật, số điều hoặc khoản để engines tra cứu chính xác hơn."
        )

    def run(self, query: str, route: RouteDecision, per_kind: int = 3) -> EngineRunResult:
        groups = self.empty_groups()
        debug: dict[str, Any] = {"strategy": "legal_only_search"}

        if route.sub_intent == "legal_article_lookup" and route.article_no and route.law_alias:
            debug["strategy"] = "exact_article_lookup"
            rows = self.retriever.search_legal_article_lookup(
                query=query,
                law_alias=route.law_alias,
                article_no=route.article_no,
                clause_no=route.clause_no,
                top_k=max(per_kind, 3),
                min_score=0.5,
            )
            groups["legal"] = rows
            answer = self._build_answer_from_result(query, rows[0], route) if rows else self._fallback_no_result(query)
            return EngineRunResult(
                engine_name=self.name,
                query_mode="legal_article_lookup",
                answer_text=answer,
                grouped_results=groups,
                debug=debug,
            )

        rows = self.retriever.search(
            query=query,
            source_kinds=["legal"],
            top_k=max(per_kind, 3),
            min_score=0.8,
        )

        law_alias = normalize_text(route.law_alias)
        if law_alias:
            filtered = []
            for r in rows:
                title = normalize_text(r.title)
                doc_title = normalize_text((r.data or {}).get("doc_title"))
                if law_alias in {"luat_ho_tich", "luat hon nhan va gia dinh"}:
                    pass
                if (
                    (law_alias == "luat_ho_tich" and ("ho tich" in title or "ho tich" in doc_title))
                    or (law_alias == "luat_hon_nhan_va_gia_dinh" and ("hon nhan va gia dinh" in title or "hon nhan va gia dinh" in doc_title))
                ):
                    filtered.append(r)
            if filtered:
                rows = filtered

        groups["legal"] = rows[:per_kind]
        answer = self._build_answer_from_result(query, rows[0], route) if rows else self._fallback_no_result(query)

        return EngineRunResult(
            engine_name=self.name,
            query_mode="legal",
            answer_text=answer,
            grouped_results=groups,
            debug=debug,
        )
