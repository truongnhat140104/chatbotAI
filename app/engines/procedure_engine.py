from __future__ import annotations

from typing import Any

from app.core.router import RouteDecision

from .base_engine import BaseEngine, EngineRunResult
from .common import (
    compact,
    dedupe_lines,
    find_values_by_key_patterns,
    keyword_overlap_score,
    make_search_result,
)


class ProcedureEngine(BaseEngine):
    name = "procedure_engine"
    query_mode = "procedure"

    def can_handle(self, query: str, route: RouteDecision, requested_mode: str = "auto") -> bool:
        if requested_mode == "procedure":
            return True
        return route.primary_intent == "procedure"

    def _procedure_score(self, query: str, proc: dict[str, Any]) -> float:
        raw = proc.get("raw", {})
        variants = proc.get("variants", [])
        tags = proc.get("tags", [])
        return (
            keyword_overlap_score(query, proc.get("title"), tags, variants) * 1.8
            + keyword_overlap_score(query, raw.get("name"), raw.get("keywords"), raw.get("aliases"))
            + keyword_overlap_score(query, raw.get("common_cases_notes"), raw.get("short_description")) * 0.8
        )

    def _extract_dossier(self, proc: dict[str, Any]) -> list[str]:
        raw = proc.get("raw", {})
        hits = find_values_by_key_patterns(
            raw,
            patterns=["ho so", "thanh phan", "giay to", "tai lieu", "dossier", "documents"],
            max_hits=6,
        )
        lines = [compact(v, limit=220) for v in hits]
        return dedupe_lines(lines, max_items=5)

    def _build_summary(self, proc: dict[str, Any]) -> str:
        lines: list[str] = [f"Thủ tục phù hợp nhất: {proc.get('title')}." ]

        dossier = self._extract_dossier(proc)
        if dossier:
            lines.append("Hồ sơ/giấy tờ nổi bật: " + "; ".join(dossier[:4]) + ".")

        authority = compact(proc.get("authority"), limit=260)
        if authority:
            lines.append(f"Thẩm quyền/nơi tiếp nhận: {authority}.")

        processing_time = compact(proc.get("processing_time"), limit=180)
        if processing_time:
            lines.append(f"Thời hạn giải quyết: {processing_time}.")

        fees = compact(proc.get("fees"), limit=180)
        if fees:
            lines.append(f"Lệ phí: {fees}.")

        submission = compact(proc.get("submission_methods"), limit=220)
        if submission:
            lines.append(f"Cách thức nộp: {submission}.")

        citations = compact(proc.get("citations"), limit=220)
        if citations:
            lines.append(f"Căn cứ/dẫn chiếu liên quan: {citations}.")

        return "\n".join(lines)

    def _make_procedure_result(self, proc: dict[str, Any], score: float) -> Any:
        summary = self._build_summary(proc)
        return make_search_result(
            kind="procedure",
            item_id=proc.get("procedure_id") or proc.get("id"),
            title=proc.get("title") or proc.get("procedure_id") or "Procedure",
            source_path=proc.get("source_path", ""),
            summary_text=summary,
            score=score,
        )

    def run(self, query: str, route: RouteDecision, per_kind: int = 3) -> EngineRunResult:
        groups = self.empty_groups()
        scored: list[tuple[float, dict[str, Any]]] = []
        for proc in self.bundle.procedures.values():
            score = self._procedure_score(query, proc)
            if score > 0:
                scored.append((score, proc))
        scored.sort(key=lambda x: (-x[0], x[1].get("title", "")))

        best = scored[:max(per_kind, 3)]
        groups["procedure"] = [self._make_procedure_result(proc, score) for score, proc in best]

        if groups["procedure"]:
            groups["authority"] = self.retriever.search(
                query=query,
                source_kinds=["authority"],
                top_k=min(2, per_kind),
                min_score=0.8,
            )
            groups["template"] = self.retriever.search(
                query=query,
                source_kinds=["template"],
                top_k=1,
                min_score=0.8,
            )
            groups["legal"] = self.retriever.search(
                query=query,
                source_kinds=["legal"],
                top_k=2,
                min_score=0.8,
            )
            answer = groups["procedure"][0].data["best_unit"]["text"]
        else:
            answer = (
                "Chưa xác định được một thủ tục đủ khớp bằng exact engine. "
                "Bạn có thể chuyển sang case engine hoặc nêu rõ tên thủ tục cần hỏi."
            )
            groups["case"] = self.retriever.search(
                query=query,
                source_kinds=["case"],
                top_k=1,
                min_score=0.8,
            )

        return EngineRunResult(
            engine_name=self.name,
            query_mode=self.query_mode,
            answer_text=answer,
            grouped_results=groups,
            debug={"matched_procedures": len(best)},
        )
