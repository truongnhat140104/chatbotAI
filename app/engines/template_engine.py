from __future__ import annotations

from typing import Any

from app.core.router import RouteDecision

from .base_engine import BaseEngine, EngineRunResult
from .common import compact, dedupe_lines, keyword_overlap_score, make_search_result, text


class TemplateEngine(BaseEngine):
    name = "template_engine"
    query_mode = "template"

    def can_handle(self, query: str, route: RouteDecision, requested_mode: str = "auto") -> bool:
        if requested_mode == "template":
            return True
        return route.primary_intent == "template"

    def _template_score(self, query: str, tpl: dict[str, Any]) -> float:
        return (
            keyword_overlap_score(query, tpl.get("title"), tpl.get("loai")) * 1.8
            + keyword_overlap_score(query, tpl.get("fields"), tpl.get("notes"), tpl.get("render"))
        )

    def _extract_field_names(self, fields: Any) -> list[str]:
        names: list[str] = []
        if isinstance(fields, list):
            for item in fields:
                if isinstance(item, dict):
                    candidate = text(item.get("label")) or text(item.get("name")) or text(item.get("field"))
                    if candidate:
                        names.append(candidate)
                else:
                    candidate = compact(item, limit=120)
                    if candidate:
                        names.append(candidate)
        elif isinstance(fields, dict):
            for k, v in fields.items():
                label = text(v.get("label")) if isinstance(v, dict) else ""
                names.append(label or text(k))
        return dedupe_lines(names, max_items=8)

    def _build_summary(self, tpl: dict[str, Any]) -> str:
        lines = [f"Biểu mẫu phù hợp nhất: {tpl.get('title')}." ]
        loai = text(tpl.get("loai"))
        if loai:
            lines.append(f"Loại biểu mẫu: {loai}.")
        output_type = text(tpl.get("output_type"))
        if output_type:
            lines.append(f"Định dạng đầu ra: {output_type}.")
        field_names = self._extract_field_names(tpl.get("fields"))
        if field_names:
            lines.append("Các trường thông tin chính: " + "; ".join(field_names[:8]) + ".")
        notes = compact(tpl.get("notes"), limit=240)
        if notes:
            lines.append(f"Lưu ý: {notes}.")
        render_hint = compact(tpl.get("render"), limit=240)
        if render_hint:
            lines.append(f"Thông tin render/cấu trúc: {render_hint}.")
        return "\n".join(lines)

    def _make_template_result(self, tpl: dict[str, Any], score: float) -> Any:
        summary = self._build_summary(tpl)
        return make_search_result(
            kind="template",
            item_id=tpl.get("template_id") or tpl.get("id"),
            title=tpl.get("title") or tpl.get("template_id") or "Template",
            source_path=tpl.get("source_path", ""),
            summary_text=summary,
            score=score,
        )

    def run(self, query: str, route: RouteDecision, per_kind: int = 3) -> EngineRunResult:
        groups = self.empty_groups()
        scored: list[tuple[float, dict[str, Any]]] = []
        for tpl in self.bundle.templates.values():
            score = self._template_score(query, tpl)
            if score > 0:
                scored.append((score, tpl))
        scored.sort(key=lambda x: (-x[0], x[1].get("title", "")))

        best = scored[:max(per_kind, 3)]
        groups["template"] = [self._make_template_result(tpl, score) for score, tpl in best]

        if groups["template"]:
            groups["procedure"] = self.retriever.search(
                query=query,
                source_kinds=["procedure"],
                top_k=1,
                min_score=0.8,
            )
            groups["legal"] = self.retriever.search(
                query=query,
                source_kinds=["legal"],
                top_k=2,
                min_score=0.8,
            )
            answer = groups["template"][0].data["best_unit"]["text"]
        else:
            answer = (
                "Chưa xác định được biểu mẫu phù hợp bằng exact engine. "
                "Bạn có thể nêu rõ tên mẫu, loại tờ khai hoặc thủ tục liên quan."
            )

        return EngineRunResult(
            engine_name=self.name,
            query_mode=self.query_mode,
            answer_text=answer,
            grouped_results=groups,
            debug={"matched_templates": len(best)},
        )
