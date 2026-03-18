from __future__ import annotations

from dataclasses import dataclass
from typing import Any
import re
import unicodedata

from app.core.hybrid_retriever import SearchResult


@dataclass
class BuiltContext:
    query: str
    context_text: str
    citation_map: dict[str, str]
    selected_items: dict[str, list[dict[str, Any]]]


class HotichContextBuilder:
    """
    Xây context sạch cho LLM từ grouped_results sau retrieval/rerank.
    """

    def __init__(
        self,
        max_procedures: int = 1,
        max_legals: int = 4,
        max_templates: int = 1,
        max_cases: int = 1,
        max_authorities: int = 1,
    ) -> None:
        self.max_procedures = max_procedures
        self.max_legals = max_legals
        self.max_templates = max_templates
        self.max_cases = max_cases
        self.max_authorities = max_authorities

    # ------------------------------------------------------------------
    # helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _text(value: Any) -> str:
        if value is None:
            return ""
        if isinstance(value, str):
            return value.strip()
        if isinstance(value, (int, float, bool)):
            return str(value)
        return ""

    def _normalize_text(self, text: Any) -> str:
        text = self._text(text)
        if not text:
            return ""

        text = text.lower().strip()
        text = unicodedata.normalize("NFD", text)
        text = "".join(ch for ch in text if unicodedata.category(ch) != "Mn")
        text = text.replace("đ", "d").replace("Đ", "d")
        text = re.sub(r"[^a-z0-9\s]", " ", text)
        text = re.sub(r"\s+", " ", text).strip()
        return text

    def _detect_law_alias(self, query: str) -> str:
        q = self._normalize_text(query)

        if "hon nhan va gia dinh" in q:
            return "luat_hon_nhan_va_gia_dinh"
        if "luat ho tich" in q:
            return "luat_ho_tich"
        return ""

    def _extract_article_lookup(self, query: str) -> dict[str, str]:
        q = self._normalize_text(query)

        m_article = re.search(r"\bdieu\s+(\d+)\b", q)
        article_no = m_article.group(1) if m_article else ""

        m_clause = re.search(r"\bkhoan\s+(\d+)\b", q)
        clause_no = m_clause.group(1) if m_clause else ""

        law_alias = self._detect_law_alias(query)

        if article_no and law_alias:
            return {
                "sub_intent": "legal_article_lookup",
                "article_no": article_no,
                "clause_no": clause_no,
                "law_alias": law_alias,
            }

        return {
            "sub_intent": "",
            "article_no": "",
            "clause_no": "",
            "law_alias": "",
        }

    def _query_mode(self, query: str) -> str:
        q = self._normalize_text(query)
        article_lookup = self._extract_article_lookup(query)

        if article_lookup["sub_intent"] == "legal_article_lookup":
            return "legal_article_lookup"

        if any(x in q for x in ["to khai", "mau", "bieu mau", "form"]):
            return "template"

        if any(x in q for x in [
            "van ban", "can cu", "phap ly", "dieu", "khoan",
            "nguyen tac co ban", "quyen va nghia vu", "noi dung dieu"
        ]):
            return "legal"

        if any(x in q for x in ["truong hop", "mat", "khong co", "uy quyen", "qua han"]):
            return "case"

        return "procedure"

    def _compact(self, value: Any, max_len: int = 300) -> str:
        text = self._flatten(value, limit=max_len)
        text = " ".join(text.split())
        return text[:max_len].rstrip()

    def _flatten(self, value: Any, limit: int = 1200) -> str:
        parts: list[str] = []

        def walk(x: Any) -> None:
            if len(" ".join(parts)) >= limit:
                return

            if x is None:
                return
            if isinstance(x, str):
                s = x.strip()
                if s:
                    parts.append(s)
                return
            if isinstance(x, (int, float, bool)):
                parts.append(str(x))
                return
            if isinstance(x, list):
                for item in x:
                    walk(item)
                return
            if isinstance(x, dict):
                for v in x.values():
                    walk(v)

        walk(value)
        return " ".join(parts)[:limit].strip()

    def _format_list(self, items: list[str], max_items: int = 5) -> str:
        clean = []
        seen = set()
        for item in items:
            item = " ".join(item.split()).strip()
            if item and item not in seen:
                seen.add(item)
                clean.append(item)
            if len(clean) >= max_items:
                break

        if not clean:
            return ""

        return "\n".join(f"- {x}" for x in clean)

    # ------------------------------------------------------------------
    # extractors
    # ------------------------------------------------------------------

    def _extract_procedure(self, result: SearchResult) -> dict[str, Any]:
        data = result.data or {}
        best_unit = data.get("best_unit", {}) if isinstance(data, dict) else {}

        title = result.title
        source_path = result.source_path

        raw = best_unit if isinstance(best_unit, dict) else {}
        text = self._text(raw.get("text")) or result.snippet

        return {
            "kind": "procedure",
            "item_id": result.item_id,
            "title": title,
            "score": result.score,
            "summary": text[:1200],
            "source_path": source_path,
        }

    def _extract_legal(self, result: SearchResult) -> dict[str, Any]:
        data = result.data or {}
        top_units = data.get("top_units", []) if isinstance(data, dict) else []
        best_unit = data.get("best_unit", {}) if isinstance(data, dict) else {}

        citation_label = self._text(best_unit.get("citation_label"))
        doc_title = self._text(data.get("doc_title")) or result.title
        doc_number = self._text(data.get("doc_number"))
        snippet = self._text(best_unit.get("text")) or result.snippet

        return {
            "kind": "legal",
            "item_id": result.item_id,
            "title": doc_title,
            "doc_id": self._text(best_unit.get("doc_id")),
            "doc_number": doc_number,
            "citation_label": citation_label,
            "article_no": self._text(best_unit.get("article_no")),
            "clause_key": self._text(best_unit.get("clause_key")),
            "article_title": self._text(best_unit.get("article_title")),
            "score": result.score,
            "top_units": top_units[:3],
            "summary": snippet[:1500],
            "source_path": result.source_path,
        }

    def _extract_template(self, result: SearchResult) -> dict[str, Any]:
        data = result.data or {}
        best_unit = data.get("best_unit", {}) if isinstance(data, dict) else {}

        return {
            "kind": "template",
            "item_id": result.item_id,
            "title": result.title,
            "score": result.score,
            "summary": (self._text(best_unit.get("text")) or result.snippet)[:1000],
            "source_path": result.source_path,
        }

    def _extract_case(self, result: SearchResult) -> dict[str, Any]:
        data = result.data or {}
        best_unit = data.get("best_unit", {}) if isinstance(data, dict) else {}

        return {
            "kind": "case",
            "item_id": result.item_id,
            "title": result.title,
            "score": result.score,
            "summary": (self._text(best_unit.get("text")) or result.snippet)[:1200],
            "source_path": result.source_path,
        }

    def _extract_authority(self, result: SearchResult) -> dict[str, Any]:
        data = result.data or {}
        best_unit = data.get("best_unit", {}) if isinstance(data, dict) else {}

        return {
            "kind": "authority",
            "item_id": result.item_id,
            "title": result.title,
            "score": result.score,
            "summary": (self._text(best_unit.get("text")) or result.snippet)[:1000],
            "source_path": result.source_path,
        }

    # ------------------------------------------------------------------
    # builders by kind
    # ------------------------------------------------------------------

    def _build_procedure_block(
        self,
        items: list[dict[str, Any]],
        citation_map: dict[str, str],
        citation_counter: list[int],
    ) -> str:
        if not items:
            return ""

        lines = ["## PROCEDURE CONTEXT"]
        for item in items:
            c = f"C{citation_counter[0]}"
            citation_counter[0] += 1

            citation_map[c] = f"{item['title']} | {item['source_path']}"
            lines.append(f"[{c}] {item['title']}")
            lines.append(item["summary"])
            lines.append("")

        return "\n".join(lines).strip()

    def _build_legal_block(
        self,
        items: list[dict[str, Any]],
        citation_map: dict[str, str],
        citation_counter: list[int],
        query: str | None = None,
    ) -> str:
        if not items:
            return ""

        query_mode = self._query_mode(query or "")
        article_lookup = self._extract_article_lookup(query or "")

        if query_mode == "legal_article_lookup":
            filtered = []
            for item in items:
                same_article = item.get("article_no", "") == article_lookup["article_no"]

                if article_lookup["clause_no"]:
                    same_clause = item.get("clause_key", "") == article_lookup["clause_no"]
                    if same_article and same_clause:
                        filtered.append(item)
                else:
                    if same_article:
                        filtered.append(item)

            if filtered:
                items = filtered
            else:
                items = items[:2]

        lines = ["## LEGAL CONTEXT"]
        for item in items:
            c = f"C{citation_counter[0]}"
            citation_counter[0] += 1

            label = item.get("citation_label") or item["title"]
            if item.get("doc_number"):
                citation_map[c] = f"{item['title']} ({item['doc_number']}) - {label}"
            else:
                citation_map[c] = f"{item['title']} - {label}"

            lines.append(f"[{c}] {item['title']}")
            if item.get("citation_label"):
                lines.append(f"Citation: {item['citation_label']}")
            lines.append(item["summary"])

            top_units = item.get("top_units", [])
            if top_units:
                unit_lines = []
                for u in top_units[:3]:
                    u_label = self._text(u.get("citation_label")) or self._text(u.get("unit_id"))
                    u_text = self._text(u.get("text"))
                    if u_label:
                        unit_lines.append(f"{u_label}: {u_text[:240]}")
                if unit_lines:
                    lines.append("Relevant units:")
                    lines.append(self._format_list(unit_lines, max_items=3))

            lines.append("")

        return "\n".join(lines).strip()

    def _build_template_block(
        self,
        items: list[dict[str, Any]],
        citation_map: dict[str, str],
        citation_counter: list[int],
    ) -> str:
        if not items:
            return ""

        lines = ["## TEMPLATE CONTEXT"]
        for item in items:
            c = f"C{citation_counter[0]}"
            citation_counter[0] += 1

            citation_map[c] = f"{item['title']} | {item['source_path']}"
            lines.append(f"[{c}] {item['title']}")
            lines.append(item["summary"])
            lines.append("")

        return "\n".join(lines).strip()

    def _build_case_block(
        self,
        items: list[dict[str, Any]],
        citation_map: dict[str, str],
        citation_counter: list[int],
    ) -> str:
        if not items:
            return ""

        lines = ["## CASE CONTEXT"]
        for item in items:
            c = f"C{citation_counter[0]}"
            citation_counter[0] += 1

            citation_map[c] = f"{item['title']} | {item['source_path']}"
            lines.append(f"[{c}] {item['title']}")
            lines.append(item["summary"])
            lines.append("")

        return "\n".join(lines).strip()

    def _build_authority_block(
        self,
        items: list[dict[str, Any]],
        citation_map: dict[str, str],
        citation_counter: list[int],
    ) -> str:
        if not items:
            return ""

        lines = ["## AUTHORITY CONTEXT"]
        for item in items:
            c = f"C{citation_counter[0]}"
            citation_counter[0] += 1

            citation_map[c] = f"{item['title']} | {item['source_path']}"
            lines.append(f"[{c}] {item['title']}")
            lines.append(item["summary"])
            lines.append("")

        return "\n".join(lines).strip()

    # ------------------------------------------------------------------
    # public build
    # ------------------------------------------------------------------

    def build(
        self,
        query: str,
        grouped_results: dict[str, list[SearchResult]],
    ) -> BuiltContext:
        selected_items: dict[str, list[dict[str, Any]]] = {
            "procedure": [],
            "legal": [],
            "template": [],
            "case": [],
            "authority": [],
        }

        query_mode = self._query_mode(query)

        if query_mode == "legal_article_lookup":
            for r in grouped_results.get("legal", [])[: self.max_legals]:
                selected_items["legal"].append(self._extract_legal(r))

        elif query_mode == "template":
            for r in grouped_results.get("template", [])[: max(self.max_templates, 2)]:
                selected_items["template"].append(self._extract_template(r))

            for r in grouped_results.get("procedure", [])[:1]:
                selected_items["procedure"].append(self._extract_procedure(r))

        elif query_mode == "legal":
            for r in grouped_results.get("legal", [])[: self.max_legals]:
                selected_items["legal"].append(self._extract_legal(r))

        else:
            for r in grouped_results.get("procedure", [])[: self.max_procedures]:
                selected_items["procedure"].append(self._extract_procedure(r))

            for r in grouped_results.get("legal", [])[: self.max_legals]:
                selected_items["legal"].append(self._extract_legal(r))

            for r in grouped_results.get("template", [])[: self.max_templates]:
                selected_items["template"].append(self._extract_template(r))

            for r in grouped_results.get("case", [])[: self.max_cases]:
                selected_items["case"].append(self._extract_case(r))

            for r in grouped_results.get("authority", [])[: self.max_authorities]:
                selected_items["authority"].append(self._extract_authority(r))

        citation_map: dict[str, str] = {}
        citation_counter = [1]

        if query_mode == "legal_article_lookup":
            blocks = [
                f"# USER QUERY\n{query}",
                self._build_legal_block(selected_items["legal"], citation_map, citation_counter, query=query),
            ]
        else:
            blocks = [
                f"# USER QUERY\n{query}",
                self._build_procedure_block(selected_items["procedure"], citation_map, citation_counter),
                self._build_legal_block(selected_items["legal"], citation_map, citation_counter, query=query),
                self._build_template_block(selected_items["template"], citation_map, citation_counter),
                self._build_case_block(selected_items["case"], citation_map, citation_counter),
                self._build_authority_block(selected_items["authority"], citation_map, citation_counter),
            ]

        context_text = "\n\n".join(block for block in blocks if block).strip()

        return BuiltContext(
            query=query,
            context_text=context_text,
            citation_map=citation_map,
            selected_items=selected_items,
        )