from __future__ import annotations
from app.core.hill_climbing_reranker import HotichHillClimbingReranker
import re
import unicodedata
from dataclasses import dataclass
from typing import Any

from app.core.loader import HotichBundle, HotichLoader
from app.core.hybrid_retriever import HotichHybridRetriever, SearchResult

@dataclass
class AnswerResult:
    query: str
    answer_text: str
    grouped_results: dict[str, list[SearchResult]]


class HotichAnswerBuilder:
    def __init__(
        self,
        bundle: HotichBundle,
        retriever: HotichHybridRetriever,
        hc_reranker: HotichHillClimbingReranker | None = None,
    ) -> None:
        self.bundle = bundle
        self.retriever = retriever
        self.hc_reranker = hc_reranker

    # ------------------------------------------------------------------
    # Text / normalize helpers
    # ------------------------------------------------------------------

    def _is_broad_legal_query(self, query: str) -> bool:
        q = self._normalize_text(query)
        return (
            "ho tich" in q
            and any(x in q for x in ["van ban", "quy dinh", "can cu", "phap ly"])
        )

    def _filter_results_for_legal_query(
        self,
        query: str,
        results: list[SearchResult],
        anchor_title: str = "",
        max_items: int = 2,
    ) -> list[SearchResult]:
        picked: list[SearchResult] = []
        for r in results:
            title = r.title or ""
            snippet = r.snippet or ""
            combined = f"{title} {snippet}"

            if self._looks_related(query, combined, anchor_title):
                picked.append(r)

        return picked[:max_items]

    @staticmethod
    def _text(value: Any) -> str:
        if value is None:
            return ""
        if isinstance(value, str):
            return value.strip()
        if isinstance(value, (int, float, bool)):
            return str(value)
        return ""

    @staticmethod
    def _normalize_text(text: Any) -> str:
        if text is None:
            return ""
        if not isinstance(text, str):
            text = str(text)

        text = text.strip().lower()
        text = unicodedata.normalize("NFD", text)
        text = "".join(ch for ch in text if unicodedata.category(ch) != "Mn")
        text = text.replace("đ", "d").replace("Đ", "d")
        text = re.sub(r"[^a-z0-9\s]", " ", text)
        text = re.sub(r"\s+", " ", text).strip()
        return text

    @classmethod
    def _tokens(cls, text: Any) -> set[str]:
        stopwords = {
            "dang", "ky", "can", "giay", "to", "gi", "giayto", "thu", "tuc",
            "mau", "to", "khai", "o", "dau", "lam", "sao", "cho", "va", "cua",
            "co", "khong", "nhung", "neu", "tai", "voi", "ve", "ban", "sao",
        }
        toks = set(cls._normalize_text(text).split())
        return {t for t in toks if t and t not in stopwords and len(t) >= 2}

    def _compact(self, value: Any, max_len: int = 300) -> str:
        text = self._format_value(value, indent=0)
        text = " ".join(text.split())
        return text[:max_len].rstrip()

    def _format_value(self, value: Any, indent: int = 0) -> str:
        pad = "  " * indent

        if value is None:
            return ""

        if isinstance(value, str):
            return value.strip()

        if isinstance(value, (int, float, bool)):
            return str(value)

        if isinstance(value, list):
            lines: list[str] = []
            for item in value:
                item_text = self._format_value(item, indent + 1).strip()
                if item_text:
                    lines.append(f"{pad}- {item_text}")
            return "\n".join(lines)

        if isinstance(value, dict):
            lines: list[str] = []
            for k, v in value.items():
                v_text = self._format_value(v, indent + 1).strip()
                if v_text:
                    lines.append(f"{pad}- {k}: {v_text}")
            return "\n".join(lines)

        return str(value)

    def _find_values_by_key_patterns(
        self,
        obj: Any,
        patterns: list[str],
        max_hits: int = 3,
    ) -> list[Any]:
        hits: list[Any] = []

        def normalize_key(k: str) -> str:
            return k.lower().replace("_", " ").strip()

        def walk(x: Any) -> None:
            if len(hits) >= max_hits:
                return

            if isinstance(x, dict):
                for k, v in x.items():
                    nk = normalize_key(str(k))
                    if any(p in nk for p in patterns):
                        hits.append(v)
                        if len(hits) >= max_hits:
                            return
                    walk(v)

            elif isinstance(x, list):
                for item in x:
                    walk(item)

        walk(obj)
        return hits

    def _first_nonempty_from_patterns(
        self,
        obj: Any,
        patterns: list[str],
        max_hits: int = 3,
        max_len: int = 400,
    ) -> str:
        values = self._find_values_by_key_patterns(obj, patterns=patterns, max_hits=max_hits)
        if not values:
            return ""

        parts: list[str] = []
        for v in values:
            t = self._compact(v, max_len=max_len)
            if t:
                parts.append(t)

        if not parts:
            return ""

        uniq: list[str] = []
        seen = set()
        for p in parts:
            if p not in seen:
                seen.add(p)
                uniq.append(p)

        return "\n".join(f"- {p}" for p in uniq[:max_hits])

    def _resolve_legal_title(self, legal_id: str, fallback: str = "") -> str:
        meta = self.bundle.meta.get(legal_id)
        if meta:
            doc = meta.get("doc", {})
            if isinstance(doc, dict):
                title = self._text(doc.get("title"))
                number = self._text(doc.get("number"))
                if title and number:
                    return f"{title} ({number})"
                if title:
                    return title
        return fallback or legal_id

    def _extract_doc_refs(self, value: Any) -> list[str]:
        refs: list[str] = []

        def walk(x: Any) -> None:
            if isinstance(x, dict):
                for k, v in x.items():
                    if str(k).lower() in {"doc_id", "document_id", "ref_id"} and isinstance(v, str):
                        refs.append(v)
                    walk(v)
            elif isinstance(x, list):
                for item in x:
                    walk(item)

        walk(value)
        out: list[str] = []
        seen = set()
        for r in refs:
            if r not in seen:
                seen.add(r)
                out.append(r)
        return out

    def _dedupe_lines(self, items: list[str], max_items: int = 5) -> list[str]:
        out: list[str] = []
        seen = set()
        for x in items:
            key = self._normalize_text(x)
            if x and key not in seen:
                seen.add(key)
                out.append(x.strip())
            if len(out) >= max_items:
                break
        return out

    def _looks_related(self, query: str, candidate_text: str, anchor_text: str = "") -> bool:
        q = self._tokens(query)
        c = self._tokens(candidate_text)
        a = self._tokens(anchor_text)
        return bool((q & c) or (a & c))

    # ------------------------------------------------------------------
    # Domain-specific formatters
    # ------------------------------------------------------------------

    def _format_authority(self, authority: Any) -> list[str]:
        if not isinstance(authority, dict):
            text = self._compact(authority, max_len=250)
            return [text] if text else []

        lines: list[str] = []

        body = self._text(authority.get("resolve_body"))
        level = self._text(authority.get("resolve_level"))
        role = self._text(authority.get("resolve_role"))
        notes = self._text(authority.get("notes"))

        first = body
        if level:
            level_map = {
                "xa": "cấp xã",
                "huyen": "cấp huyện",
                "tinh": "cấp tỉnh",
            }
            level_text = level_map.get(level.lower(), level)
            if body:
                first = f"{body} ({level_text})"
            else:
                first = level_text

        if role and first:
            first = f"{first}; người tiếp nhận/xử lý: {role}"
        elif role:
            first = f"Người tiếp nhận/xử lý: {role}"

        if first:
            lines.append(first)
        if notes:
            lines.append(notes)

        if not lines:
            compact = self._compact(authority, max_len=300)
            if compact:
                lines.append(compact)

        return self._dedupe_lines(lines, max_items=3)

    def _format_dossier_items(self, raw: dict[str, Any]) -> list[str]:
        hits = self._find_values_by_key_patterns(
            raw,
            patterns=["ho so", "thanh phan", "giay to", "tai lieu", "must submit", "documents"],
            max_hits=5,
        )

        lines: list[str] = []

        def walk(v: Any) -> None:
            if isinstance(v, list):
                for item in v:
                    walk(item)
                return

            if isinstance(v, dict):
                name = self._text(v.get("name"))
                quantity = self._text(v.get("quantity"))
                when = self._text(v.get("when"))
                notes = self._text(v.get("notes"))

                if name:
                    s = name
                    extras: list[str] = []
                    if quantity:
                        extras.append(quantity)
                    if when:
                        extras.append(when)
                    if notes:
                        extras.append(notes)
                    if extras:
                        s = f"{s} ({'; '.join(extras)})"
                    lines.append(s)
                    return

                for child in v.values():
                    walk(child)
                return

            text = self._text(v)
            if text:
                lines.append(text)

        for hit in hits:
            walk(hit)

        return self._dedupe_lines(lines, max_items=6)

    def _format_processing_time(self, value: Any) -> list[str]:
        if isinstance(value, dict):
            lines: list[str] = []
            standard = self._text(value.get("standard"))
            extended = self._text(value.get("extended_if_verification_needed"))
            if standard:
                lines.append(f"Thông thường: {standard}")
            if extended:
                lines.append(f"Khi cần xác minh: {extended}")
            return self._dedupe_lines(lines, max_items=3)

        text = self._compact(value, max_len=250)
        return [text] if text else []

    def _format_fees(self, value: Any) -> list[str]:
        if isinstance(value, dict):
            policy = self._text(value.get("policy"))
            exemptions = value.get("exemptions")
            lines: list[str] = []
            if policy:
                lines.append(policy)

            if isinstance(exemptions, list) and exemptions:
                ex = "; ".join(self._text(x) for x in exemptions if self._text(x))
                if ex:
                    lines.append(f"Miễn/giảm: {ex}")

            return self._dedupe_lines(lines, max_items=3)

        text = self._compact(value, max_len=250)
        return [text] if text else []

    def _format_submission_methods(self, value: Any) -> list[str]:
        method_map = {
            "truc_tiep": "Nộp trực tiếp",
            "truc_tuyen": "Nộp trực tuyến",
            "buu_chinh": "Nộp qua bưu chính",
        }

        lines: list[str] = []

        if isinstance(value, list):
            for item in value:
                if isinstance(item, dict):
                    method = self._text(item.get("method"))
                    note = self._text(item.get("note"))
                    label = method_map.get(method, method or "Khác")
                    if note:
                        lines.append(f"{label} ({note})")
                    else:
                        lines.append(label)
                else:
                    text = self._text(item)
                    if text:
                        lines.append(text)

        elif isinstance(value, dict):
            method = self._text(value.get("method"))
            note = self._text(value.get("note"))
            label = method_map.get(method, method or "")
            if label:
                lines.append(f"{label} ({note})" if note else label)

        else:
            text = self._compact(value, max_len=250)
            if text:
                lines.append(text)

        return self._dedupe_lines(lines, max_items=4)

    def _format_notes(self, value: Any) -> list[str]:
        if isinstance(value, list):
            lines = [self._text(x) for x in value if self._text(x)]
            return self._dedupe_lines(lines, max_items=4)

        text = self._compact(value, max_len=350)
        return [text] if text else []

    def _format_citations_from_procedure(self, procedure_item: dict[str, Any]) -> list[str]:
        refs = self._extract_doc_refs(procedure_item.get("citations", []))
        if not refs:
            return []
        return [self._resolve_legal_title(ref, fallback=ref) for ref in refs[:4]]

    # ------------------------------------------------------------------
    # Section builders
    # ------------------------------------------------------------------

    def _build_procedure_section(self, query: str, results: list[SearchResult]) -> tuple[str, dict[str, Any] | None]:
        broad_legal_query = (
                "ho tich" in self._normalize_text(query)
                and any(x in self._normalize_text(query) for x in ["van ban", "quy dinh", "can cu", "phap ly"])
        )

        if broad_legal_query:
            return "Câu hỏi này thiên về căn cứ pháp lý chung, không gắn với một thủ tục cụ thể.", None

        if not results:
            return "Chưa tìm thấy thủ tục phù hợp.", None

        top = results[0].data
        title = top.get("title", top.get("id", ""))

        lines: list[str] = [f"Thủ tục phù hợp nhất: **{title}**."]

        authority_lines = self._format_authority(top.get("authority"))
        if authority_lines:
            lines.append("\n**Nơi giải quyết / thẩm quyền**")
            lines.extend(f"- {x}" for x in authority_lines)

        dossier_lines = self._format_dossier_items(top.get("raw", {}))
        if dossier_lines:
            lines.append("\n**Hồ sơ / giấy tờ cần chuẩn bị**")
            lines.extend(f"- {x}" for x in dossier_lines)

        processing_lines = self._format_processing_time(top.get("processing_time"))
        if processing_lines:
            lines.append("\n**Thời hạn giải quyết**")
            lines.extend(f"- {x}" for x in processing_lines)

        fee_lines = self._format_fees(top.get("fees"))
        if fee_lines:
            lines.append("\n**Lệ phí**")
            lines.extend(f"- {x}" for x in fee_lines)

        submission_lines = self._format_submission_methods(top.get("submission_methods"))
        if submission_lines:
            lines.append("\n**Cách thức nộp hồ sơ**")
            lines.extend(f"- {x}" for x in submission_lines)

        note_lines = self._format_notes(top.get("common_cases_notes"))
        if note_lines:
            lines.append("\n**Lưu ý thường gặp**")
            lines.extend(f"- {x}" for x in note_lines)

        related_titles: list[str] = []
        for r in results[1:]:
            if self._looks_related(query, r.title, title):
                related_titles.append(r.title)

        related_titles = self._dedupe_lines(related_titles, max_items=2)
        if related_titles:
            lines.append("\n**Thủ tục liên quan**")
            lines.extend(f"- {x}" for x in related_titles)

        return "\n".join(lines).strip(), top

    def _build_template_section(self, query: str, results: list[SearchResult], anchor_title: str = "") -> str:
        if self._is_broad_legal_query(query):
            filtered = self._filter_results_for_legal_query(
                query=query,
                results=results,
                anchor_title=anchor_title,
                max_items=2,
            )
            picked = [f"{r.title} ({r.item_id})" for r in filtered if r.title]
        else:
            picked: list[str] = []
            for r in results:
                if self._looks_related(query, r.title, anchor_title):
                    picked.append(f"{r.title} ({r.item_id})")

        picked = self._dedupe_lines(picked, max_items=2)
        if not picked:
            return ""

        lines = ["**Biểu mẫu liên quan**"]
        lines.extend(f"- {x}" for x in picked)
        return "\n".join(lines)

    def _build_case_section(self, query: str, results: list[SearchResult], anchor_title: str = "") -> str:
        if self._is_broad_legal_query(query):
            filtered = self._filter_results_for_legal_query(
                query=query,
                results=results,
                anchor_title=anchor_title,
                max_items=2,
            )
            if not filtered:
                return ""
            picked = [r.title for r in filtered if r.title]
        else:
            picked: list[str] = []
            for r in results:
                if self._looks_related(query, r.title, anchor_title):
                    picked.append(r.title)

        picked = self._dedupe_lines(picked, max_items=2)
        if not picked:
            return ""

        lines = ["**Tình huống gần giống**"]
        lines.extend(f"- {x}" for x in picked)
        return "\n".join(lines)

    def _build_legal_section(
        self,
        legal_results: list[SearchResult],
        procedure_item: dict[str, Any] | None = None,
    ) -> str:
        refs: list[str] = []

        if procedure_item is not None:
            refs.extend(self._extract_doc_refs(procedure_item.get("citations", [])))

        for r in legal_results[:3]:
            refs.append(r.item_id)

        uniq: list[str] = []
        seen = set()
        for ref in refs:
            if ref and ref not in seen:
                seen.add(ref)
                uniq.append(ref)

        if not uniq:
            return "Chưa xác định được căn cứ pháp lý nổi bật."

        lines = ["**Căn cứ pháp lý tham khảo**"]
        for ref in uniq[:4]:
            lines.append(f"- {self._resolve_legal_title(ref, fallback=ref)}")
        return "\n".join(lines)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def answer(self, query: str, per_kind: int = 3) -> AnswerResult:
        from app.core.router import HotichRouter

        router = HotichRouter()
        decision = router.route(query)

        procedure_k = per_kind
        template_k = per_kind
        legal_k = per_kind
        case_k = per_kind
        authority_k = per_kind

        if decision.primary_intent == "procedure":
            procedure_k = 5
            template_k = 3
            legal_k = 4
            case_k = 3
            authority_k = 3
        elif decision.primary_intent == "template":
            procedure_k = 3
            template_k = 5
            legal_k = 3
            case_k = 2
            authority_k = 2
        elif decision.primary_intent == "legal":
            procedure_k = 2
            template_k = 1
            legal_k = 8
            case_k = 2
            authority_k = 2
        elif decision.primary_intent == "case":
            procedure_k = 4
            template_k = 2
            legal_k = 3
            case_k = 5
            authority_k = 2

        groups = {
            "procedure": self.retriever.search(query, source_kinds=["procedure"], top_k=procedure_k, min_score=1.0),
            "template": self.retriever.search(query, source_kinds=["template"], top_k=template_k, min_score=1.0),
            "legal": self.retriever.search(query, source_kinds=["legal"], top_k=legal_k, min_score=1.0),
            "case": self.retriever.search(query, source_kinds=["case"], top_k=case_k, min_score=1.0),
            "authority": self.retriever.search(query, source_kinds=["authority"], top_k=authority_k, min_score=1.0),
        }

        # Tối ưu tập candidates bằng Hill Climbing nếu có
        if self.hc_reranker is not None:
            total_k = procedure_k + template_k + legal_k + case_k + authority_k
            groups = self.hc_reranker.optimize_grouped(
                query=query,
                grouped_results=groups,
                total_k=total_k,
            )

        procedure_text, top_procedure = self._build_procedure_section(
            query=query,
            results=groups.get("procedure", []),
        )

        anchor_title = top_procedure.get("title", "") if top_procedure else ""

        template_text = self._build_template_section(
            query=query,
            results=groups.get("template", []),
            anchor_title=anchor_title,
        )
        legal_text = self._build_legal_section(
            groups.get("legal", []),
            procedure_item=top_procedure,
        )
        case_text = self._build_case_section(
            query=query,
            results=groups.get("case", []),
            anchor_title=anchor_title,
        )

        answer_lines: list[str]

        if decision.primary_intent == "legal":
            answer_lines = [
                f"## Trả lời cho câu hỏi: {query}",
                "",
                f"_Intent chính: {decision.primary_intent}_",
                "",
                legal_text,
                "",
                procedure_text,
            ]

            if template_text:
                answer_lines.extend(["", template_text])

            if case_text:
                answer_lines.extend(["", case_text])
        else:
            answer_lines = [
                f"## Trả lời cho câu hỏi: {query}",
                "",
                f"_Intent chính: {decision.primary_intent}_",
                "",
                procedure_text,
                "",
                template_text,
                "",
                legal_text,
                "",
                case_text,
            ]

        return AnswerResult(
            query=query,
            answer_text="\n".join(answer_lines).strip(),
            grouped_results=groups,
        )

def main() -> None:
    loader = HotichLoader()
    bundle = loader.load_all()
    retriever = HotichHybridRetriever(bundle)
    builder = HotichAnswerBuilder(bundle, retriever)

    print("Nhap cau hoi. Go 'exit' de thoat.")
    while True:
        query = input("\n> ").strip()
        if not query:
            continue
        if query.lower() in {"exit", "quit"}:
            break

        result = builder.answer(query)
        print("\n" + "=" * 80)
        print(result.answer_text)
        print("=" * 80)


if __name__ == "__main__":
    main()