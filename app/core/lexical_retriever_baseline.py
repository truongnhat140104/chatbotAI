from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass
from typing import Any, Iterable

from app.core.loader import HotichBundle, HotichLoader


@dataclass
class SearchResult:
    kind: str
    item_id: str
    title: str
    score: float
    source_path: str
    snippet: str
    data: dict[str, Any]


class HotichRetriever:
    def __init__(self, bundle: HotichBundle) -> None:
        self.bundle = bundle

    # ------------------------------------------------------------------
    # Text utils
    # ------------------------------------------------------------------

    @staticmethod
    def normalize_text(text: Any) -> str:
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
    def tokenize(cls, text: Any) -> list[str]:
        normalized = cls.normalize_text(text)
        if not normalized:
            return []
        return normalized.split()

    @staticmethod
    def unique_preserve_order(items: Iterable[str]) -> list[str]:
        seen = set()
        out = []
        for item in items:
            if item not in seen:
                seen.add(item)
                out.append(item)
        return out

    @staticmethod
    def flatten_text(value: Any, limit: int = 4000) -> str:
        """
        Rút text từ dict/list lồng nhau để search.
        Không cần lấy hết mọi thứ, chỉ cần đủ cho keyword search.
        """
        parts: list[str] = []

        def walk(x: Any) -> None:
            if len(" ".join(parts)) > limit:
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
                for _, v in x.items():
                    walk(v)

        walk(value)
        text = " ".join(parts)
        return text[:limit]

    # ------------------------------------------------------------------
    # Build search text per kind
    # ------------------------------------------------------------------

    def _procedure_text(self, item: dict[str, Any]) -> tuple[str, str]:
        parts = [
            item.get("id", ""),
            item.get("title", ""),
            self.flatten_text(item.get("authority", {}), limit=700),
            self.flatten_text(item.get("processing_time", ""), limit=300),
            self.flatten_text(item.get("fees", ""), limit=300),
            self.flatten_text(item.get("submission_methods", []), limit=500),
            self.flatten_text(item.get("citations", []), limit=700),
            " ".join(item.get("tags", []) if isinstance(item.get("tags"), list) else []),
            self.flatten_text(item.get("variants", []), limit=700),
            self.flatten_text(item.get("common_cases_notes", ""), limit=500),
        ]
        full_text = " ".join(p for p in parts if p).strip()
        snippet = item.get("title", "")
        return full_text, snippet

    def _legal_text(self, item: dict[str, Any]) -> tuple[str, str]:
        parts = [
            item.get("id", ""),
            item.get("title", ""),
            item.get("doc_type", ""),
            item.get("parse_mode", ""),
            self.flatten_text(item.get("notes", ""), limit=800),
            self.flatten_text(item.get("chapters", []), limit=2500),
        ]
        full_text = " ".join(p for p in parts if p).strip()
        snippet = item.get("title", "") or item.get("id", "")
        return full_text, snippet

    def _template_text(self, item: dict[str, Any]) -> tuple[str, str]:
        parts = [
            item.get("id", ""),
            item.get("title", ""),
            item.get("loai", ""),
            item.get("output_type", ""),
            self.flatten_text(item.get("fields", []), limit=1200),
            self.flatten_text(item.get("render", {}), limit=800),
            self.flatten_text(item.get("notes", ""), limit=400),
        ]
        full_text = " ".join(p for p in parts if p).strip()
        snippet = item.get("title", "")
        return full_text, snippet

    def _case_text(self, item: dict[str, Any]) -> tuple[str, str]:
        parts = [
            item.get("id", ""),
            item.get("title", ""),
            " ".join(item.get("topic_tags", []) if isinstance(item.get("topic_tags"), list) else []),
            self.flatten_text(item.get("assumptions", ""), limit=500),
            self.flatten_text(item.get("inputs_schema", {}), limit=700),
            self.flatten_text(item.get("legal_basis", []), limit=900),
            self.flatten_text(item.get("decision_points", []), limit=700),
            self.flatten_text(item.get("steps", []), limit=1200),
            self.flatten_text(item.get("outputs", {}), limit=700),
        ]
        full_text = " ".join(p for p in parts if p).strip()
        snippet = item.get("title", "")
        return full_text, snippet

    # ------------------------------------------------------------------
    # Scoring
    # ------------------------------------------------------------------

    def score_match(self, query: str, title: str, body: str) -> float:
        q_tokens = self.unique_preserve_order(self.tokenize(query))
        if not q_tokens:
            return 0.0

        title_norm = self.normalize_text(title)
        body_norm = self.normalize_text(body)

        score = 0.0

        # phrase bonus
        q_norm = self.normalize_text(query)
        if q_norm and q_norm in body_norm:
            score += 8.0
        if q_norm and q_norm in title_norm:
            score += 12.0

        # token scoring
        for tok in q_tokens:
            if tok in title_norm:
                score += 4.0
            if tok in body_norm:
                score += 1.5

        # coverage bonus
        matched = sum(1 for tok in q_tokens if tok in body_norm)
        if matched:
            score += (matched / max(len(q_tokens), 1)) * 6.0

        return score

    def _collect_candidates(self, kinds: list[str] | None = None) -> list[SearchResult]:
        allowed = set(kinds) if kinds else {"procedure", "legal", "template", "case"}
        out: list[SearchResult] = []

        if "procedure" in allowed:
            for item in self.bundle.procedures.values():
                text, snippet = self._procedure_text(item)
                out.append(
                    SearchResult(
                        kind="procedure",
                        item_id=item["id"],
                        title=item.get("title", ""),
                        score=0.0,
                        source_path=item.get("source_path", ""),
                        snippet=snippet,
                        data={**item, "_search_text": text},
                    )
                )

        if "legal" in allowed:
            for item in self.bundle.legal_docs.values():
                text, snippet = self._legal_text(item)
                out.append(
                    SearchResult(
                        kind="legal",
                        item_id=item["id"],
                        title=item.get("title", ""),
                        score=0.0,
                        source_path=item.get("source_path", ""),
                        snippet=snippet,
                        data={**item, "_search_text": text},
                    )
                )

        if "template" in allowed:
            for item in self.bundle.templates.values():
                text, snippet = self._template_text(item)
                out.append(
                    SearchResult(
                        kind="template",
                        item_id=item["id"],
                        title=item.get("title", ""),
                        score=0.0,
                        source_path=item.get("source_path", ""),
                        snippet=snippet,
                        data={**item, "_search_text": text},
                    )
                )

        if "case" in allowed:
            for item in self.bundle.cases.values():
                text, snippet = self._case_text(item)
                out.append(
                    SearchResult(
                        kind="case",
                        item_id=item["id"],
                        title=item.get("title", ""),
                        score=0.0,
                        source_path=item.get("source_path", ""),
                        snippet=snippet,
                        data={**item, "_search_text": text},
                    )
                )

        return out

    # ------------------------------------------------------------------
    # Public search API
    # ------------------------------------------------------------------

    def search(
        self,
        query: str,
        kinds: list[str] | None = None,
        top_k: int = 10,
        min_score: float = 1.0,
    ) -> list[SearchResult]:
        candidates = self._collect_candidates(kinds=kinds)
        results: list[SearchResult] = []

        for cand in candidates:
            score = self.score_match(
                query=query,
                title=cand.title,
                body=cand.data.get("_search_text", ""),
            )
            if score >= min_score:
                cand.score = score
                results.append(cand)

        results.sort(key=lambda x: (-x.score, x.kind, x.title))
        return results[:top_k]

    def grouped_search(
        self,
        query: str,
        per_kind: int = 3,
    ) -> dict[str, list[SearchResult]]:
        groups: dict[str, list[SearchResult]] = {}
        for kind in ["procedure", "legal", "template", "case"]:
            groups[kind] = self.search(
                query=query,
                kinds=[kind],
                top_k=per_kind,
                min_score=1.0,
            )
        return groups


def print_grouped_results(groups: dict[str, list[SearchResult]]) -> None:
    labels = {
        "procedure": "PROCEDURES",
        "legal": "LEGAL DOCS",
        "template": "TEMPLATES",
        "case": "CASES",
    }

    for kind in ["procedure", "legal", "template", "case"]:
        print("\n" + "=" * 80)
        print(labels[kind])
        print("=" * 80)

        results = groups.get(kind, [])
        if not results:
            print("Khong co ket qua phu hop.")
            continue

        for idx, r in enumerate(results, start=1):
            print(f"{idx}. [{r.kind}] {r.item_id}")
            print(f"   Tieu de : {r.title}")
            print(f"   Score   : {r.score:.2f}")
            print(f"   File    : {r.source_path}")
            print(f"   Snippet : {r.snippet}")


def main() -> None:
    loader = HotichLoader()
    bundle = loader.load_all()

    retriever = HotichRetriever(bundle)

    print("Nhap cau hoi de tim kiem. Goi 'exit' de thoat.")
    while True:
        query = input("\n> ").strip()
        if not query:
            continue
        if query.lower() in {"exit", "quit"}:
            break

        groups = retriever.grouped_search(query, per_kind=3)
        print_grouped_results(groups)


if __name__ == "__main__":
    main()