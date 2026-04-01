from __future__ import annotations

import re
import unicodedata
from typing import Any

from app.core.hybrid_retriever import SearchResult


EMPTY_GROUPS = {
    "procedure": [],
    "legal": [],
    "template": [],
    "case": [],
    "authority": [],
}


def text(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, str):
        return value.strip()
    if isinstance(value, (int, float, bool)):
        return str(value)
    return ""


def normalize_text(value: Any) -> str:
    s = text(value)
    if not s:
        return ""
    s = s.lower().strip()
    s = unicodedata.normalize("NFD", s)
    s = "".join(ch for ch in s if unicodedata.category(ch) != "Mn")
    s = s.replace("đ", "d").replace("Đ", "d")
    s = re.sub(r"[^a-z0-9\s]", " ", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s


def token_set(value: Any) -> set[str]:
    stopwords = {
        "la", "va", "cua", "cho", "ve", "o", "dau", "can", "gi", "giay", "to",
        "thu", "tuc", "dang", "ky", "khong", "co", "neu", "thi", "voi", "theo",
        "nao", "mau", "to", "khai", "form", "bieu", "mau", "bao", "lau", "phi",
    }
    return {
        tok
        for tok in normalize_text(value).split()
        if tok and tok not in stopwords and len(tok) >= 2
    }


def flatten(value: Any, limit: int = 1200) -> str:
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


def compact(value: Any, limit: int = 300) -> str:
    return " ".join(flatten(value, limit=limit).split())[:limit].strip()


def dedupe_lines(items: list[str], max_items: int = 8) -> list[str]:
    out: list[str] = []
    seen: set[str] = set()
    for item in items:
        clean = " ".join(text(item).split())
        key = normalize_text(clean)
        if clean and key not in seen:
            seen.add(key)
            out.append(clean)
        if len(out) >= max_items:
            break
    return out


def find_values_by_key_patterns(obj: Any, patterns: list[str], max_hits: int = 5) -> list[Any]:
    hits: list[Any] = []

    def walk(x: Any) -> None:
        if len(hits) >= max_hits:
            return
        if isinstance(x, dict):
            for k, v in x.items():
                nk = normalize_text(k).replace("_", " ")
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


def keyword_overlap_score(query: str, *values: Any) -> float:
    q = token_set(query)
    if not q:
        return 0.0

    total = 0.0
    for value in values:
        t = token_set(value)
        if not t:
            continue
        inter = q & t
        if not inter:
            continue
        ratio = len(inter) / max(len(q), 1)
        total += ratio * 10.0
        if q.issubset(t):
            total += 5.0
    return total


def make_empty_groups() -> dict[str, list[SearchResult]]:
    return {key: [] for key in EMPTY_GROUPS}


def make_search_result(
    *,
    kind: str,
    item_id: str,
    title: str,
    source_path: str,
    summary_text: str,
    score: float,
    best_unit_extra: dict[str, Any] | None = None,
    top_units: list[dict[str, Any]] | None = None,
    aggregated_extra: dict[str, Any] | None = None,
) -> SearchResult:
    best_unit = {
        "unit_id": item_id,
        "unit_kind": "record",
        "citation_label": title,
        "text": summary_text[:2800],
        "source_path": source_path,
    }
    if best_unit_extra:
        best_unit.update(best_unit_extra)

    if top_units is None:
        top_units = [
            {
                "unit_id": best_unit.get("unit_id", item_id),
                "unit_kind": best_unit.get("unit_kind", "record"),
                "citation_label": best_unit.get("citation_label", title),
                "score": score,
                "lexical_score": score,
                "semantic_score": 0.0,
                "text": best_unit.get("text", summary_text[:280]),
                "source_path": source_path,
                **({k: v for k, v in best_unit.items() if k not in {"text"}}),
            }
        ]

    data = {
        "record_id": item_id,
        "source_kind": kind,
        "title": title,
        "source_path": source_path,
        "top_units": top_units,
        "best_unit": best_unit,
    }
    if aggregated_extra:
        data.update(aggregated_extra)

    return SearchResult(
        kind=kind,
        item_id=item_id,
        title=title,
        score=score,
        source_path=source_path,
        snippet=summary_text[:220],
        data=data,
        lexical_score=score,
        semantic_score=0.0,
    )
