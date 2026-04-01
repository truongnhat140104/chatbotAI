from __future__ import annotations

import json
import math
import re
import unicodedata
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from sentence_transformers import SentenceTransformer

try:
    from pyvi import ViTokenizer
except ImportError:
    ViTokenizer = None


@dataclass
class SearchResult:
    kind: str
    item_id: str
    title: str
    score: float
    source_path: str
    snippet: str
    data: dict[str, Any]
    lexical_score: float = 0.0
    semantic_score: float = 0.0


class HotichHybridRetriever:
    """
    Hybrid retriever:
    - lexical / keyword retrieval
    - semantic retrieval từ embeddings.npy
    - fusion lexical + semantic
    """

    def __init__(
        self,
        index_dir: str | Path | None = None,
        embeddings_dir: str | Path | None = None,
        model_name: str = "bkai-foundation-models/vietnamese-bi-encoder",
        lexical_weight: float = 1.0,
        semantic_weight: float = 20.0,
        use_semantic: bool = True,
        use_word_segment: bool = True,
        device: str | None = None,
    ) -> None:
        self.index_dir = self._detect_index_dir(index_dir)
        self.embeddings_dir = self._detect_embeddings_dir(embeddings_dir)
        self.model_name = model_name
        self.lexical_weight = lexical_weight
        self.semantic_weight = semantic_weight
        self.use_semantic = use_semantic
        self.use_word_segment = use_word_segment
        self.device = device

        self.units = self._load_units(self.index_dir / "all_units.jsonl")
        self.units_by_kind = self._group_units_by_kind(self.units)

        self.embeddings: np.ndarray | None = None
        self.embedding_refs: list[dict[str, Any]] | None = None
        self.embedding_norms: np.ndarray | None = None
        self.model: SentenceTransformer | None = None

        if self.use_semantic:
            if self.use_word_segment and ViTokenizer is None:
                raise ImportError("Chua cai pyvi. Hay cai: pip install pyvi")

            self._load_semantic_artifacts()
            self._load_model()

    # ------------------------------------------------------------------
    # Path / load
    # ------------------------------------------------------------------

    def _detect_legal_query_mode(self, query: str) -> str:
        q = self.normalize_text(query)

        legal_definition_patterns = [
            "nguyen tac co ban",
            "noi dung cua",
            "la gi",
            "quyen va nghia vu",
            "quy dinh gi",
            "dieu kien ket hon",
            "che do hon nhan va gia dinh",
            "noi dung dieu",
        ]

        legal_basis_patterns = [
            "van ban nao",
            "can cu phap ly",
            "duoc quy dinh o dau",
            "theo quy dinh nao",
            "theo van ban nao",
        ]

        if any(p in q for p in legal_definition_patterns):
            return "legal_definition"
        if any(p in q for p in legal_basis_patterns):
            return "legal_basis"
        return ""

    def _detect_law_alias(self, query: str) -> str:
        q = self.normalize_text(query)

        if any(k in q for k in [
            "hon nhan va gia dinh",
            "luat hon nhan va gia dinh",
            "121 vbhn vpqh",
            "121/vbhn/vpqh",
        ]):
            return "luat_hon_nhan_va_gia_dinh"
        if any(k in q for k in [
            "luat ho tich",
            "60 2014 qh13",
            "60/2014/qh13",
        ]):
            return "luat_ho_tich"
        return ""

    def _extract_doc_hints(self, query: str) -> dict[str, str]:
        q = self.normalize_text(query)
        raw = (query or "").strip()

        doc_id_hint = ""
        doc_number_hint = ""

        m_doc_id_raw = re.search(r"\b(\d+_[A-Za-z0-9]+(?:_[A-Za-z0-9]+)+)\b", raw)
        if m_doc_id_raw:
            doc_id_hint = m_doc_id_raw.group(1).upper()

        m_doc_id_norm = re.search(r"\b(\d+\s+vbhn\s+vpqh)\b", q)
        if m_doc_id_norm and not doc_id_hint:
            doc_id_hint = m_doc_id_norm.group(1).upper().replace(" ", "_")

        m_doc_number = re.search(
            r"\b(\d+\s*/\s*[A-Za-z0-9]+\s*[-/]\s*[A-Za-z0-9]+)\b",
            raw,
            flags=re.IGNORECASE,
        )
        if m_doc_number:
            doc_number_hint = re.sub(r"\s+", "", m_doc_number.group(1)).upper()

        if doc_id_hint and not doc_number_hint:
            doc_number_hint = doc_id_hint.replace("_", "/", 1).replace("_", "-")

        return {
            "doc_id_hint": doc_id_hint,
            "doc_number_hint": doc_number_hint,
        }

    def _extract_article_lookup(self, query: str) -> dict[str, str]:
        q = self.normalize_text(query)

        m_article = re.search(r"\bdieu\s+(\d+)\b", q)
        article_no = m_article.group(1) if m_article else ""

        m_clause = re.search(r"\bkhoan\s+(\d+)\b", q)
        clause_no = m_clause.group(1) if m_clause else ""

        law_alias = self._detect_law_alias(query)
        doc_hints = self._extract_doc_hints(query)
        doc_id_hint = doc_hints["doc_id_hint"]
        doc_number_hint = doc_hints["doc_number_hint"]

        if article_no and (law_alias or doc_id_hint or doc_number_hint):
            return {
                "sub_intent": "legal_article_lookup",
                "article_no": article_no,
                "clause_no": clause_no,
                "law_alias": law_alias,
                "doc_id_hint": doc_id_hint,
                "doc_number_hint": doc_number_hint,
            }

        return {
            "sub_intent": "",
            "article_no": "",
            "clause_no": "",
            "law_alias": "",
            "doc_id_hint": "",
            "doc_number_hint": "",
        }

    def search_legal_article_lookup(
        self,
        query: str,
        law_alias: str,
        article_no: str,
        clause_no: str = "",
        top_k: int = 5,
        min_score: float = 0.5,
        doc_id_hint: str = "",
        doc_number_hint: str = "",
    ) -> list[SearchResult]:
        law_keywords = {
            "luat_hon_nhan_va_gia_dinh": [
                "hon nhan va gia dinh",
                "121 vbhn vpqh",
                "121/vbhn/vpqh",
                "luat hon nhan va gia dinh",
            ],
            "luat_ho_tich": [
                "luat ho tich",
                "60 2014 qh13",
                "60/2014/qh13",
            ],
        }

        keywords = law_keywords.get(law_alias, [])
        norm_doc_id_hint = self._text(doc_id_hint).upper()
        norm_doc_number_hint = self.normalize_text(doc_number_hint)

        unit_hits = self.search_units(
            query=query,
            source_kinds=["legal"],
            top_k=200,
            min_score=min_score,
        )

        filtered: list[SearchResult] = []

        for hit in unit_hits:
            u = hit.data
            doc_title = self.normalize_text(u.get("doc_title", ""))
            doc_number = self.normalize_text(u.get("doc_number", ""))
            hit_doc_id = self._text(u.get("doc_id")).upper()
            hit_article_no = self._text(u.get("article_no"))
            hit_clause_key = self._text(u.get("clause_key"))

            same_law = False
            if keywords and any(k in doc_title or k in doc_number for k in keywords):
                same_law = True
            if norm_doc_id_hint and hit_doc_id == norm_doc_id_hint:
                same_law = True
            if norm_doc_number_hint and doc_number == norm_doc_number_hint:
                same_law = True

            if not same_law or hit_article_no != str(article_no):
                continue

            boosted = SearchResult(
                kind=hit.kind,
                item_id=hit.item_id,
                title=hit.title,
                score=hit.score,
                source_path=hit.source_path,
                snippet=hit.snippet,
                data=hit.data,
                lexical_score=hit.lexical_score,
                semantic_score=hit.semantic_score,
            )
            boosted.score += 20.0
            if clause_no and hit_clause_key == str(clause_no):
                boosted.score += 10.0
            filtered.append(boosted)

        filtered.sort(
            key=lambda x: (-x.score, -x.semantic_score, -x.lexical_score, x.item_id)
        )
        return filtered[:top_k]

    def _detect_index_dir(self, explicit_dir: str | Path | None) -> Path:
        if explicit_dir is not None:
            p = Path(explicit_dir).resolve()
            if not p.exists():
                raise FileNotFoundError(f"Khong tim thay index_dir: {p}")
            return p

        here = Path(__file__).resolve()
        candidates = [
            here.parents[2] / "Data" / "hotich" / "05_index",
            Path.cwd() / "Data" / "hotich" / "05_index",
        ]

        for c in candidates:
            if c.exists():
                return c.resolve()

        raise FileNotFoundError(
            "Khong tim thay thu muc 05_index. "
            "Hay chay indexer truoc hoac truyen index_dir."
        )

    def _detect_embeddings_dir(self, explicit_dir: str | Path | None) -> Path:
        if explicit_dir is not None:
            p = Path(explicit_dir).resolve()
            if not p.exists():
                raise FileNotFoundError(f"Khong tim thay embeddings_dir: {p}")
            return p

        default_dir = self.index_dir / "embeddings"
        if default_dir.exists():
            return default_dir.resolve()

        raise FileNotFoundError(
            "Khong tim thay thu muc embeddings. "
            "Hay chay embedder truoc hoac truyen embeddings_dir."
        )

    @staticmethod
    def _load_units(path: Path) -> list[dict[str, Any]]:
        if not path.exists():
            raise FileNotFoundError(f"Khong tim thay file index: {path}")

        rows: list[dict[str, Any]] = []
        with path.open("r", encoding="utf-8") as f:
            for line_no, line in enumerate(f, start=1):
                line = line.strip()
                if not line:
                    continue
                try:
                    row = json.loads(line)
                except Exception as exc:
                    raise ValueError(f"Loi parse JSONL o dong {line_no}: {exc}") from exc

                if isinstance(row, dict):
                    rows.append(row)

        return rows

    @staticmethod
    def _read_jsonl(path: Path) -> list[dict[str, Any]]:
        rows: list[dict[str, Any]] = []
        with path.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    rows.append(json.loads(line))
        return rows

    @staticmethod
    def _group_units_by_kind(units: list[dict[str, Any]]) -> dict[str, list[dict[str, Any]]]:
        grouped: dict[str, list[dict[str, Any]]] = {}
        for unit in units:
            kind = str(unit.get("source_kind", "unknown"))
            grouped.setdefault(kind, []).append(unit)
        return grouped

    def _load_semantic_artifacts(self) -> None:
        emb_path = self.embeddings_dir / "embeddings.npy"
        refs_path = self.embeddings_dir / "unit_refs.jsonl"

        if not emb_path.exists():
            raise FileNotFoundError(f"Khong tim thay embeddings.npy: {emb_path}")
        if not refs_path.exists():
            raise FileNotFoundError(f"Khong tim thay unit_refs.jsonl: {refs_path}")

        self.embeddings = np.load(emb_path).astype(np.float32)
        self.embedding_refs = self._read_jsonl(refs_path)

        if len(self.units) != len(self.embedding_refs):
            raise ValueError(
                "So luong units trong all_units.jsonl va unit_refs.jsonl khong khop."
            )

        if self.embeddings.shape[0] != len(self.units):
            raise ValueError(
                "So dong embeddings.npy khong khop voi so units trong all_units.jsonl."
            )

        self.embedding_norms = np.linalg.norm(self.embeddings, axis=1)

    def _load_model(self) -> None:
        if self.device:
            self.model = SentenceTransformer(self.model_name, device=self.device)
        else:
            self.model = SentenceTransformer(self.model_name)

    # ------------------------------------------------------------------
    # Text utils
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

    @classmethod
    def normalize_text(cls, text: Any) -> str:
        text = cls._text(text)
        if not text:
            return ""

        text = text.strip().lower()
        text = unicodedata.normalize("NFD", text)
        text = "".join(ch for ch in text if unicodedata.category(ch) != "Mn")
        text = text.replace("đ", "d").replace("Đ", "d")
        text = re.sub(r"[^a-z0-9\s]", " ", text)
        text = re.sub(r"\s+", " ", text).strip()
        return text

    @classmethod
    def tokenize(cls, text: Any) -> list[str]:
        norm = cls.normalize_text(text)
        if not norm:
            return []
        return norm.split()

    @classmethod
    def content_tokens(cls, text: Any) -> list[str]:
        stopwords = {
            "va", "la", "cua", "cho", "tai", "theo", "duoc", "trong", "khi",
            "neu", "thi", "co", "khong", "mot", "nhung", "cac", "ve", "o",
            "tu", "den", "gi", "nao", "giay", "to", "thu", "tuc", "can", "lam",
            "sao", "hay", "voi", "de", "ngay", "mau", "don", "form", "ban",
            "ve", "cau", "hoi", "quy", "dinh",
        }
        toks = cls.tokenize(text)
        return [t for t in toks if len(t) >= 2 and t not in stopwords]

    @staticmethod
    def unique_preserve_order(items: list[str]) -> list[str]:
        seen = set()
        out: list[str] = []
        for item in items:
            if item not in seen:
                seen.add(item)
                out.append(item)
        return out

    def _segment_vi_text(self, text: str) -> str:
        text = text.strip()
        if not text:
            return ""
        if not self.use_word_segment:
            return text
        return ViTokenizer.tokenize(text)

    # ------------------------------------------------------------------
    # Query intent hints for scoring
    # ------------------------------------------------------------------

    def _kind_boost(self, query: str, unit: dict[str, Any]) -> float:
        q = self.normalize_text(query)
        kind = unit.get("source_kind", "")
        score = 0.0

        if kind == "template" and any(x in q for x in ["to khai", "mau", "bieu mau", "form"]):
            score += 3.5

        if kind == "legal" and any(
            x in q for x in ["van ban", "can cu", "phap ly", "luat", "nghi dinh", "thong tu", "vbhn", "dieu", "khoan"]
        ):
            score += 4.0

        if kind == "case" and any(
            x in q for x in ["truong hop", "tinh huong", "neu", "mat", "khong co", "uy quyen", "qua han", "bi bo roi"]
        ):
            score += 3.5

        if kind == "procedure" and any(
            x in q for x in ["dang ky", "thu tuc", "ho so", "giay to", "tham quyen", "thoi han", "le phi", "nop o dau"]
        ):
            score += 3.0

        if kind == "authority" and any(
            x in q for x in ["tham quyen", "co quan", "cap xa", "cap huyen", "giai quyet o dau"]
        ):
            score += 3.0

        return score

    # ------------------------------------------------------------------
    # Scoring
    # ------------------------------------------------------------------

    def _lexical_score(self, query: str, unit: dict[str, Any]) -> float:
        q_norm = self.normalize_text(query)
        if not q_norm:
            return 0.0

        title_norm = self.normalize_text(unit.get("title", ""))
        text_norm = self.normalize_text(unit.get("lexical_text", unit.get("text", "")))
        citation_norm = self.normalize_text(unit.get("citation_label", ""))
        combined = f"{title_norm} {citation_norm} {text_norm}".strip()

        q_tokens = self.unique_preserve_order(self.content_tokens(query))
        if not q_tokens:
            return 0.0

        score = 0.0

        if q_norm in title_norm:
            score += 12.0
        if q_norm in citation_norm:
            score += 10.0
        if q_norm in combined:
            score += 7.0

        joined = " ".join(q_tokens)
        if joined and joined in combined:
            score += 4.0

        matched_title = 0
        matched_body = 0

        for tok in q_tokens:
            if tok in title_norm:
                score += 4.0
                matched_title += 1
            if tok in citation_norm:
                score += 3.0
            if tok in combined:
                score += 1.6
                matched_body += 1

        coverage = matched_body / max(len(q_tokens), 1)
        score += coverage * 7.0

        if matched_title >= max(1, math.ceil(len(q_tokens) * 0.6)):
            score += 4.0

        score += self._kind_boost(query, unit)

        if unit.get("source_kind") == "legal":
            article_title_norm = self.normalize_text(unit.get("article_title", ""))
            doc_title_norm = self.normalize_text(unit.get("doc_title", ""))

            important_phrases = [q_norm]
            if len(q_tokens) >= 3:
                important_phrases.append(" ".join(q_tokens[:3]))
            if len(q_tokens) >= 2:
                important_phrases.append(" ".join(q_tokens[:2]))

            for ph in important_phrases:
                if not ph:
                    continue
                if ph in article_title_norm:
                    score += 10.0
                if ph in doc_title_norm:
                    score += 6.0
                if ph in combined:
                    score += 5.0

        return score

    def _encode_query(self, query: str) -> np.ndarray:
        if self.model is None:
            raise RuntimeError("Semantic model chua duoc load.")

        prepared_query = self._segment_vi_text(query)
        vec = self.model.encode(
            [prepared_query],
            convert_to_numpy=True,
            normalize_embeddings=True,
            show_progress_bar=False,
        )
        return vec[0].astype(np.float32)

    def _semantic_scores(self, query: str) -> np.ndarray | None:
        if not self.use_semantic:
            return None
        if self.embeddings is None:
            return None

        query_vec = self._encode_query(query)

        # embeddings đã normalize khi build => dot product = cosine similarity
        scores = self.embeddings @ query_vec
        return scores.astype(np.float32)

    def _fuse_score(self, lexical_score: float, semantic_score: float) -> float:
        return self.lexical_weight * lexical_score + self.semantic_weight * semantic_score

    # ------------------------------------------------------------------
    # Unit-level search
    # ------------------------------------------------------------------

    def search_units(
            self,
            query: str,
            source_kinds: list[str] | None = None,
            kinds: list[str] | None = None,
            top_k: int = 10,
            min_score: float = 1.0,
    ) -> list[SearchResult]:
        if source_kinds is None and kinds is not None:
            source_kinds = kinds
        allowed = set(source_kinds) if source_kinds else None
        results: list[SearchResult] = []

        semantic_scores = self._semantic_scores(query)

        for idx, unit in enumerate(self.units):
            kind = str(unit.get("source_kind", "unknown"))
            if allowed is not None and kind not in allowed:
                continue

            lexical_score = self._lexical_score(query, unit)
            semantic_score = float(semantic_scores[idx]) if semantic_scores is not None else 0.0
            final_score = self._fuse_score(lexical_score, semantic_score)

            if final_score < min_score:
                continue

            snippet = self._text(unit.get("citation_label")) or self._text(unit.get("text"))[:220]

            results.append(
                SearchResult(
                    kind=kind,
                    item_id=self._text(unit.get("unit_id")),
                    title=self._text(unit.get("title")) or self._text(unit.get("doc_title")) or self._text(unit.get("unit_id")),
                    score=final_score,
                    source_path=self._text(unit.get("source_path")),
                    snippet=snippet,
                    data=unit,
                    lexical_score=lexical_score,
                    semantic_score=semantic_score,
                )
            )

        results.sort(key=lambda x: (-x.score, -x.semantic_score, -x.lexical_score, x.kind, x.title, x.item_id))
        return results[:top_k]

    # ------------------------------------------------------------------
    # Record-level aggregation
    # ------------------------------------------------------------------

    def _record_key(self, unit: dict[str, Any]) -> str:
        kind = self._text(unit.get("source_kind"))

        if kind == "legal":
            return self._text(unit.get("doc_id")) or self._text(unit.get("unit_id"))
        if kind == "procedure":
            return self._text(unit.get("procedure_id")) or self._text(unit.get("unit_id"))
        if kind == "template":
            return self._text(unit.get("template_id")) or self._text(unit.get("unit_id"))
        if kind == "case":
            return self._text(unit.get("case_id")) or self._text(unit.get("unit_id"))
        if kind == "authority":
            return self._text(unit.get("authority_id")) or self._text(unit.get("unit_id"))

        return self._text(unit.get("unit_id"))

    def _record_title(self, unit: dict[str, Any]) -> str:
        kind = self._text(unit.get("source_kind"))
        if kind == "legal":
            return self._text(unit.get("doc_title")) or self._text(unit.get("title")) or self._record_key(unit)
        return self._text(unit.get("title")) or self._record_key(unit)

    def _record_snippet(self, best_unit: dict[str, Any]) -> str:
        citation = self._text(best_unit.get("citation_label"))
        text = self._text(best_unit.get("text"))
        if citation and text:
            return f"{citation} | {text[:220]}"
        return citation or text[:220]

    def search(
            self,
            query: str,
            source_kinds: list[str] | None = None,
            kinds: list[str] | None = None,
            top_k: int = 10,
            min_score: float = 1.0,
            per_record_keep_units: int = 3,
    ) -> list[SearchResult]:
        if source_kinds is None and kinds is not None:
            source_kinds = kinds

        # Quan trọng: nhánh tra cứu điều/khoản luật phải chạy cho cả hai cách gọi
        # search(query, source_kinds=["legal"]) và search(query, kinds=["legal"]).
        # Bản cũ đặt logic này trong nhánh source_kinds is None nên AnswerBuilder
        # (đang truyền source_kinds) sẽ không bao giờ đi vào article lookup.
        if source_kinds is not None and set(source_kinds) == {"legal"}:
            article_lookup = self._extract_article_lookup(query)
            if article_lookup["sub_intent"] == "legal_article_lookup":
                return self.search_legal_article_lookup(
                    query=query,
                    law_alias=article_lookup["law_alias"],
                    article_no=article_lookup["article_no"],
                    clause_no=article_lookup["clause_no"],
                    top_k=top_k,
                    min_score=min_score,
                    doc_id_hint=article_lookup.get("doc_id_hint", ""),
                    doc_number_hint=article_lookup.get("doc_number_hint", ""),
                )

        unit_hits = self.search_units(
            query=query,
            source_kinds=source_kinds,
            top_k=max(top_k * 20, 60),
            min_score=min_score,
        )

        grouped: dict[tuple[str, str], list[SearchResult]] = {}
        for hit in unit_hits:
            unit = hit.data
            record_key = self._record_key(unit)
            group_key = (hit.kind, record_key)
            grouped.setdefault(group_key, []).append(hit)

        results: list[SearchResult] = []

        for (kind, record_id), hits in grouped.items():
            hits.sort(key=lambda x: (-x.score, -x.semantic_score, -x.lexical_score, x.item_id))
            best = hits[0]

            best_score = best.score
            extra_bonus = sum(h.score for h in hits[1:per_record_keep_units]) * 0.12
            final_score = best_score + extra_bonus

            top_units = []
            for h in hits[:per_record_keep_units]:
                u = h.data
                top_units.append(
                    {
                        "unit_id": u.get("unit_id"),
                        "unit_kind": u.get("unit_kind"),
                        "citation_label": u.get("citation_label"),
                        "score": h.score,
                        "lexical_score": h.lexical_score,
                        "semantic_score": h.semantic_score,
                        "text": self._text(u.get("text"))[:280],
                        "doc_id": u.get("doc_id"),
                        "doc_number": u.get("doc_number"),
                        "article_no": u.get("article_no"),
                        "article_title": u.get("article_title"),
                        "clause_key": u.get("clause_key"),
                        "source_path": u.get("source_path"),
                    }
                )

            best_unit = best.data

            aggregated_data = {
                "record_id": record_id,
                "source_kind": kind,
                "title": self._record_title(best_unit),
                "doc_id": best_unit.get("doc_id"),
                "doc_title": best_unit.get("doc_title"),
                "doc_number": best_unit.get("doc_number"),
                "citation_label": best_unit.get("citation_label"),
                "source_path": best_unit.get("source_path"),
                "top_units": top_units,
                "best_unit": best_unit,
            }

            results.append(
                SearchResult(
                    kind=kind,
                    item_id=record_id,
                    title=self._record_title(best_unit),
                    score=final_score,
                    source_path=self._text(best_unit.get("source_path")),
                    snippet=self._record_snippet(best_unit),
                    data=aggregated_data,
                    lexical_score=best.lexical_score,
                    semantic_score=best.semantic_score,
                )
            )

        results.sort(key=lambda x: (-x.score, -x.semantic_score, -x.lexical_score, x.kind, x.title, x.item_id))
        return results[:top_k]

    def grouped_search(self, query: str, per_kind: int = 3) -> dict[str, list[SearchResult]]:
        groups: dict[str, list[SearchResult]] = {}
        for kind in ["procedure", "legal", "template", "case", "authority"]:
            groups[kind] = self.search(
                query=query,
                source_kinds=[kind],
                top_k=per_kind,
                min_score=1.0,
            )
        return groups


def print_grouped_results(groups: dict[str, list[SearchResult]]) -> None:
    labels = {
        "procedure": "PROCEDURES",
        "legal": "LEGAL",
        "template": "TEMPLATES",
        "case": "CASES",
        "authority": "AUTHORITY",
    }

    for kind in ["procedure", "legal", "template", "case", "authority"]:
        print("\n" + "=" * 80)
        print(labels[kind])
        print("=" * 80)

        rows = groups.get(kind, [])
        if not rows:
            print("Khong co ket qua phu hop.")
            continue

        for idx, r in enumerate(rows, start=1):
            print(f"{idx}. [{r.kind}] {r.item_id}")
            print(f"   Title         : {r.title}")
            print(f"   Score         : {r.score:.2f}")
            print(f"   Lexical       : {r.lexical_score:.2f}")
            print(f"   Semantic      : {r.semantic_score:.4f}")
            print(f"   Snippet       : {r.snippet}")
            print(f"   Source path   : {r.source_path}")

            top_units = r.data.get("top_units", [])
            if top_units:
                print("   Top units:")
                for u in top_units[:3]:
                    print(
                        f"     - {u.get('citation_label') or u.get('unit_id')} "
                        f"(score={u.get('score', 0):.2f}, "
                        f"lex={u.get('lexical_score', 0):.2f}, "
                        f"sem={u.get('semantic_score', 0):.4f})"
                    )


def main() -> None:
    retriever = HotichHybridRetriever(
        model_name="bkai-foundation-models/vietnamese-bi-encoder",
        use_semantic=True,
        lexical_weight=1.0,
        semantic_weight=20.0,
    )

    print("Nhap cau hoi de test hybrid retriever. Go 'exit' de thoat.")
    while True:
        query = input("\n> ").strip()
        if not query:
            continue
        if query.lower() in {"exit", "quit"}:
            break

        groups = retriever.grouped_search(query=query, per_kind=3)
        print_grouped_results(groups)


if __name__ == "__main__":
    main()