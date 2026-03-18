from __future__ import annotations

import json
import math
import re
import unicodedata
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np

from app.core.hybrid_retriever import SearchResult
from app.core.loader import HotichBundle
from app.core.router import HotichRouter


@dataclass
class HillClimbingConfig:
    k: int = 10                  # kích thước trạng thái S (Top-k để tối ưu)
    m: int = 6                   # chỉ tối ưu mạnh trên m vị trí đầu dùng làm context
    max_iter: int = 25
    lambda_redundancy: float = 0.80
    eta_constraint: float = 2.00
    gamma_coverage: float = 0.50
    relevance_weight: float = 1.00
    time_weight: float = 0.15
    level_weight: float = 0.10


@dataclass
class HCRerankResult:
    selected: list[SearchResult]
    objective_score: float
    iterations: int


class HotichHillClimbingReranker:
    """
    Steepest-Ascent Hill Climbing re-ranker cho Top-k candidates.

    Ý tưởng bám theo đồ án:
    - S0 = Top-k từ retriever
    - Tiền tính Sim[u,v]
    - Mỗi vòng lặp: xét mọi swap(i, j), chọn láng giềng tốt nhất
    - Dừng khi không còn cải thiện hoặc đạt max_iter
    """

    def __init__(
        self,
        bundle: HotichBundle,
        embeddings_dir: str | Path | None = None,
        config: HillClimbingConfig | None = None,
    ) -> None:
        self.bundle = bundle
        self.config = config or HillClimbingConfig()
        self.router = HotichRouter()

        self.embeddings_dir = self._detect_embeddings_dir(embeddings_dir)
        self.unit_id_to_vec_index = self._load_unit_ref_index(self.embeddings_dir / "unit_refs.jsonl")
        self.embeddings = np.load(self.embeddings_dir / "embeddings.npy").astype(np.float32)

    # ------------------------------------------------------------------
    # Paths / load
    # ------------------------------------------------------------------

    def _detect_embeddings_dir(self, explicit_dir: str | Path | None) -> Path:
        if explicit_dir is not None:
            p = Path(explicit_dir).resolve()
            if not p.exists():
                raise FileNotFoundError(f"Khong tim thay embeddings_dir: {p}")
            return p

        default_dir = (self.bundle.data_root / "05_index" / "embeddings").resolve()
        if default_dir.exists():
            return default_dir

        raise FileNotFoundError(
            "Khong tim thay thu muc embeddings. Hay chay embedder truoc."
        )

    @staticmethod
    def _load_unit_ref_index(path: Path) -> dict[str, int]:
        if not path.exists():
            raise FileNotFoundError(f"Khong tim thay unit_refs.jsonl: {path}")

        out: dict[str, int] = {}
        with path.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                row = json.loads(line)
                unit_id = row.get("unit_id")
                row_index = row.get("row_index")
                if isinstance(unit_id, str) and isinstance(row_index, int):
                    out[unit_id] = row_index
        return out

    # ------------------------------------------------------------------
    # Text / normalize
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
    def _normalize_text(cls, text: Any) -> str:
        text = cls._text(text)
        if not text:
            return ""
        text = text.lower().strip()
        text = unicodedata.normalize("NFD", text)
        text = "".join(ch for ch in text if unicodedata.category(ch) != "Mn")
        text = text.replace("đ", "d").replace("Đ", "d")
        text = re.sub(r"[^a-z0-9\s]", " ", text)
        text = re.sub(r"\s+", " ", text).strip()
        return text

    @classmethod
    def _token_set(cls, text: Any) -> set[str]:
        return set(cls._normalize_text(text).split())

    # ------------------------------------------------------------------
    # Candidate metadata
    # ------------------------------------------------------------------

    def _doc_id(self, result: SearchResult) -> str:
        data = result.data or {}
        if isinstance(data, dict):
            if data.get("doc_id"):
                return self._text(data.get("doc_id"))
            best_unit = data.get("best_unit")
            if isinstance(best_unit, dict):
                if best_unit.get("doc_id"):
                    return self._text(best_unit.get("doc_id"))
        return ""

    def _best_unit_id(self, result: SearchResult) -> str:
        data = result.data or {}
        if isinstance(data, dict):
            best_unit = data.get("best_unit")
            if isinstance(best_unit, dict) and best_unit.get("unit_id"):
                return self._text(best_unit.get("unit_id"))
        return ""

    def _record_level(self, result: SearchResult) -> float:
        """
        Level(d): ưu tiên nguồn sát nghiệp vụ cấp xã.
        Đây là phiên bản triển khai hóa từ ý tưởng trong đồ án:
        - procedure / authority thường hữu ích hơn cho tác nghiệp trực tiếp
        - legal vẫn quan trọng nhưng không phải lúc nào cũng nên chiếm hết context
        """
        kind = result.kind

        if kind == "procedure":
            proc = self.bundle.procedures.get(result.item_id)
            if proc:
                authority = proc.get("authority", {})
                if isinstance(authority, dict):
                    lv = self._text(authority.get("resolve_level")).lower()
                    if lv == "xa":
                        return 1.00
                    if lv == "huyen":
                        return 0.80
                    if lv == "tinh":
                        return 0.60
            return 0.80

        if kind == "authority":
            return 1.00

        if kind == "template":
            return 0.85

        if kind == "case":
            return 0.80

        if kind == "legal":
            doc_id = self._doc_id(result)
            meta = self.bundle.meta.get(doc_id, {})
            doc = meta.get("doc", {}) if isinstance(meta, dict) else {}
            if isinstance(doc, dict):
                doc_kind = self._text(doc.get("doc_kind")).lower()
                if "vbhn" in doc_kind:
                    return 0.90
                if "thong_tu" in doc_kind:
                    return 0.80
                if "nghi_dinh" in doc_kind:
                    return 0.78
                if "luat" in doc_kind:
                    return 0.75
            return 0.75

        return 0.50

    def _time_score(self, result: SearchResult) -> float:
        """
        Time(d): ưu tiên văn bản còn hiệu lực / mới hơn.
        Nếu không có metadata đủ rõ, trả về điểm trung tính.
        """
        doc_id = self._doc_id(result)
        if not doc_id:
            return 0.50

        meta = self.bundle.meta.get(doc_id, {})
        doc = meta.get("doc", {}) if isinstance(meta, dict) else {}
        if not isinstance(doc, dict):
            return 0.50

        status = self._text(doc.get("status")).lower()
        if status == "active":
            status_bonus = 0.20
        elif status in {"expired", "inactive"}:
            status_bonus = -0.20
        else:
            status_bonus = 0.0

        candidates = [
            self._text(doc.get("effective_date")),
            self._text(doc.get("issued_date")),
        ]

        parsed_dates: list[datetime] = []
        for s in candidates:
            if not s:
                continue
            try:
                parsed_dates.append(datetime.fromisoformat(s))
            except Exception:
                pass

        if not parsed_dates:
            return 0.50 + status_bonus

        newest = max(parsed_dates)
        base = newest.year / 3000.0
        return min(max(base + status_bonus, 0.0), 1.2)

    def _base_doc_score(self, result: SearchResult, rank_pos: int) -> float:
        """
        f(d) ~ Score(d) + Time(d) + Level(d)
        và dùng discount kiểu DCG theo vị trí.
        """
        cfg = self.config

        rel = result.score
        time_s = self._time_score(result)
        level_s = self._record_level(result)

        raw = (
            cfg.relevance_weight * rel
            + cfg.time_weight * time_s
            + cfg.level_weight * level_s
        )

        discount = 1.0 / math.log2(rank_pos + 2)  # rank_pos bắt đầu từ 0
        return raw * discount

    # ------------------------------------------------------------------
    # Similarity / redundancy
    # ------------------------------------------------------------------

    def _pair_similarity(self, a: SearchResult, b: SearchResult) -> float:
        """
        Ưu tiên dùng cosine similarity từ embedding của best_unit.
        Fallback: Jaccard token overlap trên title/snippet.
        """
        ua = self._best_unit_id(a)
        ub = self._best_unit_id(b)

        if ua in self.unit_id_to_vec_index and ub in self.unit_id_to_vec_index:
            ia = self.unit_id_to_vec_index[ua]
            ib = self.unit_id_to_vec_index[ub]
            va = self.embeddings[ia]
            vb = self.embeddings[ib]
            # embeddings đã normalize ở embedder => dot = cosine
            sim = float(np.dot(va, vb))
            return max(min(sim, 1.0), -1.0)

        ta = self._token_set(f"{a.title} {a.snippet}")
        tb = self._token_set(f"{b.title} {b.snippet}")
        if not ta or not tb:
            return 0.0
        return len(ta & tb) / max(len(ta | tb), 1)

    def _redundancy_penalty(self, state: list[SearchResult], m: int) -> float:
        top = state[:m]
        if len(top) <= 1:
            return 0.0

        penalty = 0.0
        for i in range(len(top)):
            for j in range(i + 1, len(top)):
                sim = self._pair_similarity(top[i], top[j])

                # phạt mạnh hơn nếu cùng doc_id
                same_doc = self._doc_id(top[i]) and self._doc_id(top[i]) == self._doc_id(top[j])
                if same_doc:
                    penalty += 0.75 + max(sim, 0.0)
                else:
                    # chỉ phạt nếu thật sự khá giống
                    if sim >= 0.55:
                        penalty += sim
        return penalty

    # ------------------------------------------------------------------
    # Constraint penalties / coverage
    # ------------------------------------------------------------------

    def _constraint_penalty(
        self,
        query: str,
        state: list[SearchResult],
        m: int,
    ) -> float:
        top = state[:m]
        kinds = [x.kind for x in top]
        q = self._normalize_text(query)
        intent = self.router.route(query).primary_intent

        penalty = 0.0

        # thiếu loại tài liệu chính theo intent
        if intent == "legal" and "legal" not in kinds:
            penalty += 2.0
        if intent == "template" and "template" not in kinds:
            penalty += 2.0
        if intent == "case" and "case" not in kinds:
            penalty += 2.0
        if intent == "procedure" and "procedure" not in kinds:
            penalty += 2.0

        # broad legal query: không nên toàn procedure
        if "ho tich" in q and any(x in q for x in ["van ban", "can cu", "quy dinh", "phap ly"]):
            legal_count = sum(1 for k in kinds if k == "legal")
            proc_count = sum(1 for k in kinds if k == "procedure")
            template_count = sum(1 for k in kinds if k == "template")

            if legal_count == 0:
                penalty += 4.0
            if legal_count < 2:
                penalty += 2.0
            if proc_count > legal_count:
                penalty += 3.0
            if template_count > 1:
                penalty += 1.5

        # query về hồ sơ/giấy tờ mà không có procedure hoặc template
        if any(x in q for x in ["ho so", "giay to", "to khai", "mau", "bieu mau"]):
            if "procedure" not in kinds and "template" not in kinds:
                penalty += 2.0

        # query về tình huống mà không có case
        if any(x in q for x in ["truong hop", "neu", "mat", "khong co", "uy quyen", "qua han", "bi bo roi"]):
            if "case" not in kinds:
                penalty += 1.5

        # Query legal về lệ phí / miễn lệ phí: ưu tiên legal có nội dung sát phí/lệ phí
        if any(x in q for x in ["mien le phi", "le phi", "mien giam", "muc thu", "phi"]):
            legal_hits = 0
            for item in top:
                if item.kind != "legal":
                    continue

                candidate_text = self._normalize_text(f"{item.title} {item.snippet}")
                if any(x in candidate_text for x in ["le phi", "mien le phi", "muc thu", "phi", "mien giam"]):
                    legal_hits += 1

            if legal_hits == 0:
                penalty += 4.0
            elif legal_hits == 1:
                penalty += 1.5

        return penalty

    def _coverage_bonus(self, state: list[SearchResult], m: int) -> float:
        top = state[:m]
        kinds = {x.kind for x in top}
        # thưởng nhẹ cho độ đa dạng nguồn/kinds, nhưng không quá mạnh
        return min(len(kinds), 4) * 0.5

    # ------------------------------------------------------------------
    # Objective G(S)
    # ------------------------------------------------------------------

    def objective(self, query: str, state: list[SearchResult]) -> float:
        cfg = self.config
        m = min(cfg.m, len(state))

        # F(S_top): nền DCG
        base = 0.0
        for idx, result in enumerate(state[:m]):
            base += self._base_doc_score(result, idx)

        red = self._redundancy_penalty(state, m)
        cons = self._constraint_penalty(query, state, m)
        cov = self._coverage_bonus(state, m)

        g = base - cfg.lambda_redundancy * red - cfg.eta_constraint * cons + cfg.gamma_coverage * cov
        return g

    # ------------------------------------------------------------------
    # Hill climbing
    # ------------------------------------------------------------------

    def _swap(self, arr: list[SearchResult], i: int, j: int) -> list[SearchResult]:
        out = arr[:]
        out[i], out[j] = out[j], out[i]
        return out

    def rerank(
        self,
        query: str,
        candidates: list[SearchResult],
        k: int | None = None,
    ) -> HCRerankResult:
        """
        Input nên là danh sách Top-N từ retriever, đã sort theo score.
        Theo đồ án, trạng thái khởi tạo S0 chính là Top-k từ Vector DB / retriever. :contentReference[oaicite:2]{index=2}
        """
        if not candidates:
            return HCRerankResult(selected=[], objective_score=0.0, iterations=0)

        cfg = self.config
        k = k or cfg.k

        # S0 = Top-k candidates đầu vào
        current = candidates[:k]
        best_score = self.objective(query, current)
        iterations = 0

        for _ in range(cfg.max_iter):
            iterations += 1
            state_best = current
            state_best_score = best_score
            improved = False

            # Steepest-ascent: duyệt toàn bộ láng giềng swap(i, j)
            for i in range(len(current)):
                for j in range(i + 1, len(current)):
                    neighbor = self._swap(current, i, j)
                    score = self.objective(query, neighbor)

                    if score > state_best_score:
                        state_best = neighbor
                        state_best_score = score
                        improved = True

            if improved and state_best_score > best_score:
                current = state_best
                best_score = state_best_score
            else:
                break

        # gắn lại score hiển thị gần với objective contribution tại thứ hạng cuối
        reranked: list[SearchResult] = []
        for idx, item in enumerate(current):
            new_item = SearchResult(
                kind=item.kind,
                item_id=item.item_id,
                title=item.title,
                score=self._base_doc_score(item, idx),
                source_path=item.source_path,
                snippet=item.snippet,
                data=item.data,
                lexical_score=item.lexical_score,
                semantic_score=item.semantic_score,
            )
            reranked.append(new_item)

        return HCRerankResult(
            selected=reranked,
            objective_score=best_score,
            iterations=iterations,
        )

    def optimize_grouped(
        self,
        query: str,
        grouped_results: dict[str, list[SearchResult]],
        total_k: int | None = None,
    ) -> dict[str, list[SearchResult]]:
        """
        Flatten tất cả candidates từ grouped_results rồi tối ưu một lần.
        Sau đó group lại để answer_builder dùng tiếp.
        """
        flat: list[SearchResult] = []
        seen: set[tuple[str, str]] = set()

        for kind, rows in grouped_results.items():
            for r in rows:
                key = (kind, r.item_id)
                if key not in seen:
                    seen.add(key)
                    flat.append(r)

        flat.sort(key=lambda x: (-x.score, -x.semantic_score, -x.lexical_score, x.kind, x.title))

        hc_result = self.rerank(query=query, candidates=flat, k=total_k or self.config.k)

        out: dict[str, list[SearchResult]] = {
            "procedure": [],
            "legal": [],
            "template": [],
            "case": [],
            "authority": [],
        }
        for r in hc_result.selected:
            out.setdefault(r.kind, []).append(r)

        return out