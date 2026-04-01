from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from app.core.hill_climbing_reranker import HotichHillClimbingReranker
from app.core.hybrid_retriever import HotichHybridRetriever, SearchResult
from app.core.loader import HotichBundle
from app.core.router import RouteDecision

from .common import make_empty_groups


@dataclass
class EngineRunResult:
    engine_name: str
    query_mode: str
    answer_text: str
    grouped_results: dict[str, list[SearchResult]]
    debug: dict[str, Any] = field(default_factory=dict)


class BaseEngine:
    name = "base"
    query_mode = "auto"

    def __init__(
        self,
        *,
        bundle: HotichBundle,
        retriever: HotichHybridRetriever,
        hc_reranker: HotichHillClimbingReranker | None = None,
    ) -> None:
        self.bundle = bundle
        self.retriever = retriever
        self.hc_reranker = hc_reranker

    def can_handle(self, query: str, route: RouteDecision, requested_mode: str = "auto") -> bool:
        return False

    def run(self, query: str, route: RouteDecision, per_kind: int = 3) -> EngineRunResult:
        raise NotImplementedError

    @staticmethod
    def empty_groups() -> dict[str, list[SearchResult]]:
        return make_empty_groups()
