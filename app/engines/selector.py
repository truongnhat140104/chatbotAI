from __future__ import annotations

from app.core.router import RouteDecision

from .base_engine import BaseEngine


class EngineSelector:
    def __init__(
        self,
        *,
        legal_engine: BaseEngine,
        procedure_engine: BaseEngine,
        template_engine: BaseEngine,
        case_engine: BaseEngine,
    ) -> None:
        self.legal_engine = legal_engine
        self.procedure_engine = procedure_engine
        self.template_engine = template_engine
        self.case_engine = case_engine

    def pick(self, query: str, route: RouteDecision, requested_mode: str = "auto") -> BaseEngine:
        mode = (requested_mode or "auto").strip().lower()

        if mode == "legal":
            return self.legal_engine
        if mode == "procedure":
            return self.procedure_engine
        if mode == "template":
            return self.template_engine
        if mode == "case":
            return self.case_engine

        if route.sub_intent == "legal_article_lookup":
            return self.legal_engine

        if route.primary_intent == "template" and route.scores.get("template", 0.0) >= 6.0:
            return self.template_engine

        if route.primary_intent == "legal" and route.scores.get("legal", 0.0) >= 6.0:
            return self.legal_engine

        if route.primary_intent == "procedure" and route.scores.get("procedure", 0.0) >= 6.0:
            return self.procedure_engine

        return self.case_engine
