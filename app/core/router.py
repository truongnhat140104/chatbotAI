from __future__ import annotations

from dataclasses import dataclass
import re
import unicodedata
from typing import Any


@dataclass
class RouteDecision:
    primary_intent: str
    scores: dict[str, float]
    reasons: list[str]
    sub_intent: str = ""
    article_no: str = ""
    clause_no: str = ""
    law_alias: str = ""


class HotichRouter:
    """
    Router rule-based ưu tiên:
    - legal cho câu hỏi điều/khoản/căn cứ pháp lý
    - case cho câu hỏi tình huống / ngoại lệ / cần suy luận
    - template cho câu hỏi biểu mẫu
    - procedure cho câu hỏi hồ sơ, thẩm quyền, lệ phí, thời hạn, nơi nộp
    """

    def __init__(self) -> None:
        self.intent_keywords: dict[str, list[str]] = {
            "procedure": [
                "thu tuc", "dang ky", "cap ban sao", "trich luc",
                "nop o dau", "co quan nao", "tham quyen", "thoi han",
                "le phi", "can giay to", "ho so", "trinh tu",
                "cach thuc nop", "giai quyet bao lau", "xu ly trong bao lau",
                "thanh phan ho so",
            ],
            "template": [
                "mau", "mau don", "to khai", "bieu mau", "form",
                "mau to khai", "mau giay", "file mau",
            ],
            "legal": [
                "can cu phap ly", "can cu", "theo luat nao", "theo nghi dinh nao",
                "thong tu", "nghi dinh", "luat", "van ban", "quy dinh",
                "dieu may", "khoan may", "quy dinh o dau", "duoc quy dinh trong",
                "theo quy dinh nao", "theo van ban nao",
            ],
            "case": [
                "neu", "truong hop", "tinh huong", "mat", "khong co",
                "bi bo roi", "co duoc khong", "xu ly sao", "lam sao", "phai lam gi",
                "uy quyen", "qua han", "khong xac dinh cha", "mang thai ho",
                "sai thong tin", "cai chinh", "cap lai", "khong con giay to",
            ],
        }

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

    def _detect_law_alias(self, q: str) -> str:
        if "hon nhan va gia dinh" in q:
            return "luat_hon_nhan_va_gia_dinh"

        if any(x in q for x in ["luat ho tich", "60 2014 qh13", "60/2014/qh13"]):
            return "luat_ho_tich"

        return ""

    def _extract_article_lookup(self, q: str) -> dict[str, str]:
        article_no = ""
        clause_no = ""

        m_article = re.search(r"\bdieu\s+(\d+)\b", q)
        if m_article:
            article_no = m_article.group(1)

        m_clause = re.search(r"\bkhoan\s+(\d+)\b", q)
        if m_clause:
            clause_no = m_clause.group(1)

        law_alias = self._detect_law_alias(q)

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

    def route(self, query: str) -> RouteDecision:
        q = self._normalize_text(query)
        article_lookup = self._extract_article_lookup(q)

        scores = {
            "procedure": 0.0,
            "template": 0.0,
            "legal": 0.0,
            "case": 0.0,
        }
        reasons: list[str] = []

        for intent, keywords in self.intent_keywords.items():
            for kw in keywords:
                if kw in q:
                    scores[intent] += 3.0
                    reasons.append(f"{intent}: matched '{kw}'")

        # ----------------------------
        # Strong pattern groups
        # ----------------------------
        legal_markers = [
            "luat", "nghi dinh", "thong tu", "vbhn", "van ban",
            "quy dinh", "can cu phap ly", "can cu", "dieu ", "khoan ",
            "theo quy dinh nao", "theo van ban nao", "quy dinh o dau",
        ]
        case_markers = [
            "neu", "truong hop", "tinh huong", "co duoc khong", "mat",
            "khong co", "qua han", "uy quyen", "bi bo roi",
            "sai thong tin", "cai chinh", "cap lai", "khong con giay to",
        ]
        procedure_markers = [
            "ho so", "giay to", "can gi", "can nhung gi", "thanh phan ho so",
            "thoi han", "le phi", "nop o dau", "tham quyen", "co quan nao",
            "trinh tu", "cach thuc nop", "giai quyet bao lau",
        ]
        template_markers = [
            "mau", "mau don", "to khai", "bieu mau", "form",
        ]

        if any(x in q for x in legal_markers):
            scores["legal"] += 8.0
            reasons.append("legal: strong legal marker boost")

        if any(x in q for x in case_markers):
            scores["case"] += 8.0
            reasons.append("case: strong case marker boost")

        if any(x in q for x in procedure_markers):
            scores["procedure"] += 6.0
            reasons.append("procedure: strong procedure marker boost")

        if any(x in q for x in template_markers):
            scores["template"] += 7.0
            reasons.append("template: strong template marker boost")

        # ----------------------------
        # Domain-specific boosts
        # ----------------------------
        if any(x in q for x in ["dang ky khai sinh", "dang ky ket hon", "dang ky khai tu", "giam ho"]):
            scores["procedure"] += 4.0
            reasons.append("procedure: matched core procedure phrase")

        if re.search(r"\bdieu\s+\d+\b", q) or re.search(r"\bkhoan\s+\d+\b", q):
            scores["legal"] += 10.0
            scores["procedure"] -= 3.0
            scores["template"] -= 3.0
            reasons.append("legal: direct article/clause lookup boost")

        if any(x in q for x in ["noi dung dieu", "dieu nao", "khoan nao", "quy dinh gi"]):
            scores["legal"] += 8.0
            reasons.append("legal: article content question boost")

        if any(x in q for x in ["co duoc khong", "truong hop", "neu", "qua han", "uy quyen"]):
            scores["case"] += 6.0
            scores["procedure"] -= 2.0
            reasons.append("case: scenario reasoning boost")

        if any(x in q for x in ["mau to khai", "to khai", "bieu mau"]) and not any(x in q for x in legal_markers):
            scores["template"] += 6.0
            reasons.append("template: explicit form request boost")

        if any(x in q for x in ["ho so", "giay to", "thanh phan ho so", "le phi", "thoi han"]):
            scores["procedure"] += 5.0
            reasons.append("procedure: dossier/fee/time boost")

        if any(x in q for x in ["tham quyen", "co quan nao", "nop o dau"]):
            scores["procedure"] += 5.0
            reasons.append("procedure: authority/place boost")

        # ----------------------------
        # Penalties to reduce wrong routing
        # ----------------------------
        if any(x in q for x in legal_markers):
            scores["procedure"] -= 2.0
            reasons.append("procedure: penalty under legal-style query")

        if any(x in q for x in case_markers):
            scores["legal"] -= 1.0
            reasons.append("legal: slight penalty under case-style query")

        if any(x in q for x in template_markers):
            scores["procedure"] -= 1.0
            reasons.append("procedure: slight penalty under template-style query")

        # Special handling for legal article lookup
        if article_lookup["sub_intent"] == "legal_article_lookup":
            scores["legal"] += 20.0
            scores["procedure"] -= 5.0
            scores["template"] -= 4.0
            scores["case"] -= 4.0
            reasons.append(
                f"legal_article_lookup boost: law={article_lookup['law_alias']}, "
                f"article={article_lookup['article_no']}, clause={article_lookup['clause_no'] or '-'}"
            )

        # Priority order when tie
        priority = ["legal", "case", "template", "procedure"]

        primary_intent = sorted(
            scores.keys(),
            key=lambda k: (-scores[k], priority.index(k))
        )[0]

        return RouteDecision(
            primary_intent=primary_intent,
            scores=scores,
            reasons=reasons,
            sub_intent=article_lookup["sub_intent"],
            article_no=article_lookup["article_no"],
            clause_no=article_lookup["clause_no"],
            law_alias=article_lookup["law_alias"],
        )