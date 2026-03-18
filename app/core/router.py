from __future__ import annotations
from dataclasses import dataclass
import re
import unicodedata
from dataclasses import dataclass
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
    Router rule-based đơn giản cho giai đoạn MVP.
    Intent chính:
    - procedure
    - template
    - legal
    - case
    """

    def __init__(self) -> None:
        self.intent_keywords: dict[str, list[str]] = {
            "procedure": [
                "thu tuc", "dang ky", "cap ban sao", "trich luc", "nop o dau",
                "co quan nao", "tham quyen", "thoi han", "le phi", "can giay to",
                "ho so", "trinh tu", "cach thuc nop", "giai quyet bao lau",
            ],
            "template": [
                "mau", "mau don", "to khai", "bieu mau", "form", "mau to khai",
                "mau giay", "file mau",
            ],
            "legal": [
                "can cu phap ly", "can cu", "theo luat nao", "theo nghi dinh nao",
                "thong tu", "nghi dinh", "luat", "van ban", "dieu may", "khoan may",
                "quy dinh", "co duoc khong theo luat",
            ],
            "case": [
                "neu", "truong hop", "tinh huong", "mat", "khong co", "bi bo roi",
                "co duoc khong", "xu ly sao", "lam sao", "phai lam gi", "sinh tai nha",
                "uy quyen", "qua han", "khong xac dinh cha", "mang thai ho",
            ],
        }

    def _detect_law_alias(self, q: str) -> str:
        if "hon nhan va gia dinh" in q:
            return "luat_hon_nhan_va_gia_dinh"
        if "luat ho tich" in q or "ho tich" in q:
            return "luat_ho_tich"
        return ""

    def _extract_article_lookup(self, q: str) -> dict[str, str]:
        """
        Nhận diện các câu kiểu:
        - dieu 1 cua luat hon nhan va gia dinh
        - khoan 2 dieu 5 luat ho tich
        - noi dung dieu 2 ...
        """
        article_no = ""
        clause_no = ""
        law_alias = ""

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

        # Heuristic boosters
        if any(x in q for x in ["dang ky khai sinh", "dang ky ket hon", "dang ky khai tu", "giam ho"]):
            scores["procedure"] += 4.0
            reasons.append("procedure: matched core procedure phrase")

        if any(x in q for x in ["to khai", "mau", "bieu mau"]):
            scores["template"] += 4.0
            reasons.append("template: matched form phrase")

        if any(x in q for x in ["luat", "nghi dinh", "thong tu", "vbhn", "dieu ", "khoan "]):
            scores["legal"] += 4.0
            reasons.append("legal: matched legal reference phrase")

        if any(x in q for x in ["neu", "khong co", "mat", "qua han", "bi bo roi", "uy quyen"]):
            scores["case"] += 4.0
            reasons.append("case: matched case-like phrase")

        # Nếu hỏi giấy tờ/hồ sơ thì thường vẫn là procedure trước
        if any(x in q for x in ["giay to", "ho so", "can gi", "can nhung gi"]):
            scores["procedure"] += 4.0
            reasons.append("procedure: dossier question boost")



        # Tie-break ưu tiên thực dụng hơn cho trợ lý hành chính
        priority = ["procedure", "template", "legal", "case"]

        legal_strong_patterns = [
            "van ban nao",
            "duoc quy dinh trong",
            "duoc quy dinh o dau",
            "can cu phap ly",
            "can cu",
            "theo van ban nao",
            "theo quy dinh nao",
            "dieu nao",
            "khoan nao",
        ]

        if any(p in q for p in legal_strong_patterns):
            scores["legal"] += 10.0
            reasons.append("legal: strong legal query boost")

        if "ho tich" in q and any(x in q for x in ["van ban", "quy dinh", "can cu", "phap ly"]):
            scores["legal"] += 6.0
            reasons.append("legal: broad hộ tịch legal scope boost")

        if "ho tich" in q and any(x in q for x in ["van ban", "quy dinh", "can cu", "phap ly"]):
            scores["procedure"] -= 3.0
            reasons.append("procedure: penalty under broad legal query")

        if any(x in q for x in ["mien le phi", "le phi", "mien giam", "muc thu"]):
            scores["legal"] += 4.0
            reasons.append("legal: fee/legal-finance phrase boost")

        if article_lookup["sub_intent"] == "legal_article_lookup":
            scores["legal"] += 15.0
            scores["procedure"] -= 5.0
            scores["template"] -= 4.0
            scores["case"] -= 4.0
            reasons.append(
                f"legal_article_lookup boost: law={article_lookup['law_alias']}, "
                f"article={article_lookup['article_no']}, clause={article_lookup['clause_no'] or '-'}"
            )

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