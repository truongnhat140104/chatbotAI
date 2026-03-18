from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import requests

from app.core.context_builder import BuiltContext


@dataclass
class LLMAnswerResult:
    query: str
    answer_text: str
    model_name: str
    mode: str
    raw_response: dict[str, Any]


class QwenLLMAnswerer:
    """
    Gọi Qwen2.5 qua OpenAI-compatible API.
    """

    def __init__(
        self,
        base_url: str = "http://127.0.0.1:1234/v1",
        model_name: str = "qwen2.5:latest",
        api_key: str = "not-needed",
        temperature: float = 0.1,
        max_tokens: int = 1200,
        timeout: int = 120,
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self.model_name = model_name
        self.api_key = api_key
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.timeout = timeout

    # ------------------------------------------------------------------
    # mode detection
    # ------------------------------------------------------------------

    @staticmethod
    def _norm(text: str) -> str:
        import re
        import unicodedata

        text = (text or "").strip().lower()
        text = unicodedata.normalize("NFD", text)
        text = "".join(ch for ch in text if unicodedata.category(ch) != "Mn")
        text = text.replace("đ", "d").replace("Đ", "d")
        text = re.sub(r"[^a-z0-9\s]", " ", text)
        text = re.sub(r"\s+", " ", text).strip()
        return text

    def _detect_mode(self, built_context: BuiltContext) -> str:
        q = self._norm(built_context.query)

        if any(x in q for x in ["to khai", "mau", "bieu mau", "form"]):
            return "template"

        if any(x in q for x in ["dieu ", "khoan ", "noi dung dieu", "nguyen tac co ban", "quyen va nghia vu"]):
            return "legal_article_lookup"

        if any(x in q for x in ["van ban nao", "can cu phap ly", "duoc quy dinh o dau", "theo quy dinh nao"]):
            return "legal"

        if any(x in q for x in ["truong hop", "mat", "khong co", "uy quyen", "qua han", "bi bo roi"]):
            return "case"

        return "procedure"

    # ------------------------------------------------------------------
    # prompts
    # ------------------------------------------------------------------

    def _base_rules(self) -> str:
        return (
            "Bạn là trợ lý pháp lý hành chính lĩnh vực hộ tịch.\n"
            "Bạn CHỈ được trả lời dựa trên context được cung cấp.\n"
            "Không được bịa thông tin ngoài context.\n"
            "Nếu context chưa đủ thì phải nói rõ là chưa đủ căn cứ trong dữ liệu hiện có.\n"
            "Mỗi nhận định quan trọng phải gắn citation dạng [C1], [C2] nếu context có nguồn tương ứng.\n"
            "Không được viện dẫn nguồn không có trong citation map.\n"
            "Ưu tiên trả lời ngắn gọn, đúng trọng tâm câu hỏi.\n"
        )

    def _system_prompt(self, built_context: BuiltContext) -> str:
        mode = self._detect_mode(built_context)

        common = self._base_rules()

        if mode == "legal_article_lookup":
            return common + (
                "Đây là câu hỏi tra cứu điều luật cụ thể.\n"
                "Chỉ tập trung vào nội dung điều/khoản được hỏi.\n"
                "Không mở rộng sang thủ tục hành chính, biểu mẫu, tình huống khác nếu context không yêu cầu.\n"
                "Cấu trúc trả lời:\n"
                "1. Kết luận / nội dung chính của điều luật\n"
                "2. Diễn giải ngắn gọn\n"
                "3. Citation\n"
            )

        if mode == "template":
            return common + (
                "Đây là câu hỏi về biểu mẫu / tờ khai.\n"
                "Ưu tiên trả lời theo cấu trúc:\n"
                "1. Tên biểu mẫu\n"
                "2. Biểu mẫu dùng để làm gì\n"
                "3. Các trường thông tin chính cần điền\n"
                "4. Lưu ý khi điền\n"
                "5. Thủ tục liên quan chỉ nêu ngắn gọn nếu thật sự có trong context\n"
                "Không lan man sang thủ tục nếu người dùng chỉ hỏi về biểu mẫu.\n"
            )

        if mode == "legal":
            return common + (
                "Đây là câu hỏi pháp lý/căn cứ pháp lý.\n"
                "Ưu tiên nêu văn bản, điều khoản, phạm vi áp dụng và citation.\n"
                "Không ép sang một thủ tục cụ thể nếu câu hỏi mang tính pháp lý chung.\n"
            )

        if mode == "case":
            return common + (
                "Đây là câu hỏi theo tình huống.\n"
                "Ưu tiên trả lời theo cấu trúc:\n"
                "1. Kết luận sơ bộ\n"
                "2. Cách xử lý / bước cần làm\n"
                "3. Hồ sơ hoặc điều kiện liên quan\n"
                "4. Căn cứ pháp lý\n"
            )

        return common + (
            "Đây là câu hỏi thủ tục hành chính.\n"
            "Ưu tiên trả lời theo cấu trúc:\n"
            "1. Thủ tục phù hợp\n"
            "2. Hồ sơ / giấy tờ cần chuẩn bị\n"
            "3. Thẩm quyền / nơi nộp\n"
            "4. Thời hạn / lệ phí nếu có\n"
            "5. Biểu mẫu liên quan nếu có trong context\n"
        )

    def _user_prompt(self, built_context: BuiltContext) -> str:
        mode = self._detect_mode(built_context)

        citation_lines = [f"{k}: {v}" for k, v in built_context.citation_map.items()]
        citation_text = "\n".join(citation_lines) if citation_lines else "Không có citation map."

        extra_instruction = ""
        if mode == "legal_article_lookup":
            extra_instruction = (
                "Yêu cầu đặc biệt: nếu context có đúng điều/khoản được hỏi thì hãy trả lời trực tiếp nội dung điều luật đó. "
                "Nếu context không có đúng điều/khoản thì nói rõ là context hiện tại chưa chứa đúng điều/khoản này.\n"
            )
        elif mode == "template":
            extra_instruction = (
                "Yêu cầu đặc biệt: nếu context có biểu mẫu, hãy liệt kê các trường thông tin chính có thể nhìn thấy trong context.\n"
            )

        return (
            f"CÂU HỎI NGƯỜI DÙNG:\n{built_context.query}\n\n"
            f"MODE SUY LUẬN:\n{mode}\n\n"
            f"CITATION MAP:\n{citation_text}\n\n"
            f"CONTEXT:\n{built_context.context_text}\n\n"
            f"{extra_instruction}"
            "Hãy trả lời bằng tiếng Việt.\n"
            "Chỉ dùng thông tin trong CONTEXT.\n"
            "Nếu thiếu căn cứ thì nói rõ.\n"
            "Mỗi nhận định quan trọng nên gắn citation [C?].\n"
        )

    # ------------------------------------------------------------------
    # API call
    # ------------------------------------------------------------------

    def _headers(self) -> dict[str, str]:
        return {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.api_key}",
        }

    def _payload(self, built_context: BuiltContext) -> dict[str, Any]:
        return {
            "model": self.model_name,
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
            "messages": [
                {"role": "system", "content": self._system_prompt(built_context)},
                {"role": "user", "content": self._user_prompt(built_context)},
            ],
        }

    def _extract_text(self, response_json: dict[str, Any]) -> str:
        choices = response_json.get("choices", [])
        if not choices:
            return "Không nhận được nội dung trả lời từ mô hình."

        first = choices[0]
        message = first.get("message", {})
        content = message.get("content", "")

        if isinstance(content, str):
            return content.strip()

        if isinstance(content, list):
            chunks = []
            for part in content:
                if isinstance(part, dict) and part.get("type") == "text":
                    chunks.append(part.get("text", ""))
            return "\n".join(chunks).strip()

        return str(content).strip()

    def answer(self, built_context: BuiltContext) -> LLMAnswerResult:
        url = f"{self.base_url}/chat/completions"
        payload = self._payload(built_context)
        mode = self._detect_mode(built_context)

        resp = requests.post(
            url,
            headers=self._headers(),
            json=payload,
            timeout=self.timeout,
        )
        resp.raise_for_status()

        response_json = resp.json()
        answer_text = self._extract_text(response_json)

        return LLMAnswerResult(
            query=built_context.query,
            answer_text=answer_text,
            model_name=self.model_name,
            mode=mode,
            raw_response=response_json,
        )