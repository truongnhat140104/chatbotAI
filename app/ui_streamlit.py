from __future__ import annotations

import requests
import streamlit as st

API_URL_DEFAULT = "http://127.0.0.1:8000"


def call_health(api_base: str) -> dict | None:
    try:
        resp = requests.get(f"{api_base}/health", timeout=10)
        resp.raise_for_status()
        return resp.json()
    except Exception:
        return None


def call_ask(api_base: str, query: str, per_kind: int = 3, use_llm: bool = True) -> dict:
    resp = requests.post(
        f"{api_base}/ask",
        json={
            "query": query,
            "per_kind": per_kind,
            "use_llm": use_llm,
        },
        timeout=120,
    )
    resp.raise_for_status()
    return resp.json()


def render_result_group(title: str, items: list[dict]) -> None:
    st.markdown(f"### {title}")
    if not items:
        st.info("Không có kết quả phù hợp.")
        return

    for idx, item in enumerate(items, start=1):
        with st.expander(f"{idx}. {item['title']}"):
            st.write(f"**ID:** {item['item_id']}")
            st.write(f"**Score:** {item['score']:.2f}")
            st.write(f"**Lexical:** {item.get('lexical_score', 0):.2f}")
            st.write(f"**Semantic:** {item.get('semantic_score', 0):.4f}")
            st.write(f"**Snippet:** {item['snippet']}")
            st.code(item["source_path"])


def render_citation_map(citation_map: dict | None) -> None:
    st.markdown("### Citation map")
    if not citation_map:
        st.info("Không có citation map.")
        return

    for k, v in citation_map.items():
        st.markdown(f"**{k}**: {v}")


def main() -> None:
    st.set_page_config(
        page_title="Trợ lý ảo Hộ tịch",
        page_icon="📄",
        layout="wide",
    )

    st.title("📄 Trợ lý ảo Hộ tịch")
    st.caption("Demo UI cho bộ dữ liệu hộ tịch + FastAPI backend + Hybrid RAG")

    with st.sidebar:
        st.header("Cấu hình")
        api_base = st.text_input("API base URL", value=API_URL_DEFAULT)
        per_kind = st.slider("Số kết quả mỗi nhóm", min_value=1, max_value=6, value=3)
        use_llm = st.checkbox("Use LLM (Qwen2.5)", value=True)

        st.divider()
        st.subheader("Câu hỏi gợi ý")
        suggestions = [
            "đăng ký khai sinh cần giấy tờ gì",
            "mất giấy khai sinh thì làm sao",
            "mẫu tờ khai đăng ký kết hôn",
            "thủ tục hành chính đăng kí kết hôn",
            "miễn lệ phí đăng ký hộ tịch được quy định trong văn bản nào",
            "Những nguyên tắc cơ bản của chế độ hôn nhân và gia đình",
        ]
        for q in suggestions:
            if st.button(q, use_container_width=True):
                st.session_state["pending_query"] = q

        st.divider()
        health = call_health(api_base)
        if health and health.get("status") == "ok":
            st.success("Backend đang hoạt động")
            stats = health.get("stats", {})
            st.write("**Thống kê dữ liệu**")
            st.write(
                {
                    "legal_docs": stats.get("legal_docs"),
                    "procedures": stats.get("procedures"),
                    "templates": stats.get("templates"),
                    "cases": stats.get("cases"),
                    "warnings": stats.get("warnings"),
                    "errors": stats.get("errors"),
                }
            )
        else:
            st.error("Không kết nối được backend")

    if "chat_history" not in st.session_state:
        st.session_state["chat_history"] = []

    pending_query = st.session_state.pop("pending_query", "")
    query = st.chat_input("Nhập câu hỏi về hộ tịch...")

    if pending_query and not query:
        query = pending_query

    if query:
        st.session_state["chat_history"].append({"role": "user", "content": query})

        try:
            data = call_ask(api_base, query, per_kind=per_kind, use_llm=use_llm)
            st.session_state["chat_history"].append(
                {
                    "role": "assistant",
                    "content": data["answer_text"],
                    "payload": data,
                }
            )
        except Exception as exc:
            st.session_state["chat_history"].append(
                {
                    "role": "assistant",
                    "content": f"Lỗi khi gọi API: {exc}",
                    "payload": None,
                }
            )

    for msg in st.session_state["chat_history"]:
        with st.chat_message("user" if msg["role"] == "user" else "assistant"):
            st.markdown(msg["content"])

            payload = msg.get("payload")
            if msg["role"] == "assistant" and payload:
                col1, col2 = st.columns([2, 1])

                with col1:
                    st.markdown("## Kết quả truy xuất")
                    results = payload.get("results", {})
                    intent = payload.get("intent")

                    if intent == "template":
                        render_result_group("Templates", results.get("template", []))
                        render_result_group("Procedures", results.get("procedure", []))
                        render_result_group("Legal docs", results.get("legal", []))
                        render_result_group("Cases", results.get("case", []))
                        render_result_group("Authority", results.get("authority", []))
                    elif intent == "legal":
                        render_result_group("Legal docs", results.get("legal", []))
                        render_result_group("Procedures", results.get("procedure", []))
                        render_result_group("Templates", results.get("template", []))
                        render_result_group("Cases", results.get("case", []))
                        render_result_group("Authority", results.get("authority", []))
                    else:
                        render_result_group("Procedures", results.get("procedure", []))
                        render_result_group("Legal docs", results.get("legal", []))
                        render_result_group("Templates", results.get("template", []))
                        render_result_group("Cases", results.get("case", []))
                        render_result_group("Authority", results.get("authority", []))

                with col2:
                    st.markdown("## Thông tin thêm")
                    st.write(f"**Intent:** {payload.get('intent')}")
                    st.write(f"**Answer mode:** {payload.get('answer_mode')}")
                    st.write(f"**LLM mode:** {payload.get('llm_mode')}")
                    st.write(f"**LLM used:** {payload.get('llm_used')}")

                    llm_error = payload.get("llm_error")
                    if llm_error:
                        st.warning(f"LLM fallback: {llm_error}")

                    st.write("**Điểm intent**")
                    st.json(payload.get("scores", {}))

                    render_citation_map(payload.get("citation_map"))

                    with st.expander("Xem JSON response"):
                        st.json(payload)


if __name__ == "__main__":
    main()