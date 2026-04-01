from __future__ import annotations

import requests
import streamlit as st

API_URL_DEFAULT = "http://127.0.0.1:8000"
MODE_LABELS = {
    "Tự động (Auto)": "auto",
    "Tra cứu luật": "legal",
    "Thủ tục hành chính": "procedure",
    "Biểu mẫu": "template",
    "Tình huống": "case",
}


# --- API FUNCTIONS ---
def call_health(api_base: str) -> dict | None:
    try:
        resp = requests.get(f"{api_base}/health", timeout=10)
        resp.raise_for_status()
        return resp.json()
    except Exception:
        return None


def call_ask(
        api_base: str,
        query: str,
        per_kind: int = 3,
        use_llm: bool = True,
        mode: str = "auto",
) -> dict:
    resp = requests.post(
        f"{api_base}/ask",
        json={
            "query": query,
            "per_kind": per_kind,
            "use_llm": use_llm,
            "mode": mode,
        },
        timeout=120,
    )
    resp.raise_for_status()
    return resp.json()


# --- UI RENDER FUNCTIONS ---
def render_result_group(title: str, items: list[dict]) -> None:
    if not items:
        st.info(f"Không tìm thấy dữ liệu cho nhóm **{title}**.")
        return

    for idx, item in enumerate(items, start=1):
        with st.expander(f"📄 {idx}. {item['title']}", expanded=False):
            st.markdown(f"**Trích đoạn:**\n> {item['snippet']}")
            st.caption(f"📁 Nguồn: `{item['source_path']}` | 🔑 ID: `{item['item_id']}`")

            # Trực quan hóa điểm số bằng cột
            c1, c2, c3 = st.columns(3)
            c1.metric("Tổng điểm (Score)", f"{item['score']:.2f}")
            c2.metric("Ngữ nghĩa (Semantic)", f"{item.get('semantic_score', 0):.4f}")
            c3.metric("Từ vựng (Lexical)", f"{item.get('lexical_score', 0):.2f}")


def render_results_by_mode(payload: dict) -> None:
    results = payload.get("results", {})
    display_mode = payload.get("intent") or "procedure"

    # Sắp xếp thứ tự ưu tiên các tab dựa trên intent
    orders = {
        "template": [("Biểu mẫu", "template"), ("Thủ tục", "procedure"), ("Văn bản luật", "legal"),
                     ("Tình huống", "case"), ("Thẩm quyền", "authority")],
        "legal": [("Văn bản luật", "legal"), ("Thủ tục", "procedure"), ("Biểu mẫu", "template"), ("Tình huống", "case"),
                  ("Thẩm quyền", "authority")],
        "case": [("Tình huống", "case"), ("Thủ tục", "procedure"), ("Văn bản luật", "legal"), ("Biểu mẫu", "template"),
                 ("Thẩm quyền", "authority")],
        "procedure": [("Thủ tục", "procedure"), ("Văn bản luật", "legal"), ("Biểu mẫu", "template"),
                      ("Tình huống", "case"), ("Thẩm quyền", "authority")],
    }

    current_order = orders.get(display_mode, orders["procedure"])

    # Sử dụng Tabs thay vì dàn hàng dọc dài ngoằng
    tabs = st.tabs([title for title, _ in current_order])

    for tab, (title, key) in zip(tabs, current_order):
        with tab:
            render_result_group(title, results.get(key, []))


# --- MAIN APP ---
def main() -> None:
    st.set_page_config(
        page_title="Trợ lý ảo Hộ tịch",
        page_icon="⚖️",
        layout="wide",
        initial_sidebar_state="expanded"
    )

    # Khởi tạo state
    if "chat_history" not in st.session_state:
        st.session_state["chat_history"] = []
    if "current_query" not in st.session_state:
        st.session_state["current_query"] = ""

    # Header
    st.title("⚖️ Trợ lý ảo Hộ tịch")
    st.caption("Tra cứu quy định pháp luật, thủ tục, tình huống và biểu mẫu Hộ tịch.")

    # --- SIDEBAR ---
    with st.sidebar:
        st.header("⚙️ Cấu hình hệ thống")

        with st.expander("Tham số kết nối & Model", expanded=False):
            api_base = st.text_input("API Base URL", value=API_URL_DEFAULT)
            use_llm = st.toggle("Sử dụng LLM (Qwen2.5) sinh câu trả lời", value=True)
            per_kind = st.slider("Số lượng tài liệu truy xuất (mỗi loại)", min_value=1, max_value=6, value=3)

        selected_mode_label = st.selectbox("🎯 Chế độ truy vấn ưu tiên", list(MODE_LABELS.keys()), index=0)
        selected_mode = MODE_LABELS[selected_mode_label]

        st.divider()
        st.subheader("💡 Câu hỏi gợi ý")
        suggestions = [
            "Đăng ký khai sinh cần chuẩn bị giấy tờ gì?",
            "Làm mất bản chính giấy khai sinh thì có cấp lại được không?",
            "Cho tôi xin mẫu tờ khai đăng ký kết hôn.",
            "Người nước ngoài kết hôn với người Việt Nam làm thủ tục ở đâu?",
            "Trường hợp nào được miễn lệ phí hộ tịch?",
        ]

        for q in suggestions:
            if st.button(q, use_container_width=True):
                st.session_state["current_query"] = q

        st.divider()
        # Health check
        with st.spinner("Đang kiểm tra kết nối..."):
            health = call_health(api_base)

        if health and health.get("status") == "ok":
            st.success("🟢 Backend đang hoạt động")
            stats = health.get("stats", {})

            with st.expander("📊 Thống kê dữ liệu Index"):
                st.write(f"- **Văn bản luật:** {stats.get('legal_docs', 0)}")
                st.write(f"- **Thủ tục:** {stats.get('procedures', 0)}")
                st.write(f"- **Biểu mẫu:** {stats.get('templates', 0)}")
                st.write(f"- **Tình huống:** {stats.get('cases', 0)}")
        else:
            st.error("🔴 Không thể kết nối tới Backend")

    # --- CHAT INTERFACE ---

    # Hiển thị lịch sử chat
    for i, msg in enumerate(st.session_state["chat_history"]):
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

            payload = msg.get("payload")
            if msg["role"] == "assistant" and payload:
                # Chỉ mở rộng (expand) chi tiết kỹ thuật ở câu trả lời mới nhất
                is_latest = (i == len(st.session_state["chat_history"]) - 1)

                with st.expander("🔎 Xem nguồn tài liệu & Thông tin kỹ thuật", expanded=is_latest):
                    tab_docs, tab_meta, tab_json = st.tabs(["📚 Nguồn tài liệu", "⚙️ Metadata", "💻 JSON Response"])

                    with tab_docs:
                        render_results_by_mode(payload)

                    with tab_meta:
                        col_a, col_b = st.columns(2)
                        with col_a:
                            st.markdown("**Trạng thái Intent**")
                            st.info(
                                f"Yêu cầu: `{payload.get('requested_mode')}`\n\nAuto Intent: `{payload.get('auto_intent')}`\n\nThực thi: `{payload.get('intent')}`")
                        with col_b:
                            st.markdown("**Thông tin LLM**")
                            st.info(
                                f"Query Mode: `{payload.get('query_mode')}`\n\nLLM Mode: `{payload.get('llm_mode')}`\n\nModel: `{payload.get('llm_used')}`")

                        llm_error = payload.get("llm_error")
                        if llm_error:
                            st.warning(f"⚠️ LLM Fallback Error: {llm_error}")

                        st.markdown("**Phân tích điểm Intent (Auto)**")
                        st.json(payload.get("scores", {}))

                        if payload.get("citation_map"):
                            st.markdown("**Citation Map**")
                            st.json(payload.get("citation_map"))

                    with tab_json:
                        st.json(payload)

    # Xử lý input (từ thanh chat hoặc nút gợi ý)
    user_input = st.chat_input("Nhập câu hỏi về quy định, thủ tục hộ tịch...")
    query = user_input or st.session_state.get("current_query")

    if query:
        # Reset current_query state
        st.session_state["current_query"] = ""

        # Thêm câu hỏi của user vào lịch sử
        st.session_state["chat_history"].append({"role": "user", "content": query})

        # Render tin nhắn user ngay lập tức
        with st.chat_message("user"):
            st.markdown(query)

        # Gọi API và lấy phản hồi
        with st.chat_message("assistant"):
            with st.spinner("Đang tra cứu dữ liệu hộ tịch..."):
                try:
                    data = call_ask(
                        api_base,
                        query,
                        per_kind=per_kind,
                        use_llm=use_llm,
                        mode=selected_mode,
                    )
                    st.markdown(data["answer_text"])
                    st.session_state["chat_history"].append(
                        {
                            "role": "assistant",
                            "content": data["answer_text"],
                            "payload": data,
                        }
                    )
                    st.rerun()  # Rerun để render lại giao diện với expander đúng trạng thái
                except Exception as exc:
                    error_msg = f"❌ Đã có lỗi xảy ra khi gọi API: `{exc}`"
                    st.error(error_msg)
                    st.session_state["chat_history"].append(
                        {
                            "role": "assistant",
                            "content": error_msg,
                            "payload": None,
                        }
                    )


if __name__ == "__main__":
    main()