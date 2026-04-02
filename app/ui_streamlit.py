from __future__ import annotations

import requests
import streamlit as st

API_URL_DEFAULT = "http://127.0.0.1:8000"
MODE_LABELS = {
    "Tu dong (Auto)": "auto",
    "Tra cuu luat": "legal",
    "Thu tuc hanh chinh": "procedure",
    "Bieu mau": "template",
    "Tinh huong": "case",
}


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


def render_result_group(title: str, items: list[dict]) -> None:
    if not items:
        st.info(f"Khong tim thay du lieu cho nhom **{title}**.")
        return

    for idx, item in enumerate(items, start=1):
        with st.expander(f"{idx}. {item['title']}", expanded=False):
            st.markdown(f"**Trich doan:**\n> {item['snippet']}")
            st.caption(f"Nguon: `{item['source_path']}` | ID: `{item['item_id']}`")
            c1, c2, c3 = st.columns(3)
            c1.metric("Tong diem", f"{item['score']:.2f}")
            c2.metric("Ngu nghia", f"{item.get('semantic_score', 0):.4f}")
            c3.metric("Tu vung", f"{item.get('lexical_score', 0):.2f}")


def render_results_by_mode(payload: dict) -> None:
    results = payload.get("results", {})
    display_mode = payload.get("query_mode") or payload.get("intent") or "procedure"

    orders = {
        "template": [("Bieu mau", "template"), ("Thu tuc", "procedure"), ("Van ban luat", "legal"), ("Tinh huong", "case"), ("Tham quyen", "authority")],
        "legal": [("Van ban luat", "legal"), ("Thu tuc", "procedure"), ("Bieu mau", "template"), ("Tinh huong", "case"), ("Tham quyen", "authority")],
        "legal_article_lookup": [("Van ban luat", "legal"), ("Thu tuc", "procedure"), ("Bieu mau", "template"), ("Tinh huong", "case"), ("Tham quyen", "authority")],
        "case": [("Tinh huong", "case"), ("Thu tuc", "procedure"), ("Van ban luat", "legal"), ("Bieu mau", "template"), ("Tham quyen", "authority")],
        "procedure": [("Thu tuc", "procedure"), ("Van ban luat", "legal"), ("Bieu mau", "template"), ("Tinh huong", "case"), ("Tham quyen", "authority")],
    }

    current_order = orders.get(display_mode, orders["procedure"])
    tabs = st.tabs([title for title, _ in current_order])
    for tab, (title, key) in zip(tabs, current_order):
        with tab:
            render_result_group(title, results.get(key, []))


def render_meta(payload: dict) -> None:
    col_a, col_b, col_c = st.columns(3)
    with col_a:
        st.markdown("**Router**")
        st.info(
            f"Requested: `{payload.get('requested_mode')}`\n\n"
            f"Auto intent: `{payload.get('auto_intent')}`\n\n"
            f"Intent: `{payload.get('intent')}`"
        )
    with col_b:
        st.markdown("**Engine**")
        st.info(
            f"Selected engine: `{payload.get('selected_engine', 'n/a')}`\n\n"
            f"Query mode: `{payload.get('query_mode')}`\n\n"
            f"Answer mode: `{payload.get('answer_mode')}`"
        )
    with col_c:
        st.markdown("**LLM**")
        st.info(
            f"LLM used: `{payload.get('llm_used')}`\n\n"
            f"LLM mode: `{payload.get('llm_mode')}`\n\n"
            f"Sub intent: `{payload.get('sub_intent')}`"
        )

    llm_error = payload.get("llm_error")
    if llm_error:
        st.warning(f"LLM/Error: {llm_error}")

    engine_debug = payload.get("engine_debug")
    if engine_debug:
        st.markdown("**Engine debug**")
        st.json(engine_debug)

    st.markdown("**Intent scores**")
    st.json(payload.get("scores", {}))

    if payload.get("citation_map"):
        st.markdown("**Citation Map**")
        st.json(payload.get("citation_map"))


def main() -> None:
    st.set_page_config(
        page_title="Tro ly ao Ho tich",
        page_icon="⚖️",
        layout="wide",
        initial_sidebar_state="expanded",
    )

    if "chat_history" not in st.session_state:
        st.session_state["chat_history"] = []
    if "current_query" not in st.session_state:
        st.session_state["current_query"] = ""

    st.title("⚖️ Tro ly ao Ho tich")
    st.caption("Tra cuu quy dinh phap luat, thu tuc, tinh huong va bieu mau Ho tich.")

    with st.sidebar:
        st.header("Cau hinh he thong")
        with st.expander("Ket noi va truy van", expanded=False):
            api_base = st.text_input("API Base URL", value=API_URL_DEFAULT)
            use_llm = st.toggle("Su dung LLM de dien dat cau tra loi", value=True)
            per_kind = st.slider("So luong tai lieu truy xuat moi loai", min_value=1, max_value=6, value=3)

        selected_mode_label = st.selectbox("Che do truy van uu tien", list(MODE_LABELS.keys()), index=0)
        selected_mode = MODE_LABELS[selected_mode_label]

        st.divider()
        st.subheader("Cau hoi goi y")
        suggestions = [
            "Dang ky khai sinh can chuan bi giay to gi?",
            "Lam mat ban chinh giay khai sinh thi co cap lai duoc khong?",
            "Cho toi xin mau to khai dang ky ket hon.",
            "Nguoi nuoc ngoai ket hon voi nguoi Viet Nam lam thu tuc o dau?",
            "Truong hop nao duoc mien le phi ho tich?",
            "Dieu 17 Luat Ho tich quy dinh gi?",
        ]
        for q in suggestions:
            if st.button(q, use_container_width=True):
                st.session_state["current_query"] = q

        st.divider()
        with st.spinner("Dang kiem tra ket noi..."):
            health = call_health(api_base)

        if health and health.get("status") == "ok":
            st.success("Backend dang hoat dong")
            stats = health.get("stats", {})
            with st.expander("Thong ke du lieu"):
                st.write(f"- Van ban luat: {stats.get('legal_docs', 0)}")
                st.write(f"- Thu tuc: {stats.get('procedures', 0)}")
                st.write(f"- Bieu mau: {stats.get('templates', 0)}")
                st.write(f"- Tinh huong: {stats.get('cases', 0)}")
                st.write(f"- Kien truc: {stats.get('architecture', 'unknown')}")
                warnings = stats.get("init_warnings") or []
                if warnings:
                    st.warning("\n".join(warnings))
        else:
            st.error("Khong the ket noi toi Backend")

    for i, msg in enumerate(st.session_state["chat_history"]):
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])
            payload = msg.get("payload")
            if msg["role"] == "assistant" and payload:
                is_latest = i == len(st.session_state["chat_history"]) - 1
                with st.expander("Xem nguon tai lieu va thong tin ky thuat", expanded=is_latest):
                    tab_docs, tab_meta, tab_json = st.tabs(["Nguon tai lieu", "Metadata", "JSON Response"])
                    with tab_docs:
                        render_results_by_mode(payload)
                    with tab_meta:
                        render_meta(payload)
                    with tab_json:
                        st.json(payload)

    user_input = st.chat_input("Nhap cau hoi ve quy dinh, thu tuc ho tich...")
    query = user_input or st.session_state.get("current_query")

    if query:
        st.session_state["current_query"] = ""
        st.session_state["chat_history"].append({"role": "user", "content": query})

        with st.chat_message("user"):
            st.markdown(query)

        with st.chat_message("assistant"):
            with st.spinner("Dang tra cuu du lieu ho tich..."):
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
                    st.rerun()
                except Exception as exc:
                    error_msg = f"Loi goi API: `{exc}`"
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
