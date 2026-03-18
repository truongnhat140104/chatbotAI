from app.core.hybrid_retriever import HotichHybridRetriever, print_grouped_results

retriever = HotichHybridRetriever(
    model_name="bkai-foundation-models/vietnamese-bi-encoder",
    use_semantic=True,
    lexical_weight=1.0,
    semantic_weight=20.0,
)

while True:
    query = input("Nhap cau hoi: ").strip()
    if not query or query.lower() in {"exit", "quit"}:
        break

    groups = retriever.grouped_search(query, per_kind=3)
    print_grouped_results(groups)