from app.core.loader import HotichLoader
from app.core.hybrid_retriever import HotichHybridRetriever, print_grouped_results
from app.core.hill_climbing_reranker import HotichHillClimbingReranker, HillClimbingConfig

loader = HotichLoader()
bundle = loader.load_all()

retriever = HotichHybridRetriever(
    model_name="bkai-foundation-models/vietnamese-bi-encoder",
    use_semantic=True,
    lexical_weight=1.0,
    semantic_weight=20.0,
)

reranker = HotichHillClimbingReranker(
    bundle=bundle,
    config=HillClimbingConfig(
        k=10,
        m=6,
        max_iter=20,
        lambda_redundancy=0.80,
        eta_constraint=2.00,
        gamma_coverage=0.50,
    ),
)

while True:
    query = input("Nhap cau hoi: ").strip()
    if not query or query.lower() in {"exit", "quit"}:
        break

    groups = retriever.grouped_search(query, per_kind=5)

    print("\n=== BEFORE HC ===")
    print_grouped_results(groups)

    optimized = reranker.optimize_grouped(query, groups, total_k=10)

    print("\n=== AFTER HC ===")
    print_grouped_results(optimized)