from app.core.loader import HotichLoader
from app.core.hybrid_retriever import HotichHybridRetriever
from app.core.hill_climbing_reranker import HotichHillClimbingReranker, HillClimbingConfig
from app.core.context_builder import HotichContextBuilder

loader = HotichLoader()
bundle = loader.load_all()

retriever = HotichHybridRetriever(
    model_name="bkai-foundation-models/vietnamese-bi-encoder",
    use_semantic=True,
    lexical_weight=1.0,
    semantic_weight=20.0,
)

hc_reranker = HotichHillClimbingReranker(
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

query = input("Nhap cau hoi: ").strip()

groups = {
    "procedure": retriever.search(query, source_kinds=["procedure"], top_k=5, min_score=1.0),
    "legal": retriever.search(query, source_kinds=["legal"], top_k=8, min_score=1.0),
    "template": retriever.search(query, source_kinds=["template"], top_k=3, min_score=1.0),
    "case": retriever.search(query, source_kinds=["case"], top_k=3, min_score=1.0),
    "authority": retriever.search(query, source_kinds=["authority"], top_k=3, min_score=1.0),
}

groups = hc_reranker.optimize_grouped(query=query, grouped_results=groups, total_k=10)

builder = HotichContextBuilder(
    max_procedures=1,
    max_legals=4,
    max_templates=1,
    max_cases=1,
    max_authorities=1,
)

ctx = builder.build(query=query, grouped_results=groups)

print("\n" + "=" * 80)
print(ctx.context_text)
print("=" * 80)
print("\nCITATION MAP:")
for k, v in ctx.citation_map.items():
    print(f"{k}: {v}")