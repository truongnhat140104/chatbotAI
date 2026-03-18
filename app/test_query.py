from app.core.loader import HotichLoader
from app.core.hybrid_retriever import HotichHybridRetriever
from app.core.answer_builder import HotichAnswerBuilder
from app.core.hill_climbing_reranker import HotichHillClimbingReranker, HillClimbingConfig

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

builder = HotichAnswerBuilder(
    bundle,
    retriever,
    hc_reranker=hc_reranker,
)

query = input("Nhap cau hoi: ").strip()
result = builder.answer(query, per_kind=3)

print("\n" + "=" * 80)
print(result.answer_text)
print("=" * 80)