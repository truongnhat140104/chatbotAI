from app.core.loader import HotichLoader
from app.core.hybrid_retriever import HotichHybridRetriever
from app.core.hill_climbing_reranker import HotichHillClimbingReranker, HillClimbingConfig
from app.core.context_builder import HotichContextBuilder
from app.core.llm_answerer import QwenLLMAnswerer

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

context_builder = HotichContextBuilder(
    max_procedures=1,
    max_legals=4,
    max_templates=1,
    max_cases=1,
    max_authorities=1,
)

built_context = context_builder.build(query=query, grouped_results=groups)

answerer = QwenLLMAnswerer(
    base_url="http://127.0.0.1:11434/",   # sửa nếu server khác
    model_name="qwen2.5:3b",           # sửa theo model bạn đang serve
    api_key="ollama",
    temperature=0.1,
    max_tokens=1200,
)

result = answerer.answer(built_context)

print("\n" + "=" * 80)
print(result.answer_text)
print("=" * 80)
print("\nMODEL:", result.model_name)