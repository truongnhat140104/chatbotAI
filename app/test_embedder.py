from app.core.embedder import HotichEmbedder

MODEL_NAME = "bkai-foundation-models/vietnamese-bi-encoder"

embedder = HotichEmbedder(
    model_name=MODEL_NAME,
    use_word_segment=True,
)
result = embedder.build()
result.print_summary()