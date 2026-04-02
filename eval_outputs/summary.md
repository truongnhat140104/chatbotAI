# Kết quả đánh giá RAG/Engine

- Dataset: `rag_eval_normalized.jsonl`
- API Base: `http://127.0.0.1:8000`
- Mode: `auto`
- Use LLM: `True`

## Tổng quan

- Số mẫu: **150**
- Thành công: **150**
- Lỗi: **0**
- Exact Match: **0.0%**
- Token F1: **20.02%**
- Routing Accuracy: **48.67%**
- Citation Rate (required only): **100.0%**
- Grounded Accuracy: **0.0%**
- SourceHit@1: **53.33%**
- SourceHit@3: **70.67%**
- SourceHit@5: **84.0%**
- Avg latency: **9080.67 ms**

## Theo category

| Category | N | EM | F1 | Routing | Citation | Grounded | Hit@1 | Hit@3 | Hit@5 | Avg ms |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| authority | 18 | 0.0% | 23.55% | 88.89% | 0.0% | 0.0% | 94.44% | 94.44% | 94.44% | 8674.44 |
| case | 13 | 0.0% | 27.18% | 38.46% | 0.0% | 0.0% | 0.0% | 0.0% | 38.46% | 9893.85 |
| edge_case | 15 | 0.0% | 20.99% | 13.33% | 0.0% | 0.0% | 0.0% | 0.0% | 13.33% | 9250.58 |
| legal | 33 | 0.0% | 22.12% | 15.15% | 100.0% | 0.0% | 15.15% | 93.94% | 93.94% | 8177.7 |
| multi_source | 15 | 0.0% | 24.74% | 0.0% | 100.0% | 0.0% | 100.0% | 100.0% | 100.0% | 8930.81 |
| procedure | 30 | 0.0% | 15.91% | 96.67% | 0.0% | 0.0% | 100.0% | 100.0% | 100.0% | 9209.69 |
| procedure_catalog | 13 | 0.0% | 13.7% | 46.15% | 0.0% | 0.0% | 100.0% | 100.0% | 100.0% | 10064.7 |
| template | 13 | 0.0% | 11.84% | 76.92% | 0.0% | 0.0% | 0.0% | 0.0% | 100.0% | 9817.22 |

## Engine confusion

| Expected -> Predicted | Count |
|---|---:|
| case -> case | 7 |
| case -> legal | 2 |
| case -> procedure | 31 |
| case -> template | 3 |
| legal -> case | 12 |
| legal -> legal | 5 |
| legal -> procedure | 16 |
| procedure -> case | 8 |
| procedure -> legal | 1 |
| procedure -> procedure | 51 |
| procedure -> template | 1 |
| template -> procedure | 3 |
| template -> template | 10 |