#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import os
import re
import statistics
import sys
import time
import unicodedata
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import requests


CATEGORY_TO_ENGINE = {
    "authority": "procedure",
    "procedure": "procedure",
    "procedure_catalog": "procedure",
    "legal": "legal",
    "template": "template",
    "case": "case",
    "edge_case": "case",
    "multi_source": "case",
}


@dataclass
class EvalRow:
    qid: str
    category: str
    question: str
    gold_answer: str
    predicted_answer: str
    exact_match: float
    token_f1: float
    expected_engine: str
    predicted_engine: str
    routing_correct: float | None
    must_have_citation: bool
    citation_present: float
    grounded_correct: float
    gold_source_kinds: list[str]
    selected_kinds: list[str]
    source_hit_at_1: float
    source_hit_at_3: float
    source_hit_at_5: float
    latency_ms: float | None
    status: str
    error: str
    answer_mode: str
    query_mode: str
    auto_intent: str
    intent: str
    selected_engine_raw: str


# ---------------------------------------------------------------------------
# Text normalization and answer matching
# ---------------------------------------------------------------------------

def normalize_text(text: Any) -> str:
    if text is None:
        return ""
    text = str(text)
    text = unicodedata.normalize("NFC", text).strip().lower()
    text = text.replace("đ", "d").replace("Đ", "d")
    text = unicodedata.normalize("NFD", text)
    text = "".join(ch for ch in text if unicodedata.category(ch) != "Mn")
    text = re.sub(r"[^a-z0-9\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def tokenize(text: Any) -> list[str]:
    norm = normalize_text(text)
    return norm.split() if norm else []


def exact_match(pred: str, golds: Iterable[str]) -> float:
    pred_n = normalize_text(pred)
    for gold in golds:
        if pred_n == normalize_text(gold):
            return 1.0
    return 0.0


def token_f1_single(pred: str, gold: str) -> float:
    pred_tokens = tokenize(pred)
    gold_tokens = tokenize(gold)
    if not pred_tokens and not gold_tokens:
        return 1.0
    if not pred_tokens or not gold_tokens:
        return 0.0

    pred_counter = Counter(pred_tokens)
    gold_counter = Counter(gold_tokens)
    overlap = sum((pred_counter & gold_counter).values())
    if overlap == 0:
        return 0.0

    precision = overlap / len(pred_tokens)
    recall = overlap / len(gold_tokens)
    return 2 * precision * recall / (precision + recall)


def best_token_f1(pred: str, golds: Iterable[str]) -> float:
    scores = [token_f1_single(pred, gold) for gold in golds]
    return max(scores) if scores else 0.0


# ---------------------------------------------------------------------------
# Dataset loading
# ---------------------------------------------------------------------------

def load_dataset(path: str) -> list[dict[str, Any]]:
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"Không tìm thấy dataset: {path}")

    if p.suffix.lower() == ".jsonl":
        rows: list[dict[str, Any]] = []
        with p.open("r", encoding="utf-8") as f:
            for line_no, line in enumerate(f, start=1):
                line = line.strip()
                if not line:
                    continue
                try:
                    rows.append(json.loads(line))
                except json.JSONDecodeError as exc:
                    raise ValueError(f"JSONL lỗi ở dòng {line_no}: {exc}") from exc
        return rows

    with p.open("r", encoding="utf-8") as f:
        data = json.load(f)

    if isinstance(data, list):
        return data
    if isinstance(data, dict) and "items" in data and isinstance(data["items"], list):
        return data["items"]
    raise ValueError("Dataset phải là list JSON hoặc JSONL mỗi dòng 1 object")


# ---------------------------------------------------------------------------
# API helpers
# ---------------------------------------------------------------------------

def build_session() -> requests.Session:
    session = requests.Session()
    session.headers.update({"Content-Type": "application/json; charset=utf-8"})
    return session


def call_api(
    session: requests.Session,
    api_base: str,
    query: str,
    *,
    per_kind: int,
    use_llm: bool,
    mode: str,
    timeout: int,
) -> tuple[dict[str, Any] | None, float | None, str]:
    started = time.perf_counter()
    try:
        resp = session.post(
            f"{api_base.rstrip('/')}/ask",
            json={
                "query": query,
                "per_kind": per_kind,
                "use_llm": use_llm,
                "mode": mode,
            },
            timeout=timeout,
        )
        resp.raise_for_status()
        payload = resp.json()
        latency_ms = (time.perf_counter() - started) * 1000.0
        return payload, latency_ms, "ok"
    except Exception as exc:
        latency_ms = (time.perf_counter() - started) * 1000.0
        return None, latency_ms, str(exc)


# ---------------------------------------------------------------------------
# Evaluation helpers
# ---------------------------------------------------------------------------

def infer_expected_engine(category: str) -> str:
    return CATEGORY_TO_ENGINE.get((category or "").strip().lower(), "case")


def normalize_engine_family(value: Any) -> str:
    if value is None:
        return ""
    text = str(value).strip().lower()
    if not text:
        return ""
    if "legal_article_lookup" in text or "legal" in text:
        return "legal"
    if "template" in text:
        return "template"
    if "case" in text:
        return "case"
    if "procedure" in text or "authority" in text:
        return "procedure"
    return ""


def extract_predicted_engine(payload: dict[str, Any]) -> tuple[str, str]:
    candidates = [
        payload.get("selected_engine"),
        payload.get("intent"),
        payload.get("query_mode"),
        payload.get("auto_intent"),
        payload.get("sub_intent"),
    ]
    for c in candidates:
        family = normalize_engine_family(c)
        if family:
            return family, "" if c is None else str(c)
    return "", ""


def has_citation(payload: dict[str, Any], answer_text: str) -> bool:
    citation_map = payload.get("citation_map")
    if isinstance(citation_map, dict) and citation_map:
        return True
    if isinstance(citation_map, list) and citation_map:
        return True
    if re.search(r"\[(?:C\d+|\d+)\]", answer_text or ""):
        return True
    return False


def flatten_selected_kinds(payload: dict[str, Any]) -> list[str]:
    ordered: list[str] = []

    selected_items = payload.get("selected_items")
    if isinstance(selected_items, dict):
        for key in ["procedure", "authority", "legal", "template", "case"]:
            items = selected_items.get(key, [])
            if not isinstance(items, list):
                continue
            for item in items:
                if isinstance(item, dict):
                    kind = normalize_engine_family(item.get("kind") or key)
                    ordered.append(kind or normalize_engine_family(key) or key)

    if ordered:
        return ordered

    results = payload.get("results")
    if isinstance(results, dict):
        for key in ["procedure", "authority", "legal", "template", "case"]:
            items = results.get(key, [])
            if not isinstance(items, list):
                continue
            for _ in items:
                ordered.append(normalize_engine_family(key) or key)
    return ordered


def source_hit_at_k(selected_kinds: list[str], gold_source_kinds: list[str], k: int) -> float:
    gold_norm = {normalize_engine_family(x) or normalize_text(x) for x in gold_source_kinds if x}
    selected_norm = [normalize_engine_family(x) or normalize_text(x) for x in selected_kinds[:k]]
    return 1.0 if any(kind in gold_norm for kind in selected_norm) else 0.0


def safe_mean(values: list[float]) -> float:
    return sum(values) / len(values) if values else 0.0


def pct(value: float) -> float:
    return round(value * 100.0, 2)


def summarize_rows(rows: list[EvalRow], mode: str, use_llm: bool, dataset_path: str, api_base: str) -> dict[str, Any]:
    overall = {
        "num_samples": len(rows),
        "num_success": sum(1 for r in rows if r.status == "ok"),
        "num_errors": sum(1 for r in rows if r.status != "ok"),
        "exact_match": pct(safe_mean([r.exact_match for r in rows])),
        "token_f1": pct(safe_mean([r.token_f1 for r in rows])),
        "citation_rate_required_only": pct(
            safe_mean([r.citation_present for r in rows if r.must_have_citation])
        ),
        "grounded_accuracy": pct(safe_mean([r.grounded_correct for r in rows])),
        "source_hit_at_1": pct(safe_mean([r.source_hit_at_1 for r in rows])),
        "source_hit_at_3": pct(safe_mean([r.source_hit_at_3 for r in rows])),
        "source_hit_at_5": pct(safe_mean([r.source_hit_at_5 for r in rows])),
        "avg_latency_ms": round(safe_mean([r.latency_ms for r in rows if r.latency_ms is not None]), 2),
        "median_latency_ms": round(statistics.median([r.latency_ms for r in rows if r.latency_ms is not None]), 2)
        if any(r.latency_ms is not None for r in rows)
        else 0.0,
    }

    routing_rows = [r for r in rows if r.routing_correct is not None]
    overall["routing_accuracy"] = pct(safe_mean([r.routing_correct or 0.0 for r in routing_rows])) if routing_rows else None

    by_category: dict[str, Any] = {}
    categories = sorted({r.category for r in rows})
    for cat in categories:
        group = [r for r in rows if r.category == cat]
        route_group = [r for r in group if r.routing_correct is not None]
        by_category[cat] = {
            "num_samples": len(group),
            "exact_match": pct(safe_mean([r.exact_match for r in group])),
            "token_f1": pct(safe_mean([r.token_f1 for r in group])),
            "citation_rate_required_only": pct(
                safe_mean([r.citation_present for r in group if r.must_have_citation])
            ),
            "grounded_accuracy": pct(safe_mean([r.grounded_correct for r in group])),
            "source_hit_at_1": pct(safe_mean([r.source_hit_at_1 for r in group])),
            "source_hit_at_3": pct(safe_mean([r.source_hit_at_3 for r in group])),
            "source_hit_at_5": pct(safe_mean([r.source_hit_at_5 for r in group])),
            "routing_accuracy": pct(safe_mean([r.routing_correct or 0.0 for r in route_group])) if route_group else None,
            "avg_latency_ms": round(safe_mean([r.latency_ms for r in group if r.latency_ms is not None]), 2),
        }

    engine_confusion = Counter()
    for r in rows:
        engine_confusion[f"{r.expected_engine} -> {r.predicted_engine or 'unknown'}"] += 1

    return {
        "meta": {
            "dataset_path": dataset_path,
            "api_base": api_base,
            "mode": mode,
            "use_llm": use_llm,
        },
        "overall": overall,
        "by_category": by_category,
        "engine_confusion": dict(engine_confusion),
    }


def to_csv_rows(rows: list[EvalRow]) -> list[dict[str, Any]]:
    return [
        {
            "qid": r.qid,
            "category": r.category,
            "question": r.question,
            "gold_answer": r.gold_answer,
            "predicted_answer": r.predicted_answer,
            "exact_match": r.exact_match,
            "token_f1": round(r.token_f1, 6),
            "expected_engine": r.expected_engine,
            "predicted_engine": r.predicted_engine,
            "routing_correct": "" if r.routing_correct is None else r.routing_correct,
            "must_have_citation": r.must_have_citation,
            "citation_present": r.citation_present,
            "grounded_correct": r.grounded_correct,
            "gold_source_kinds": "|".join(r.gold_source_kinds),
            "selected_kinds": "|".join(r.selected_kinds),
            "source_hit_at_1": r.source_hit_at_1,
            "source_hit_at_3": r.source_hit_at_3,
            "source_hit_at_5": r.source_hit_at_5,
            "latency_ms": "" if r.latency_ms is None else round(r.latency_ms, 2),
            "status": r.status,
            "error": r.error,
            "answer_mode": r.answer_mode,
            "query_mode": r.query_mode,
            "auto_intent": r.auto_intent,
            "intent": r.intent,
            "selected_engine_raw": r.selected_engine_raw,
        }
        for r in rows
    ]


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def write_markdown_summary(path: Path, summary: dict[str, Any]) -> None:
    overall = summary["overall"]
    lines = [
        "# Kết quả đánh giá RAG/Engine",
        "",
        f"- Dataset: `{summary['meta']['dataset_path']}`",
        f"- API Base: `{summary['meta']['api_base']}`",
        f"- Mode: `{summary['meta']['mode']}`",
        f"- Use LLM: `{summary['meta']['use_llm']}`",
        "",
        "## Tổng quan",
        "",
        f"- Số mẫu: **{overall['num_samples']}**",
        f"- Thành công: **{overall['num_success']}**",
        f"- Lỗi: **{overall['num_errors']}**",
        f"- Exact Match: **{overall['exact_match']}%**",
        f"- Token F1: **{overall['token_f1']}%**",
        f"- Routing Accuracy: **{overall['routing_accuracy']}%**" if overall.get("routing_accuracy") is not None else "- Routing Accuracy: **N/A**",
        f"- Citation Rate (required only): **{overall['citation_rate_required_only']}%**",
        f"- Grounded Accuracy: **{overall['grounded_accuracy']}%**",
        f"- SourceHit@1: **{overall['source_hit_at_1']}%**",
        f"- SourceHit@3: **{overall['source_hit_at_3']}%**",
        f"- SourceHit@5: **{overall['source_hit_at_5']}%**",
        f"- Avg latency: **{overall['avg_latency_ms']} ms**",
        "",
        "## Theo category",
        "",
        "| Category | N | EM | F1 | Routing | Citation | Grounded | Hit@1 | Hit@3 | Hit@5 | Avg ms |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]

    for cat, stats in summary["by_category"].items():
        routing = "N/A" if stats["routing_accuracy"] is None else f"{stats['routing_accuracy']}%"
        lines.append(
            f"| {cat} | {stats['num_samples']} | {stats['exact_match']}% | {stats['token_f1']}% | {routing} | {stats['citation_rate_required_only']}% | {stats['grounded_accuracy']}% | {stats['source_hit_at_1']}% | {stats['source_hit_at_3']}% | {stats['source_hit_at_5']}% | {stats['avg_latency_ms']} |"
        )

    lines.extend([
        "",
        "## Engine confusion",
        "",
        "| Expected -> Predicted | Count |",
        "|---|---:|",
    ])
    for key, count in sorted(summary["engine_confusion"].items()):
        lines.append(f"| {key} | {count} |")

    path.write_text("\n".join(lines), encoding="utf-8")


# ---------------------------------------------------------------------------
# Main evaluation loop
# ---------------------------------------------------------------------------

def evaluate(args: argparse.Namespace) -> tuple[list[EvalRow], dict[str, Any]]:
    dataset = load_dataset(args.dataset)
    if args.offset:
        dataset = dataset[args.offset :]
    if args.limit is not None:
        dataset = dataset[: args.limit]

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    session = build_session()
    rows: list[EvalRow] = []

    print(f"[INFO] Loaded {len(dataset)} samples from {args.dataset}")
    print(f"[INFO] API base: {args.api_base}")
    print(f"[INFO] mode={args.mode} use_llm={args.use_llm} per_kind={args.per_kind}")

    for idx, sample in enumerate(dataset, start=1):
        qid = str(sample.get("qid", f"row_{idx:04d}"))
        category = str(sample.get("category", "unknown"))
        question = str(sample.get("question", "")).strip()
        gold_answer = str(sample.get("gold_answer", "")).strip()
        accepted_answers = sample.get("accepted_answers") or []
        if not isinstance(accepted_answers, list):
            accepted_answers = [str(accepted_answers)]
        if gold_answer and gold_answer not in accepted_answers:
            accepted_answers = [gold_answer, *accepted_answers]
        must_have_citation = bool(sample.get("must_have_citation", False))
        gold_source_kinds = sample.get("gold_source_kind_hint") or []
        if not isinstance(gold_source_kinds, list):
            gold_source_kinds = [str(gold_source_kinds)]

        payload, latency_ms, error = call_api(
            session,
            args.api_base,
            question,
            per_kind=args.per_kind,
            use_llm=args.use_llm,
            mode=args.mode,
            timeout=args.timeout,
        )

        if payload is None:
            row = EvalRow(
                qid=qid,
                category=category,
                question=question,
                gold_answer=gold_answer,
                predicted_answer="",
                exact_match=0.0,
                token_f1=0.0,
                expected_engine=infer_expected_engine(category),
                predicted_engine="",
                routing_correct=0.0 if args.mode == "auto" else None,
                must_have_citation=must_have_citation,
                citation_present=0.0,
                grounded_correct=0.0,
                gold_source_kinds=gold_source_kinds,
                selected_kinds=[],
                source_hit_at_1=0.0,
                source_hit_at_3=0.0,
                source_hit_at_5=0.0,
                latency_ms=latency_ms,
                status="error",
                error=error,
                answer_mode="",
                query_mode="",
                auto_intent="",
                intent="",
                selected_engine_raw="",
            )
            rows.append(row)
            print(f"[{idx}/{len(dataset)}] {qid} ERROR: {error}")
            if args.stop_on_error:
                break
            if args.sleep_ms:
                time.sleep(args.sleep_ms / 1000.0)
            continue

        answer_text = str(payload.get("answer_text", "")).strip()
        em = exact_match(answer_text, accepted_answers)
        f1 = best_token_f1(answer_text, accepted_answers)
        predicted_engine, selected_engine_raw = extract_predicted_engine(payload)
        expected_engine = infer_expected_engine(category)
        routing_correct = None if args.mode != "auto" else float(predicted_engine == expected_engine)
        citation_present = float(has_citation(payload, answer_text))
        grounded_correct = float(em == 1.0 and (not must_have_citation or citation_present == 1.0))
        selected_kinds = flatten_selected_kinds(payload)
        hit1 = source_hit_at_k(selected_kinds, gold_source_kinds, 1)
        hit3 = source_hit_at_k(selected_kinds, gold_source_kinds, 3)
        hit5 = source_hit_at_k(selected_kinds, gold_source_kinds, 5)

        row = EvalRow(
            qid=qid,
            category=category,
            question=question,
            gold_answer=gold_answer,
            predicted_answer=answer_text,
            exact_match=em,
            token_f1=f1,
            expected_engine=expected_engine,
            predicted_engine=predicted_engine,
            routing_correct=routing_correct,
            must_have_citation=must_have_citation,
            citation_present=citation_present,
            grounded_correct=grounded_correct,
            gold_source_kinds=gold_source_kinds,
            selected_kinds=selected_kinds,
            source_hit_at_1=hit1,
            source_hit_at_3=hit3,
            source_hit_at_5=hit5,
            latency_ms=latency_ms,
            status="ok",
            error="",
            answer_mode=str(payload.get("answer_mode", "")),
            query_mode=str(payload.get("query_mode", "")),
            auto_intent=str(payload.get("auto_intent", "")),
            intent=str(payload.get("intent", "")),
            selected_engine_raw=selected_engine_raw,
        )
        rows.append(row)

        print(
            f"[{idx}/{len(dataset)}] {qid} | cat={category} | em={int(em)} | f1={f1:.3f} | "
            f"route={row.predicted_engine or 'unknown'} | hit@5={int(hit5)} | {latency_ms:.1f} ms"
        )

        if args.sleep_ms:
            time.sleep(args.sleep_ms / 1000.0)

    summary = summarize_rows(
        rows,
        mode=args.mode,
        use_llm=args.use_llm,
        dataset_path=args.dataset,
        api_base=args.api_base,
    )
    return rows, summary


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Đánh giá hệ thống RAG/engine qua API /ask")
    parser.add_argument("--dataset", default="rag_eval_normalized.jsonl", help="Đường dẫn file JSON hoặc JSONL")
    parser.add_argument("--api-base", default="http://127.0.0.1:8000", help="API base URL")
    parser.add_argument("--out-dir", default="eval_outputs", help="Thư mục xuất kết quả")
    parser.add_argument("--mode", default="auto", choices=["auto", "legal", "procedure", "template", "case"], help="Mode gửi lên API")
    parser.add_argument("--per-kind", type=int, default=3, help="per_kind gửi lên API")
    parser.add_argument("--use-llm", dest="use_llm", action="store_true", help="Bật LLM")
    parser.add_argument("--no-llm", dest="use_llm", action="store_false", help="Tắt LLM")
    parser.set_defaults(use_llm=True)
    parser.add_argument("--timeout", type=int, default=120, help="Timeout cho mỗi request")
    parser.add_argument("--limit", type=int, default=None, help="Chỉ chạy N mẫu đầu tiên")
    parser.add_argument("--offset", type=int, default=0, help="Bỏ qua N mẫu đầu tiên")
    parser.add_argument("--sleep-ms", type=int, default=0, help="Ngủ giữa các request")
    parser.add_argument("--stop-on-error", action="store_true", help="Dừng ngay khi gặp lỗi")
    return parser


def main() -> int:
    parser = build_arg_parser()
    args = parser.parse_args()

    rows, summary = evaluate(args)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    results_csv = out_dir / "results.csv"
    summary_json = out_dir / "summary.json"
    summary_md = out_dir / "summary.md"

    write_csv(results_csv, to_csv_rows(rows))
    summary_json.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    write_markdown_summary(summary_md, summary)

    print("\n[INFO] Saved:")
    print(f"- {results_csv}")
    print(f"- {summary_json}")
    print(f"- {summary_md}")
    print("\n[INFO] Overall summary:")
    print(json.dumps(summary["overall"], ensure_ascii=False, indent=2))

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
