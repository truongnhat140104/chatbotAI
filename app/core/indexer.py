from __future__ import annotations

import json
import re
import unicodedata
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from app.core.loader import HotichBundle, HotichLoader


@dataclass
class IndexBuildResult:
    output_dir: Path
    counts: dict[str, int]

    def print_summary(self) -> None:
        print("=" * 80)
        print("HOTICH INDEXER SUMMARY")
        print("=" * 80)
        print(f"Output dir: {self.output_dir}")
        for k, v in self.counts.items():
            print(f"{k:20}: {v}")
        print("=" * 80)


class HotichIndexer:
    """
    Tạo retrieval units từ bundle đã load.

    Output chính:
    - 05_index/legal_units.jsonl
    - 05_index/procedure_units.jsonl
    - 05_index/template_units.jsonl
    - 05_index/case_units.jsonl
    - 05_index/authority_units.jsonl
    - 05_index/all_units.jsonl
    - 05_index/manifest.json
    """

    def __init__(self, bundle: HotichBundle, output_dir: str | Path | None = None) -> None:
        self.bundle = bundle
        self.output_dir = (
            Path(output_dir).resolve()
            if output_dir
            else (bundle.data_root / "05_index").resolve()
        )

    # ------------------------------------------------------------------
    # Text helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _text(value: Any) -> str:
        if value is None:
            return ""
        if isinstance(value, str):
            return value.strip()
        if isinstance(value, (int, float, bool)):
            return str(value)
        return ""

    @classmethod
    def _normalize_text(cls, text: Any) -> str:
        text = cls._text(text)
        if not text:
            return ""

        text = text.lower().strip()
        text = unicodedata.normalize("NFD", text)
        text = "".join(ch for ch in text if unicodedata.category(ch) != "Mn")
        text = text.replace("đ", "d").replace("Đ", "d")
        text = re.sub(r"[^a-z0-9\s]", " ", text)
        text = re.sub(r"\s+", " ", text).strip()
        return text

    @classmethod
    def _slug(cls, text: Any) -> str:
        norm = cls._normalize_text(text)
        if not norm:
            return "na"
        return norm.replace(" ", "_")

    @staticmethod
    def _join_nonempty(parts: list[str], sep: str = " | ") -> str:
        return sep.join([p for p in parts if p])

    def _flatten_text(self, value: Any, limit: int = 3000) -> str:
        parts: list[str] = []

        def walk(x: Any) -> None:
            if len(" ".join(parts)) > limit:
                return

            if x is None:
                return
            if isinstance(x, str):
                s = x.strip()
                if s:
                    parts.append(s)
                return
            if isinstance(x, (int, float, bool)):
                parts.append(str(x))
                return
            if isinstance(x, list):
                for item in x:
                    walk(item)
                return
            if isinstance(x, dict):
                for v in x.values():
                    walk(v)

        walk(value)
        return " ".join(parts)[:limit].strip()

    # ------------------------------------------------------------------
    # Meta resolution
    # ------------------------------------------------------------------

    def _resolve_doc_meta(self, doc_id: str) -> dict[str, Any]:
        meta = self.bundle.meta.get(doc_id, {})
        doc = meta.get("doc", {}) if isinstance(meta, dict) else {}
        return doc if isinstance(doc, dict) else {}

    def _resolve_doc_title(self, doc_id: str, fallback: str = "") -> str:
        doc = self._resolve_doc_meta(doc_id)
        title = self._text(doc.get("title"))
        return title or fallback or doc_id

    def _resolve_doc_number(self, doc_id: str) -> str:
        doc = self._resolve_doc_meta(doc_id)
        return self._text(doc.get("number"))

    def _resolve_doc_kind(self, doc_id: str) -> str:
        doc = self._resolve_doc_meta(doc_id)
        return self._text(doc.get("doc_kind"))

    def _iter_legal_articles(
        self,
        chapter: dict[str, Any],
    ) -> list[tuple[dict[str, Any], str, str]]:
        article_rows: list[tuple[dict[str, Any], str, str]] = []

        chapter_articles = chapter.get("articles", []) if isinstance(chapter, dict) else []
        for article in chapter_articles:
            if isinstance(article, dict):
                article_rows.append((article, "", ""))

        sections = chapter.get("sections", []) if isinstance(chapter, dict) else []
        for section in sections:
            if not isinstance(section, dict):
                continue
            section_no = self._text(section.get("section_no"))
            section_title = self._text(section.get("title"))
            section_articles = section.get("articles", [])
            if not isinstance(section_articles, list):
                continue
            for article in section_articles:
                if isinstance(article, dict):
                    article_rows.append((article, section_no, section_title))

        return article_rows

    # ------------------------------------------------------------------
    # Unit builders
    # ------------------------------------------------------------------

    def _make_base_unit(
        self,
        *,
        unit_id: str,
        source_kind: str,
        unit_kind: str,
        title: str,
        text: str,
        source_path: str,
        module: str = "hotich",
        extra: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        extra = extra or {}

        lexical_text = self._join_nonempty(
            [
                title,
                text,
                extra.get("citation_label", ""),
                extra.get("doc_title", ""),
                extra.get("doc_number", ""),
                extra.get("article_title", ""),
                extra.get("section_title", ""),
                extra.get("chapter_title", ""),
                extra.get("tags_text", ""),
            ],
            sep=" ",
        )

        embedding_text = self._join_nonempty(
            [
                f"[{source_kind}/{unit_kind}]",
                extra.get("doc_title", ""),
                extra.get("doc_number", ""),
                extra.get("chapter_title", ""),
                extra.get("section_title", ""),
                extra.get("citation_label", ""),
                title,
                text,
            ],
            sep="\n",
        )

        unit = {
            "unit_id": unit_id,
            "source_kind": source_kind,
            "unit_kind": unit_kind,
            "module": module,
            "title": title,
            "text": text,
            "source_path": source_path,
            "lexical_text": lexical_text,
            "embedding_text": embedding_text,
        }
        unit.update(extra)
        return unit

    def build_legal_units(self) -> list[dict[str, Any]]:
        units: list[dict[str, Any]] = []

        for doc_id, item in self.bundle.legal_docs.items():
            raw = item.get("raw", {})
            chapters = raw.get("chapters", []) if isinstance(raw, dict) else []

            doc_title = self._resolve_doc_title(doc_id, fallback=item.get("title", doc_id))
            doc_number = self._resolve_doc_number(doc_id)
            doc_kind = self._resolve_doc_kind(doc_id)

            for chapter in chapters:
                chapter_no = self._text(chapter.get("chapter_no"))
                chapter_title = self._text(chapter.get("title"))

                for article, section_no, section_title in self._iter_legal_articles(chapter):
                    article_no = self._text(article.get("article_no"))
                    article_title = self._text(article.get("title"))
                    raw_paragraphs = article.get("raw_paragraphs", []) if isinstance(article, dict) else []
                    clauses = article.get("clauses", []) if isinstance(article, dict) else []

                    # Article-level unit (useful when article has raw_paragraphs or overview text)
                    article_text_parts: list[str] = []
                    if article_title:
                        article_text_parts.append(article_title)
                    if isinstance(raw_paragraphs, list):
                        article_text_parts.extend([self._text(x) for x in raw_paragraphs if self._text(x)])

                    article_text = " ".join(article_text_parts).strip()
                    if article_text:
                        citation_label = f"Điều {article_no}"
                        if doc_number:
                            citation_label += f" {doc_number}"

                        units.append(
                            self._make_base_unit(
                                unit_id=f"{doc_id}__art_{article_no}",
                                source_kind="legal",
                                unit_kind="article",
                                title=article_title or f"Điều {article_no}",
                                text=article_text,
                                source_path=item.get("source_path", ""),
                                extra={
                                    "doc_id": doc_id,
                                    "doc_title": doc_title,
                                    "doc_number": doc_number,
                                    "doc_kind": doc_kind,
                                    "chapter_no": chapter_no,
                                    "chapter_title": chapter_title,
                                    "section_no": section_no,
                                    "section_title": section_title,
                                    "article_no": article_no,
                                    "article_title": article_title,
                                    "clause_key": "",
                                    "point_key": "",
                                    "citation_label": citation_label,
                                },
                            )
                        )

                    # Clause-level units
                    for clause in clauses:
                        clause_key = self._text(clause.get("clause_key"))
                        clause_text = self._text(clause.get("text"))
                        raw_lines = clause.get("raw_lines", []) if isinstance(clause, dict) else []
                        clause_extra_lines = [self._text(x) for x in raw_lines if self._text(x)]
                        clause_full_text = " ".join(
                            [x for x in [clause_text, *clause_extra_lines] if x]
                        ).strip()

                        if clause_full_text:
                            citation_label = f"Điều {article_no}"
                            if clause_key:
                                citation_label += f" khoản {clause_key}"
                            if doc_number:
                                citation_label += f" {doc_number}"

                            units.append(
                                self._make_base_unit(
                                    unit_id=f"{doc_id}__art_{article_no}__cl_{self._slug(clause_key)}",
                                    source_kind="legal",
                                    unit_kind="clause",
                                    title=article_title or f"Điều {article_no}",
                                    text=clause_full_text,
                                    source_path=item.get("source_path", ""),
                                    extra={
                                        "doc_id": doc_id,
                                        "doc_title": doc_title,
                                        "doc_number": doc_number,
                                        "doc_kind": doc_kind,
                                        "chapter_no": chapter_no,
                                        "chapter_title": chapter_title,
                                        "section_no": section_no,
                                        "section_title": section_title,
                                        "article_no": article_no,
                                        "article_title": article_title,
                                        "clause_key": clause_key,
                                        "point_key": "",
                                        "citation_label": citation_label,
                                    },
                                )
                            )

                        # Point-level units
                        points = clause.get("points", []) if isinstance(clause, dict) else []
                        for point in points:
                            point_key = self._text(point.get("point_key"))
                            point_text = self._text(point.get("text"))

                            if not point_text:
                                continue

                            parent_context = self._join_nonempty(
                                [
                                    article_title,
                                    clause_text,
                                ],
                                sep=" | ",
                            )

                            point_full_text = self._join_nonempty(
                                [parent_context, point_text],
                                sep="\n",
                            )

                            citation_label = f"Điều {article_no}"
                            if clause_key:
                                citation_label += f" khoản {clause_key}"
                            if point_key:
                                citation_label += f" điểm {point_key}"
                            if doc_number:
                                citation_label += f" {doc_number}"

                            units.append(
                                self._make_base_unit(
                                    unit_id=(
                                        f"{doc_id}__art_{article_no}"
                                        f"__cl_{self._slug(clause_key)}"
                                        f"__pt_{self._slug(point_key)}"
                                    ),
                                    source_kind="legal",
                                    unit_kind="point",
                                    title=article_title or f"Điều {article_no}",
                                    text=point_full_text,
                                    source_path=item.get("source_path", ""),
                                    extra={
                                        "doc_id": doc_id,
                                        "doc_title": doc_title,
                                        "doc_number": doc_number,
                                        "doc_kind": doc_kind,
                                        "chapter_no": chapter_no,
                                        "chapter_title": chapter_title,
                                        "section_no": section_no,
                                        "section_title": section_title,
                                        "article_no": article_no,
                                        "article_title": article_title,
                                        "clause_key": clause_key,
                                        "point_key": point_key,
                                        "citation_label": citation_label,
                                    },
                                )
                            )

        return units

    def build_procedure_units(self) -> list[dict[str, Any]]:
        units: list[dict[str, Any]] = []

        for procedure_id, item in self.bundle.procedures.items():
            tags = item.get("tags", []) if isinstance(item.get("tags"), list) else []
            tags_text = " ".join(str(x) for x in tags)

            text = self._join_nonempty(
                [
                    item.get("title", ""),
                    self._flatten_text(item.get("authority", {}), limit=700),
                    self._flatten_text(item.get("processing_time", ""), limit=400),
                    self._flatten_text(item.get("fees", ""), limit=400),
                    self._flatten_text(item.get("submission_methods", []), limit=500),
                    self._flatten_text(item.get("citations", []), limit=800),
                    self._flatten_text(item.get("common_cases_notes", ""), limit=600),
                    self._flatten_text(item.get("raw", {}), limit=2500),
                ],
                sep="\n",
            )

            units.append(
                self._make_base_unit(
                    unit_id=procedure_id,
                    source_kind="procedure",
                    unit_kind="record",
                    title=item.get("title", procedure_id),
                    text=text,
                    source_path=item.get("source_path", ""),
                    extra={
                        "procedure_id": procedure_id,
                        "tags": tags,
                        "tags_text": tags_text,
                        "citation_label": item.get("title", procedure_id),
                    },
                )
            )

        return units

    def build_template_units(self) -> list[dict[str, Any]]:
        units: list[dict[str, Any]] = []

        for template_id, item in self.bundle.templates.items():
            text = self._join_nonempty(
                [
                    item.get("title", ""),
                    self._text(item.get("loai")),
                    self._text(item.get("output_type")),
                    self._flatten_text(item.get("fields", []), limit=1600),
                    self._flatten_text(item.get("render", {}), limit=1000),
                    self._flatten_text(item.get("notes", ""), limit=500),
                ],
                sep="\n",
            )

            units.append(
                self._make_base_unit(
                    unit_id=template_id,
                    source_kind="template",
                    unit_kind="record",
                    title=item.get("title", template_id),
                    text=text,
                    source_path=item.get("source_path", ""),
                    extra={
                        "template_id": template_id,
                        "citation_label": item.get("title", template_id),
                    },
                )
            )

        return units

    def build_case_units(self) -> list[dict[str, Any]]:
        units: list[dict[str, Any]] = []

        for case_id, item in self.bundle.cases.items():
            tags = item.get("topic_tags", []) if isinstance(item.get("topic_tags"), list) else []
            tags_text = " ".join(str(x) for x in tags)

            text = self._join_nonempty(
                [
                    item.get("title", ""),
                    tags_text,
                    self._flatten_text(item.get("assumptions", ""), limit=500),
                    self._flatten_text(item.get("inputs_schema", {}), limit=1000),
                    self._flatten_text(item.get("legal_basis", []), limit=1200),
                    self._flatten_text(item.get("decision_points", []), limit=1000),
                    self._flatten_text(item.get("steps", []), limit=1500),
                    self._flatten_text(item.get("outputs", {}), limit=900),
                ],
                sep="\n",
            )

            units.append(
                self._make_base_unit(
                    unit_id=case_id,
                    source_kind="case",
                    unit_kind="record",
                    title=item.get("title", case_id),
                    text=text,
                    source_path=item.get("source_path", ""),
                    extra={
                        "case_id": case_id,
                        "tags": tags,
                        "tags_text": tags_text,
                        "citation_label": item.get("title", case_id),
                    },
                )
            )

        return units

    def build_authority_units(self) -> list[dict[str, Any]]:
        units: list[dict[str, Any]] = []

        for auth_id, item in self.bundle.authority_rules.items():
            rules = item.get("rules", [])
            if not isinstance(rules, list):
                rules = [rules]

            for idx, rule in enumerate(rules, start=1):
                text = self._join_nonempty(
                    [
                        self._flatten_text(rule, limit=1500),
                    ],
                    sep="\n",
                )
                if not text:
                    continue

                units.append(
                    self._make_base_unit(
                        unit_id=f"{auth_id}__rule_{idx}",
                        source_kind="authority",
                        unit_kind="rule",
                        title=f"Authority rule {idx}",
                        text=text,
                        source_path=item.get("source_path", ""),
                        extra={
                            "authority_id": auth_id,
                            "rule_index": idx,
                            "citation_label": f"Authority rule {idx}",
                        },
                    )
                )

        return units

    # ------------------------------------------------------------------
    # Save helpers
    # ------------------------------------------------------------------

    def _write_jsonl(self, path: Path, rows: list[dict[str, Any]]) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", encoding="utf-8") as f:
            for row in rows:
                f.write(json.dumps(row, ensure_ascii=False) + "\n")

    def _write_json(self, path: Path, obj: Any) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", encoding="utf-8") as f:
            json.dump(obj, f, ensure_ascii=False, indent=2)

    # ------------------------------------------------------------------
    # Public build
    # ------------------------------------------------------------------

    def build_all(self) -> IndexBuildResult:
        legal_units = self.build_legal_units()
        procedure_units = self.build_procedure_units()
        template_units = self.build_template_units()
        case_units = self.build_case_units()
        authority_units = self.build_authority_units()

        all_units = [
            *legal_units,
            *procedure_units,
            *template_units,
            *case_units,
            *authority_units,
        ]

        self.output_dir.mkdir(parents=True, exist_ok=True)

        self._write_jsonl(self.output_dir / "legal_units.jsonl", legal_units)
        self._write_jsonl(self.output_dir / "procedure_units.jsonl", procedure_units)
        self._write_jsonl(self.output_dir / "template_units.jsonl", template_units)
        self._write_jsonl(self.output_dir / "case_units.jsonl", case_units)
        self._write_jsonl(self.output_dir / "authority_units.jsonl", authority_units)
        self._write_jsonl(self.output_dir / "all_units.jsonl", all_units)

        by_source_kind = Counter(unit["source_kind"] for unit in all_units)
        by_unit_kind = Counter(unit["unit_kind"] for unit in all_units)

        manifest = {
            "output_dir": str(self.output_dir),
            "counts": {
                "legal_units": len(legal_units),
                "procedure_units": len(procedure_units),
                "template_units": len(template_units),
                "case_units": len(case_units),
                "authority_units": len(authority_units),
                "all_units": len(all_units),
            },
            "by_source_kind": dict(by_source_kind),
            "by_unit_kind": dict(by_unit_kind),
        }
        self._write_json(self.output_dir / "manifest.json", manifest)

        return IndexBuildResult(
            output_dir=self.output_dir,
            counts=manifest["counts"],
        )


def main() -> None:
    loader = HotichLoader()
    bundle = loader.load_all()

    indexer = HotichIndexer(bundle)
    result = indexer.build_all()
    result.print_summary()

    print("\nDa tao cac file index tai:")
    print(f"- {result.output_dir / 'legal_units.jsonl'}")
    print(f"- {result.output_dir / 'procedure_units.jsonl'}")
    print(f"- {result.output_dir / 'template_units.jsonl'}")
    print(f"- {result.output_dir / 'case_units.jsonl'}")
    print(f"- {result.output_dir / 'authority_units.jsonl'}")
    print(f"- {result.output_dir / 'all_units.jsonl'}")
    print(f"- {result.output_dir / 'manifest.json'}")


if __name__ == "__main__":
    main()
