from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml


SUPPORTED_EXTS = {".json", ".yaml", ".yml"}


@dataclass
class LoadMessage:
    level: str   # INFO | WARNING | ERROR
    path: str
    message: str


@dataclass
class HotichBundle:
    data_root: Path
    registry: dict[str, Any] | None = None
    meta: dict[str, dict[str, Any]] = field(default_factory=dict)
    legal_docs: dict[str, dict[str, Any]] = field(default_factory=dict)
    procedures: dict[str, dict[str, Any]] = field(default_factory=dict)
    authority_rules: dict[str, dict[str, Any]] = field(default_factory=dict)
    templates: dict[str, dict[str, Any]] = field(default_factory=dict)
    cases: dict[str, dict[str, Any]] = field(default_factory=dict)
    procedure_catalog: dict[str, dict[str, Any]] = field(default_factory=dict)
    qa: dict[str, dict[str, Any]] = field(default_factory=dict)
    messages: list[LoadMessage] = field(default_factory=list)

    def add_message(self, level: str, path: Path | str, message: str) -> None:
        self.messages.append(LoadMessage(level=level, path=str(path), message=message))

    @property
    def error_count(self) -> int:
        return sum(1 for m in self.messages if m.level == "ERROR")

    @property
    def warning_count(self) -> int:
        return sum(1 for m in self.messages if m.level == "WARNING")

    @property
    def info_count(self) -> int:
        return sum(1 for m in self.messages if m.level == "INFO")

    def summary(self) -> dict[str, Any]:
        return {
            "data_root": str(self.data_root),
            "registry": 1 if self.registry else 0,
            "meta": len(self.meta),
            "legal_docs": len(self.legal_docs),
            "procedures": len(self.procedures),
            "authority_rules": len(self.authority_rules),
            "templates": len(self.templates),
            "cases": len(self.cases),
            "procedure_catalog": len(self.procedure_catalog),
            "qa": len(self.qa),
            "errors": self.error_count,
            "warnings": self.warning_count,
            "infos": self.info_count,
        }

    def print_summary(self) -> None:
        s = self.summary()
        print("=" * 80)
        print("HOTICH LOADER SUMMARY")
        print("=" * 80)
        print(f"Data root          : {s['data_root']}")
        print(f"Registry           : {s['registry']}")
        print(f"Meta               : {s['meta']}")
        print(f"Legal docs         : {s['legal_docs']}")
        print(f"Procedures         : {s['procedures']}")
        print(f"Authority rules    : {s['authority_rules']}")
        print(f"Templates          : {s['templates']}")
        print(f"Cases              : {s['cases']}")
        print(f"Procedure catalog  : {s['procedure_catalog']}")
        print(f"QA                 : {s['qa']}")
        print(f"Errors             : {s['errors']}")
        print(f"Warnings           : {s['warnings']}")
        print(f"Infos              : {s['infos']}")
        print("=" * 80)

    def print_messages(self, level: str | None = None, limit: int | None = None) -> None:
        rows = self.messages
        if level:
            rows = [m for m in rows if m.level == level.upper()]
        if limit is not None:
            rows = rows[:limit]

        if not rows:
            print("Khong co message nao.")
            return

        print("-" * 80)
        for m in rows:
            print(f"[{m.level}] {m.path}")
            print(f"  -> {m.message}")
        print("-" * 80)


class HotichLoader:
    """
    Loader cho bo du lieu Data/hotich.
    - Doc JSON/YAML
    - Validate field bat buoc
    - Normalize thanh format noi bo thong nhat
    - Ghi WARNING/ERROR thay vi ep moi file giong het nhau
    """

    def __init__(self, data_root: str | Path | None = None) -> None:
        self.data_root = self._detect_data_root(data_root)

    def _detect_data_root(self, explicit_root: str | Path | None) -> Path:
        if explicit_root is not None:
            root = Path(explicit_root).resolve()
            if not root.exists():
                raise FileNotFoundError(f"Khong tim thay data_root: {root}")
            return root

        here = Path(__file__).resolve()

        # Thu mot so vi tri pho bien
        candidates = [
            here.parents[2] / "Data" / "hotich",  # app/core/loader.py -> project/Data/hotich
            here.parents[1] / "Data" / "hotich",
            Path.cwd() / "Data" / "hotich",
        ]

        for candidate in candidates:
            if candidate.exists():
                return candidate.resolve()

        raise FileNotFoundError(
            "Khong tu dong tim thay Data/hotich. "
            "Hay truyen data_root khi khoi tao HotichLoader."
        )

    # -------------------------------------------------------------------------
    # File helpers
    # -------------------------------------------------------------------------

    @staticmethod
    def _load_file(path: Path) -> Any:
        suffix = path.suffix.lower()
        with path.open("r", encoding="utf-8") as f:
            if suffix == ".json":
                return json.load(f)
            if suffix in {".yaml", ".yml"}:
                return yaml.safe_load(f)
        raise ValueError(f"Unsupported file type: {path}")

    @staticmethod
    def _iter_supported_files(folder: Path) -> list[Path]:
        if not folder.exists():
            return []
        return sorted(
            p for p in folder.iterdir()
            if p.is_file() and p.suffix.lower() in SUPPORTED_EXTS
        )

    @staticmethod
    def _require_fields(
        data: dict[str, Any],
        required_fields: list[str],
    ) -> list[str]:
        missing = []
        for field_name in required_fields:
            if field_name not in data:
                missing.append(field_name)
        return missing

    @staticmethod
    def _soft_expect_type(
        data: dict[str, Any],
        field_name: str,
        expected_type: type | tuple[type, ...],
    ) -> str | None:
        if field_name not in data:
            return None
        value = data[field_name]
        if not isinstance(value, expected_type):
            expected_name = (
                ", ".join(t.__name__ for t in expected_type)
                if isinstance(expected_type, tuple)
                else expected_type.__name__
            )
            return (
                f"Field '{field_name}' nen co kieu {expected_name}, "
                f"nhung dang la {type(value).__name__}"
            )
        return None

    @staticmethod
    def _normalize_text(value: Any) -> str:
        if value is None:
            return ""
        if isinstance(value, str):
            return value.strip()
        return str(value).strip()

    def _read_folder_as_objects(
        self,
        folder: Path,
        bundle: HotichBundle,
    ) -> list[tuple[Path, dict[str, Any]]]:
        items: list[tuple[Path, dict[str, Any]]] = []

        for path in self._iter_supported_files(folder):
            try:
                raw = self._load_file(path)
            except Exception as exc:
                bundle.add_message("ERROR", path, f"Khong doc duoc file: {exc}")
                continue

            if not isinstance(raw, dict):
                bundle.add_message(
                    "ERROR",
                    path,
                    f"Root phai la object/dict, nhung dang la {type(raw).__name__}",
                )
                continue

            items.append((path, raw))

        return items

    # -------------------------------------------------------------------------
    # Normalizers
    # -------------------------------------------------------------------------

    def _normalize_meta(self, path: Path, raw: dict[str, Any], bundle: HotichBundle) -> dict[str, Any] | None:
        required = ["schema_version", "module", "doc", "relations", "sources", "files"]
        missing = self._require_fields(raw, required)
        if missing:
            bundle.add_message("ERROR", path, f"02_meta thieu field bat buoc: {missing}")
            return None

        doc = raw.get("doc", {})
        if not isinstance(doc, dict):
            bundle.add_message("ERROR", path, "Field 'doc' phai la object")
            return None

        doc_id = doc.get("doc_id") or path.stem
        title = doc.get("title", "")
        doc_kind = doc.get("doc_kind", "")

        return {
            "id": doc_id,
            "type": "meta",
            "title": self._normalize_text(title),
            "doc_kind": self._normalize_text(doc_kind),
            "module": raw.get("module"),
            "doc": doc,
            "relations": raw.get("relations", {}),
            "sources": raw.get("sources", {}),
            "files": raw.get("files", {}),
            "raw": raw,
            "source_path": str(path),
        }

    def _normalize_legal(self, path: Path, raw: dict[str, Any], bundle: HotichBundle) -> dict[str, Any] | None:
        required = ["schema_version", "module", "doc_id", "type", "source", "chapters"]
        missing = self._require_fields(raw, required)
        if missing:
            bundle.add_message("ERROR", path, f"legal thieu field bat buoc: {missing}")
            return None

        soft_checks = [
            self._soft_expect_type(raw, "chapters", list),
        ]
        for msg in soft_checks:
            if msg:
                bundle.add_message("WARNING", path, msg)

        doc_id = raw.get("doc_id", path.stem)
        doc_type = raw.get("type", "")
        source = raw.get("source")
        notes = raw.get("notes")
        parse_mode = raw.get("parse_mode", "")

        title = ""
        if isinstance(source, dict):
            title = self._normalize_text(source.get("title"))
        elif isinstance(source, str):
            title = source

        return {
            "id": doc_id,
            "type": "legal",
            "title": title,
            "doc_type": doc_type,
            "module": raw.get("module"),
            "parse_mode": parse_mode,
            "source": source,
            "notes": notes,
            "chapters": raw.get("chapters", []),
            "raw": raw,
            "source_path": str(path),
        }

    def _normalize_procedure(self, path: Path, raw: dict[str, Any], bundle: HotichBundle) -> dict[str, Any] | None:
        required = [
            "schema",
            "module",
            "procedure_id",
            "name",
            "authority",
            "processing_time",
            "fees",
            "submission_methods",
            "citations",
            "effectivity",
            "tags",
            "variants",
        ]
        missing = self._require_fields(raw, required)
        if missing:
            bundle.add_message("ERROR", path, f"procedure thieu field bat buoc: {missing}")
            return None

        soft_checks = [
            self._soft_expect_type(raw, "authority", dict),
            self._soft_expect_type(raw, "submission_methods", (list, dict)),
            self._soft_expect_type(raw, "citations", (list, dict)),
            self._soft_expect_type(raw, "tags", list),
            self._soft_expect_type(raw, "variants", list),
        ]
        for msg in soft_checks:
            if msg:
                bundle.add_message("WARNING", path, msg)

        procedure_id = raw.get("procedure_id", path.stem)
        name = self._normalize_text(raw.get("name"))

        if "common_cases_notes" in raw:
            bundle.add_message(
                "INFO",
                path,
                "Co field optional 'common_cases_notes'",
            )

        return {
            "id": procedure_id,
            "type": "procedure",
            "title": name,
            "module": raw.get("module"),
            "procedure_id": procedure_id,
            "source_procedure_code": raw.get("source_procedure_code"),
            "authority": raw.get("authority", {}),
            "citations": raw.get("citations", []),
            "processing_time": raw.get("processing_time"),
            "fees": raw.get("fees"),
            "submission_methods": raw.get("submission_methods", []),
            "effectivity": raw.get("effectivity"),
            "tags": raw.get("tags", []),
            "variants": raw.get("variants", []),
            "common_cases_notes": raw.get("common_cases_notes"),
            "raw": raw,
            "source_path": str(path),
        }

    def _normalize_authority(self, path: Path, raw: dict[str, Any], bundle: HotichBundle) -> dict[str, Any] | None:
        required = ["schema_version", "module", "doc_type", "generated_from", "rules"]
        missing = self._require_fields(raw, required)
        if missing:
            bundle.add_message("ERROR", path, f"authority thieu field bat buoc: {missing}")
            return None

        rule_id = raw.get("generated_from") or path.stem

        return {
            "id": self._normalize_text(rule_id),
            "type": "authority",
            "module": raw.get("module"),
            "doc_type": raw.get("doc_type"),
            "generated_from": raw.get("generated_from"),
            "effective_from": raw.get("effective_from"),
            "rules": raw.get("rules", []),
            "raw": raw,
            "source_path": str(path),
        }

    def _normalize_template(self, path: Path, raw: dict[str, Any], bundle: HotichBundle) -> dict[str, Any] | None:
        required = ["template_id", "ten_mau", "output_type", "fields", "render"]
        missing = self._require_fields(raw, required)
        if missing:
            bundle.add_message("ERROR", path, f"template thieu field bat buoc: {missing}")
            return None

        soft_checks = [
            self._soft_expect_type(raw, "fields", (list, dict)),
            self._soft_expect_type(raw, "render", dict),
        ]
        for msg in soft_checks:
            if msg:
                bundle.add_message("WARNING", path, msg)

        if "structure" in raw:
            bundle.add_message(
                "INFO",
                path,
                "Co field optional 'structure'",
            )

        template_id = raw.get("template_id", path.stem)

        return {
            "id": template_id,
            "type": "template",
            "title": self._normalize_text(raw.get("ten_mau")),
            "module": "hotich",
            "template_id": template_id,
            "loai": raw.get("loai"),
            "language": raw.get("language"),
            "output_type": raw.get("output_type"),
            "fields": raw.get("fields"),
            "render": raw.get("render"),
            "notes": raw.get("notes"),
            "source_file": raw.get("source_file"),
            "structure": raw.get("structure"),
            "raw": raw,
            "source_path": str(path),
        }

    def _normalize_case(self, path: Path, raw: dict[str, Any], bundle: HotichBundle) -> dict[str, Any] | None:
        required = [
            "case_id",
            "title",
            "module",
            "inputs_schema",
            "legal_basis",
            "decision_points",
            "steps",
            "outputs",
            "topic_tags",
        ]
        missing = self._require_fields(raw, required)
        if missing:
            bundle.add_message("ERROR", path, f"case thieu field bat buoc: {missing}")
            return None

        case_id = raw.get("case_id", path.stem)

        return {
            "id": case_id,
            "type": "case",
            "title": self._normalize_text(raw.get("title")),
            "module": raw.get("module"),
            "assumptions": raw.get("assumptions"),
            "inputs_schema": raw.get("inputs_schema", {}),
            "legal_basis": raw.get("legal_basis", []),
            "decision_points": raw.get("decision_points", []),
            "steps": raw.get("steps", []),
            "outputs": raw.get("outputs"),
            "topic_tags": raw.get("topic_tags", []),
            "raw": raw,
            "source_path": str(path),
        }

    def _normalize_procedure_catalog(
        self,
        path: Path,
        raw: dict[str, Any],
        bundle: HotichBundle,
    ) -> dict[str, Any] | None:
        required = ["doc_id", "doc_type", "module", "scope", "sections"]
        missing = self._require_fields(raw, required)
        if missing:
            bundle.add_message("ERROR", path, f"procedure_catalog thieu field bat buoc: {missing}")
            return None

        doc_id = raw.get("doc_id", path.stem)

        return {
            "id": doc_id,
            "type": "procedure_catalog",
            "module": raw.get("module"),
            "doc_type": raw.get("doc_type"),
            "scope": raw.get("scope"),
            "effective_from": raw.get("effective_from"),
            "effective_to": raw.get("effective_to"),
            "sections": raw.get("sections", []),
            "raw": raw,
            "source_path": str(path),
        }

    def _normalize_qa(self, path: Path, raw: dict[str, Any], bundle: HotichBundle) -> dict[str, Any] | None:
        qa_id = raw.get("qa_id") or path.stem
        return {
            "id": qa_id,
            "type": "qa",
            "raw": raw,
            "source_path": str(path),
        }

    # -------------------------------------------------------------------------
    # Public load
    # -------------------------------------------------------------------------

    def load_all(self) -> HotichBundle:
        bundle = HotichBundle(data_root=self.data_root)

        # 00_registry
        registry_dir = self.data_root / "00_registry"
        registry_files = self._read_folder_as_objects(registry_dir, bundle)
        if not registry_files:
            bundle.add_message("WARNING", registry_dir, "Khong tim thay file registry nao")
        else:
            if len(registry_files) > 1:
                bundle.add_message("WARNING", registry_dir, "Co nhieu hon 1 file registry")
            path, raw = registry_files[0]
            missing = self._require_fields(raw, ["schema_version", "module", "items"])
            if missing:
                bundle.add_message("ERROR", path, f"registry thieu field bat buoc: {missing}")
            else:
                bundle.registry = raw

        # 02_meta
        meta_dir = self.data_root / "02_meta"
        for path, raw in self._read_folder_as_objects(meta_dir, bundle):
            item = self._normalize_meta(path, raw, bundle)
            if item:
                bundle.meta[item["id"]] = item

        # 04_structured/legal
        legal_dir = self.data_root / "04_structured" / "legal"
        for path, raw in self._read_folder_as_objects(legal_dir, bundle):
            item = self._normalize_legal(path, raw, bundle)
            if item:
                bundle.legal_docs[item["id"]] = item

        # 04_structured/procedures
        procedures_dir = self.data_root / "04_structured" / "procedures"
        for path, raw in self._read_folder_as_objects(procedures_dir, bundle):
            item = self._normalize_procedure(path, raw, bundle)
            if item:
                bundle.procedures[item["id"]] = item

        # 04_structured/authority
        authority_dir = self.data_root / "04_structured" / "authority"
        for path, raw in self._read_folder_as_objects(authority_dir, bundle):
            item = self._normalize_authority(path, raw, bundle)
            if item:
                bundle.authority_rules[item["id"]] = item

        # 04_structured/templates
        templates_dir = self.data_root / "04_structured" / "templates"
        for path, raw in self._read_folder_as_objects(templates_dir, bundle):
            item = self._normalize_template(path, raw, bundle)
            if item:
                bundle.templates[item["id"]] = item

        # 04_structured/cases
        cases_dir = self.data_root / "04_structured" / "cases"
        for path, raw in self._read_folder_as_objects(cases_dir, bundle):
            item = self._normalize_case(path, raw, bundle)
            if item:
                bundle.cases[item["id"]] = item

        # 04_structured/procedure_catalog
        procedure_catalog_dir = self.data_root / "04_structured" / "procedure_catalog"
        for path, raw in self._read_folder_as_objects(procedure_catalog_dir, bundle):
            item = self._normalize_procedure_catalog(path, raw, bundle)
            if item:
                bundle.procedure_catalog[item["id"]] = item

        # 04_structured/qa
        qa_dir = self.data_root / "04_structured" / "qa"
        for path, raw in self._read_folder_as_objects(qa_dir, bundle):
            item = self._normalize_qa(path, raw, bundle)
            if item:
                bundle.qa[item["id"]] = item

        # Cross checks nhe
        self._post_checks(bundle)

        return bundle

    def _post_checks(self, bundle: HotichBundle) -> None:
        # Meta co file structured khong?
        for meta_id, meta in bundle.meta.items():
            files_section = meta.get("files", {})
            if isinstance(files_section, dict):
                structured = files_section.get("structured", {})
                if isinstance(structured, dict):
                    structured_path = structured.get("path")
                    if structured_path:
                        abs_path = self.data_root / structured_path
                        if not abs_path.exists():
                            bundle.add_message(
                                "WARNING",
                                meta["source_path"],
                                f"Structured path khong ton tai: {structured_path}",
                            )

        # Legal doc co meta tuong ung khong?
        for legal_id in bundle.legal_docs:
            if legal_id not in bundle.meta:
                bundle.add_message(
                    "WARNING",
                    legal_id,
                    "Khong tim thay 02_meta tuong ung cho legal doc nay",
                )

    # -------------------------------------------------------------------------
    # Convenience helpers cho giai doan retrieval sau nay
    # -------------------------------------------------------------------------

    def build_search_corpus(self, bundle: HotichBundle) -> list[dict[str, Any]]:
        """
        Tra ve danh sach record don gian de sau nay search bang keyword.
        """
        corpus: list[dict[str, Any]] = []

        for item in bundle.legal_docs.values():
            title = item.get("title", "")
            text = " ".join([
                item.get("id", ""),
                title,
                self._normalize_text(item.get("doc_type")),
                self._normalize_text(item.get("parse_mode")),
            ]).strip()
            corpus.append({
                "id": item["id"],
                "kind": "legal",
                "title": title,
                "text": text,
            })

        for item in bundle.procedures.values():
            text = " ".join([
                item.get("id", ""),
                item.get("title", ""),
                " ".join(item.get("tags", []) if isinstance(item.get("tags"), list) else []),
            ]).strip()
            corpus.append({
                "id": item["id"],
                "kind": "procedure",
                "title": item.get("title", ""),
                "text": text,
            })

        for item in bundle.templates.values():
            text = " ".join([
                item.get("id", ""),
                item.get("title", ""),
                self._normalize_text(item.get("loai")),
            ]).strip()
            corpus.append({
                "id": item["id"],
                "kind": "template",
                "title": item.get("title", ""),
                "text": text,
            })

        for item in bundle.cases.values():
            tags = item.get("topic_tags", [])
            text = " ".join([
                item.get("id", ""),
                item.get("title", ""),
                " ".join(tags if isinstance(tags, list) else []),
            ]).strip()
            corpus.append({
                "id": item["id"],
                "kind": "case",
                "title": item.get("title", ""),
                "text": text,
            })

        return corpus


def main() -> None:
    loader = HotichLoader()
    bundle = loader.load_all()
    bundle.print_summary()

    print("\nMot vai warning/error dau tien:")
    bundle.print_messages(limit=15)

    corpus = loader.build_search_corpus(bundle)
    print(f"\nSearch corpus size: {len(corpus)}")


if __name__ == "__main__":
    main()