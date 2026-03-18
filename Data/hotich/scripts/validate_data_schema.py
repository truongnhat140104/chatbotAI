from __future__ import annotations

import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import yaml


SUPPORTED_EXTS = {".json", ".yaml", ".yml"}


def infer_signature(value: Any) -> Any:
    if value is None:
        return "null"
    if isinstance(value, bool):
        return "bool"
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        return "number"
    if isinstance(value, str):
        return "string"

    if isinstance(value, list):
        if not value:
            return {"type": "list", "items": "empty"}

        item_signatures = []
        seen = set()

        for item in value:
            sig = infer_signature(item)
            sig_key = json.dumps(sig, sort_keys=True, ensure_ascii=False)
            if sig_key not in seen:
                seen.add(sig_key)
                item_signatures.append(sig)

        item_signatures = sorted(
            item_signatures,
            key=lambda x: json.dumps(x, sort_keys=True, ensure_ascii=False)
        )
        return {"type": "list", "items": item_signatures}

    if isinstance(value, dict):
        return {
            "type": "object",
            "properties": {
                k: infer_signature(v)
                for k, v in sorted(value.items())
            }
        }

    return f"unknown:{type(value).__name__}"


def load_structured_file(path: Path) -> Any:
    suffix = path.suffix.lower()

    with path.open("r", encoding="utf-8") as f:
        if suffix == ".json":
            return json.load(f)
        if suffix in {".yaml", ".yml"}:
            return yaml.safe_load(f)

    raise ValueError(f"Unsupported file type: {path.name}")


def compare_top_keys(data: dict, canonical_keys: set[str]) -> tuple[list[str], list[str]]:
    current_keys = set(data.keys())
    missing = sorted(canonical_keys - current_keys)
    extra = sorted(current_keys - canonical_keys)
    return missing, extra


def analyze_folder(folder: Path) -> dict:
    files = sorted([p for p in folder.iterdir() if p.is_file() and p.suffix.lower() in SUPPORTED_EXTS])

    result = {
        "folder": str(folder),
        "file_count": len(files),
        "parse_errors": [],
        "non_object_files": [],
        "schema_groups": [],
        "majority_top_keys": [],
        "deviations": [],
        "status": "OK",
    }

    if not files:
        result["status"] = "EMPTY"
        return result

    loaded_files: list[tuple[Path, dict]] = []
    signature_groups = defaultdict(list)
    top_key_counter = Counter()

    for file_path in files:
        try:
            data = load_structured_file(file_path)
        except Exception as e:
            result["parse_errors"].append({
                "file": file_path.name,
                "error": str(e),
            })
            continue

        if not isinstance(data, dict):
            result["non_object_files"].append(file_path.name)
            continue

        loaded_files.append((file_path, data))

        sig = infer_signature(data)
        sig_key = json.dumps(sig, sort_keys=True, ensure_ascii=False)
        signature_groups[sig_key].append(file_path.name)

        top_keys = tuple(sorted(data.keys()))
        top_key_counter[top_keys] += 1

    if result["parse_errors"] or result["non_object_files"]:
        result["status"] = "ERROR"

    if not loaded_files:
        return result

    for sig_key, files_in_group in sorted(signature_groups.items(), key=lambda x: (-len(x[1]), x[0])):
        result["schema_groups"].append({
            "count": len(files_in_group),
            "files": files_in_group,
            "signature": json.loads(sig_key),
        })

    majority_key_tuple, _ = top_key_counter.most_common(1)[0]
    majority_keys = set(majority_key_tuple)
    result["majority_top_keys"] = list(majority_key_tuple)

    for file_path, data in loaded_files:
        missing, extra = compare_top_keys(data, majority_keys)
        if missing or extra:
            result["deviations"].append({
                "file": file_path.name,
                "missing_keys": missing,
                "extra_keys": extra,
            })

    if len(signature_groups) > 1:
        result["status"] = "WARNING"

    if result["parse_errors"] or result["non_object_files"]:
        result["status"] = "ERROR"

    return result


def print_report(report: list[dict]) -> None:
    print("=" * 90)
    print("KIEM TRA SCHEMA JSON/YAML")
    print("=" * 90)

    for item in report:
        print(f"\n[{item['folder']}]")
        print(f"  - so file: {item['file_count']}")
        print(f"  - trang thai: {item['status']}")

        if item["parse_errors"]:
            print("  - loi parse:")
            for err in item["parse_errors"]:
                print(f"      * {err['file']}: {err['error']}")

        if item["non_object_files"]:
            print("  - file khong phai object:")
            for name in item["non_object_files"]:
                print(f"      * {name}")

        print(f"  - so nhom schema: {len(item['schema_groups'])}")

        if item["majority_top_keys"]:
            print(f"  - top-level keys pho bien: {item['majority_top_keys']}")

        if len(item["schema_groups"]) > 1:
            print("  - cac nhom schema:")
            for idx, group in enumerate(item["schema_groups"], start=1):
                preview = ", ".join(group["files"][:5])
                more = ""
                if len(group["files"]) > 5:
                    more = f" ... (+{len(group['files']) - 5} file)"
                print(f"      #{idx}: {group['count']} file -> {preview}{more}")

        if item["deviations"]:
            print("  - file lech top-level keys:")
            for dev in item["deviations"][:10]:
                print(f"      * {dev['file']} | missing={dev['missing_keys']} | extra={dev['extra_keys']}")
            if len(item["deviations"]) > 10:
                print(f"      ... con {len(item['deviations']) - 10} file")

    print("\n" + "=" * 90)


def main() -> None:
    script_path = Path(__file__).resolve()
    hotich_root = script_path.parents[1]   # .../Data/hotich
    project_root = hotich_root.parents[1]  # .../PythonProject1

    targets = [
        hotich_root / "00_registry",
        hotich_root / "02_meta",
        hotich_root / "04_structured" / "authority",
        hotich_root / "04_structured" / "cases",
        hotich_root / "04_structured" / "legal",
        hotich_root / "04_structured" / "procedure_catalog",
        hotich_root / "04_structured" / "procedures",
        hotich_root / "04_structured" / "qa",
        hotich_root / "04_structured" / "templates",
    ]

    report = []
    for folder in targets:
        if folder.exists():
            report.append(analyze_folder(folder))
        else:
            report.append({
                "folder": str(folder),
                "file_count": 0,
                "parse_errors": [],
                "non_object_files": [],
                "schema_groups": [],
                "majority_top_keys": [],
                "deviations": [],
                "status": "MISSING",
            })

    print_report(report)

    output_path = hotich_root / "_schema_report_all.json"
    with output_path.open("w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)

    print(f"Da ghi bao cao tai: {output_path}")


if __name__ == "__main__":
    main()