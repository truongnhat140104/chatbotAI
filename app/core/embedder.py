from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from sentence_transformers import SentenceTransformer

try:
    from pyvi import ViTokenizer
except ImportError:
    ViTokenizer = None


@dataclass
class EmbedBuildResult:
    output_dir: Path
    num_units: int
    embedding_dim: int
    model_name: str
    normalized: bool

    def print_summary(self) -> None:
        print("=" * 80)
        print("HOTICH EMBEDDER SUMMARY")
        print("=" * 80)
        print(f"Output dir     : {self.output_dir}")
        print(f"Units embedded : {self.num_units}")
        print(f"Embedding dim  : {self.embedding_dim}")
        print(f"Model          : {self.model_name}")
        print(f"Normalized     : {self.normalized}")
        print("=" * 80)


class HotichEmbedder:
    def __init__(
        self,
        index_dir: str | Path | None = None,
        output_dir: str | Path | None = None,
        model_name: str = "",
        batch_size: int = 32,
        normalize_embeddings: bool = True,
        device: str | None = None,
        use_word_segment: bool = True,
    ) -> None:
        self.index_dir = self._detect_index_dir(index_dir)
        self.output_dir = (
            Path(output_dir).resolve()
            if output_dir
            else (self.index_dir / "embeddings").resolve()
        )

        if not model_name.strip():
            raise ValueError(
                "Ban phai truyen model_name cua Vietnamese bi-encoder, "
                "vi du: bkai-foundation-models/vietnamese-bi-encoder"
            )

        self.model_name = model_name.strip()
        self.batch_size = batch_size
        self.normalize_embeddings = normalize_embeddings
        self.device = device
        self.use_word_segment = use_word_segment

        if self.use_word_segment and ViTokenizer is None:
            raise ImportError(
                "Chua cai pyvi. Hay cai bang: pip install pyvi"
            )

    def _detect_index_dir(self, explicit_dir: str | Path | None) -> Path:
        if explicit_dir is not None:
            p = Path(explicit_dir).resolve()
            if not p.exists():
                raise FileNotFoundError(f"Khong tim thay index_dir: {p}")
            return p

        here = Path(__file__).resolve()
        candidates = [
            here.parents[2] / "Data" / "hotich" / "05_index",
            Path.cwd() / "Data" / "hotich" / "05_index",
        ]

        for c in candidates:
            if c.exists():
                return c.resolve()

        raise FileNotFoundError(
            "Khong tim thay thu muc 05_index. "
            "Hay chay indexer truoc hoac truyen --index-dir."
        )

    @staticmethod
    def _read_jsonl(path: Path) -> list[dict[str, Any]]:
        if not path.exists():
            raise FileNotFoundError(f"Khong tim thay file: {path}")

        rows: list[dict[str, Any]] = []
        with path.open("r", encoding="utf-8") as f:
            for line_no, line in enumerate(f, start=1):
                line = line.strip()
                if not line:
                    continue
                try:
                    obj = json.loads(line)
                except Exception as exc:
                    raise ValueError(f"Loi parse JSONL tai dong {line_no}: {exc}") from exc

                if isinstance(obj, dict):
                    rows.append(obj)

        return rows

    @staticmethod
    def _write_json(path: Path, obj: Any) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", encoding="utf-8") as f:
            json.dump(obj, f, ensure_ascii=False, indent=2)

    @staticmethod
    def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", encoding="utf-8") as f:
            for row in rows:
                f.write(json.dumps(row, ensure_ascii=False) + "\n")

    @staticmethod
    def _text(value: Any) -> str:
        if value is None:
            return ""
        if isinstance(value, str):
            return value.strip()
        if isinstance(value, (int, float, bool)):
            return str(value)
        return ""

    def _load_units(self) -> list[dict[str, Any]]:
        return self._read_jsonl(self.index_dir / "all_units.jsonl")

    def _segment_vi_text(self, text: str) -> str:
        text = text.strip()
        if not text:
            return ""

        if not self.use_word_segment:
            return text

        return ViTokenizer.tokenize(text)

    def _prepare_embedding_text(self, text: str) -> str:
        text = text.strip()
        if not text:
            return ""
        return self._segment_vi_text(text)

    def _make_embedding_inputs(self, units: list[dict[str, Any]]) -> list[str]:
        texts: list[str] = []
        for unit in units:
            text = self._text(unit.get("embedding_text"))
            if not text:
                title = self._text(unit.get("title"))
                body = self._text(unit.get("text"))
                text = "\n".join(x for x in [title, body] if x).strip()

            texts.append(self._prepare_embedding_text(text))
        return texts

    def _make_unit_refs(self, units: list[dict[str, Any]]) -> list[dict[str, Any]]:
        refs: list[dict[str, Any]] = []
        for i, unit in enumerate(units):
            refs.append(
                {
                    "row_index": i,
                    "unit_id": unit.get("unit_id"),
                    "source_kind": unit.get("source_kind"),
                    "unit_kind": unit.get("unit_kind"),
                    "title": unit.get("title"),
                    "source_path": unit.get("source_path"),
                    "doc_id": unit.get("doc_id"),
                    "doc_title": unit.get("doc_title"),
                    "doc_number": unit.get("doc_number"),
                    "procedure_id": unit.get("procedure_id"),
                    "template_id": unit.get("template_id"),
                    "case_id": unit.get("case_id"),
                    "authority_id": unit.get("authority_id"),
                    "citation_label": unit.get("citation_label"),
                }
            )
        return refs

    def _load_model(self) -> SentenceTransformer:
        if self.device:
            return SentenceTransformer(self.model_name, device=self.device)
        return SentenceTransformer(self.model_name)

    def build(self) -> EmbedBuildResult:
        units = self._load_units()
        texts = self._make_embedding_inputs(units)
        refs = self._make_unit_refs(units)

        self.output_dir.mkdir(parents=True, exist_ok=True)

        print(f"[INFO] Loading model: {self.model_name}")
        model = self._load_model()

        print(f"[INFO] Encoding {len(texts)} units...")
        embeddings = model.encode(
            texts,
            batch_size=self.batch_size,
            show_progress_bar=True,
            convert_to_numpy=True,
            normalize_embeddings=self.normalize_embeddings,
        )

        if not isinstance(embeddings, np.ndarray):
            embeddings = np.asarray(embeddings, dtype=np.float32)

        embeddings = embeddings.astype(np.float32)

        emb_path = self.output_dir / "embeddings.npy"
        refs_path = self.output_dir / "unit_refs.jsonl"
        manifest_path = self.output_dir / "manifest.json"

        np.save(emb_path, embeddings)
        self._write_jsonl(refs_path, refs)

        manifest = {
            "model_name": self.model_name,
            "num_units": int(embeddings.shape[0]),
            "embedding_dim": int(embeddings.shape[1]) if embeddings.ndim == 2 else 0,
            "normalize_embeddings": self.normalize_embeddings,
            "batch_size": self.batch_size,
            "device": self.device or "auto",
            "use_word_segment": self.use_word_segment,
            "index_dir": str(self.index_dir),
            "output_dir": str(self.output_dir),
            "source_file": str(self.index_dir / "all_units.jsonl"),
            "artifacts": {
                "embeddings": str(emb_path),
                "unit_refs": str(refs_path),
            },
        }
        self._write_json(manifest_path, manifest)

        return EmbedBuildResult(
            output_dir=self.output_dir,
            num_units=int(embeddings.shape[0]),
            embedding_dim=int(embeddings.shape[1]) if embeddings.ndim == 2 else 0,
            model_name=self.model_name,
            normalized=self.normalize_embeddings,
        )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build embeddings for Hotich index")
    parser.add_argument(
        "--model-name",
        required=True,
        help="Ten model SentenceTransformer / Vietnamese bi-encoder",
    )
    parser.add_argument(
        "--index-dir",
        default=None,
        help="Thu muc 05_index. Mac dinh: Data/hotich/05_index",
    )
    parser.add_argument(
        "--output-dir",
        default=None,
        help="Thu muc output embeddings. Mac dinh: <index_dir>/embeddings",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=32,
        help="Batch size khi encode",
    )
    parser.add_argument(
        "--device",
        default=None,
        help="cpu / cuda / mps ... Neu bo trong thi tu dong",
    )
    parser.add_argument(
        "--no-normalize",
        action="store_true",
        help="Tat normalize_embeddings",
    )
    parser.add_argument(
        "--no-word-segment",
        action="store_true",
        help="Tat tach tu tieng Viet truoc khi encode",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    embedder = HotichEmbedder(
        index_dir=args.index_dir,
        output_dir=args.output_dir,
        model_name=args.model_name,
        batch_size=args.batch_size,
        normalize_embeddings=not args.no_normalize,
        device=args.device,
        use_word_segment=not args.no_word_segment,
    )
    result = embedder.build()
    result.print_summary()

    print("\nDa tao artifacts:")
    print(f"- {result.output_dir / 'embeddings.npy'}")
    print(f"- {result.output_dir / 'unit_refs.jsonl'}")
    print(f"- {result.output_dir / 'manifest.json'}")


if __name__ == "__main__":
    main()