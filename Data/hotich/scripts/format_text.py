import re
import json
from pathlib import Path

import yaml
from docx import Document

# ======================
# PATHS (theo cấu trúc của bạn)
# ======================
ROOT = Path(__file__).resolve().parents[1]  # thư mục hotich/
REGISTRY = ROOT / "00_registry" / "registry.yaml"
TEXT_DIR = ROOT / "03_text" / "format"
OUT_DIR = ROOT / "04_structured" / "legal"
OUT_DIR.mkdir(parents=True, exist_ok=True)

# ======================
# REGEX nhận diện cấu trúc
# ======================
RE_CHAPTER = re.compile(r"^\s*Chương\s+([IVXLCDM]+)\.?\s*(.*)$", re.IGNORECASE)
RE_SECTION = re.compile(r"^\s*Mục\s+(\d+)\.?\s*(.*)$", re.IGNORECASE)
RE_ARTICLE = re.compile(r"^\s*Điều\s+(\d+)\.?\s*(.*)$", re.IGNORECASE)

# Khoản: hỗ trợ "2." , "2.[13]" , "2a.[19]" , "1a.[41]" ...
RE_CLAUSE = re.compile(r"^\s*(\d+[a-z]?)\.\s*(?:\[(\d+)\])?\s*(.*)$", re.IGNORECASE)

# Điểm: hỗ trợ "d) ..." , "d)[15] ..." (marker sát dấu ")")
RE_POINT = re.compile(r"^\s*([a-zđ])\)\s*(?:\[(\d+)\])?\s*(.*)$", re.IGNORECASE)

# note refs dạng [2] có thể xuất hiện ở mọi nơi trong text
RE_NOTE_REF_ANYWHERE = re.compile(r"\[(\d+)\]")

# note definition: "[2] Khoản này được sửa đổi..."
RE_NOTE_DEF = re.compile(r"^\s*\[(\d+)\]\s*(.*)$")

def normalize_line(s: str) -> str:
    s = (s or "").strip()
    s = re.sub(r"\s+", " ", s)
    return s

def read_docx_lines(path: Path) -> list[str]:
    doc = Document(str(path))
    lines = []
    for p in doc.paragraphs:
        t = normalize_line(p.text)
        if t:
            lines.append(t)
    return lines

def read_text_lines(path: Path) -> list[str]:
    txt = path.read_text(encoding="utf-8", errors="ignore")
    lines = []
    for line in txt.splitlines():
        t = normalize_line(line)
        if t:
            lines.append(t)
    return lines

def resolve_source(doc_id: str, item: dict) -> tuple[list[str], str]:
    """
    Ưu tiên:
    1) item.source_path (nếu có)  -> đường dẫn tương đối từ ROOT
    2) 03_text/<doc_id>.docx
    3) 03_text/<doc_id>.md
    4) 03_text/<doc_id>.txt
    """
    source_path = item.get("source_path")
    if source_path:
        p = (ROOT / source_path).resolve()
        if not p.exists():
            raise FileNotFoundError(f"source_path không tồn tại: {p}")
        if p.suffix.lower() == ".docx":
            return read_docx_lines(p), str(Path(source_path))
        return read_text_lines(p), str(Path(source_path))

    cand_docx = TEXT_DIR / f"{doc_id}.docx"
    cand_md = TEXT_DIR / f"{doc_id}.md"
    cand_txt = TEXT_DIR / f"{doc_id}.txt"

    if cand_docx.exists():
        return read_docx_lines(cand_docx), f"03_text/format/{cand_docx.name}"
    if cand_md.exists():
        return read_text_lines(cand_md), f"03_text/{cand_md.name}"
    if cand_txt.exists():
        return read_text_lines(cand_txt), f"03_text/{cand_txt.name}"

    raise FileNotFoundError(f"Không tìm thấy {doc_id} trong 03_text (docx/md/txt)")

def extract_note_refs_anywhere(text: str) -> list[int]:
    return sorted({int(x) for x in RE_NOTE_REF_ANYWHERE.findall(text or "")})

def strip_note_markers(text: str) -> str:
    # bỏ các [n] trong câu, dọn dấu câu và khoảng trắng
    t = RE_NOTE_REF_ANYWHERE.sub("", text or "")
    t = re.sub(r"\s+([,.;:)\]])", r"\1", t)
    t = re.sub(r"\s+", " ", t).strip()
    return t

def set_or_merge_note_refs(obj: dict, refs: list[int]):
    if not refs:
        return
    cur = set(obj.get("note_refs", []) or [])
    cur.update(refs)
    obj["note_refs"] = sorted(cur)

def build_structure(lines: list[str], doc_id: str, keep_note_markers: bool) -> dict:
    """
    Parse VBHN:
    - Marker tham chiếu nằm trong Điều/Khoản/Điểm: 4.[2], 2a.[19], d)[15]...
    - Nội dung chú thích nằm ở cuối dạng: [2] ...
    Output:
      chapters -> sections -> articles -> clauses -> points
      notes: { "2": {"text": "..."} , ... }
    """
    data = {
        "doc_id": doc_id,
        "type": "legal",
        "chapters": [],
        "notes": {}
    }

    cur_chap = None
    cur_sec = None
    cur_art = None
    cur_clause = None

    # trạng thái khi vào block notes [n] ...
    in_notes_block = False
    cur_note_no = None
    cur_note_buf: list[str] = []

    def flush_note():
        nonlocal cur_note_no, cur_note_buf
        if cur_note_no is not None:
            text = " ".join([x for x in cur_note_buf if x]).strip()
            if text:
                data["notes"][str(cur_note_no)] = {"text": text, "ref_doc_hint": None}
        cur_note_no = None
        cur_note_buf = []

    def ensure_chapter(ch_no: str, title: str):
        nonlocal cur_chap, cur_sec, cur_art, cur_clause
        refs = extract_note_refs_anywhere(title)
        title2 = title if keep_note_markers else strip_note_markers(title)
        cur_chap = {
            "chapter_no": ch_no,
            "title": title2.strip(),
            "note_refs": refs,
            "sections": [],
            "articles": []
        }
        data["chapters"].append(cur_chap)
        cur_sec = None
        cur_art = None
        cur_clause = None

    def ensure_section(sec_no: str, title: str):
        nonlocal cur_sec, cur_art, cur_clause
        if cur_chap is None:
            ensure_chapter("?", "")
        refs = extract_note_refs_anywhere(title)
        title2 = title if keep_note_markers else strip_note_markers(title)
        cur_sec = {
            "section_no": sec_no,
            "title": title2.strip(),
            "note_refs": refs,
            "articles": []
        }
        cur_chap["sections"].append(cur_sec)
        cur_art = None
        cur_clause = None

    def add_article(art_no: str, title: str):
        nonlocal cur_art, cur_clause
        if cur_chap is None:
            ensure_chapter("?", "")
        refs = extract_note_refs_anywhere(title)
        title2 = title if keep_note_markers else strip_note_markers(title)
        cur_art = {
            "article_no": int(art_no),
            "title": title2.strip(),
            "note_refs": refs,
            "clauses": [],
            "raw_paragraphs": []
        }
        if cur_sec is not None:
            cur_sec["articles"].append(cur_art)
        else:
            cur_chap["articles"].append(cur_art)
        cur_clause = None

    def add_clause(cl_key: str, note_inline: str | None, text: str):
        nonlocal cur_clause
        if cur_art is None:
            return

        refs = []
        if note_inline:
            refs.append(int(note_inline))
        refs.extend(extract_note_refs_anywhere(text))

        text2 = text if keep_note_markers else strip_note_markers(text)

        cur_clause = {
            "clause_key": cl_key,     # ví dụ: "2", "2a", "1a"
            "text": text2,
            "note_refs": sorted(set(refs)),
            "points": [],
            "raw_lines": []
        }
        cur_art["clauses"].append(cur_clause)

    def add_point(pt_key: str, note_inline: str | None, text: str):
        nonlocal cur_clause
        if cur_art is None:
            return
        if cur_clause is None:
            add_clause("0", None, "")

        refs = []
        if note_inline:
            refs.append(int(note_inline))
        refs.extend(extract_note_refs_anywhere(text))

        text2 = text if keep_note_markers else strip_note_markers(text)

        cur_clause["points"].append({
            "point_key": pt_key.lower(),
            "text": text2,
            "note_refs": sorted(set(refs))
        })

    for line in lines:
        # 1) Nếu đã vào notes block thì chỉ parse notes, không parse chương/điều nữa
        if in_notes_block:
            mdef = RE_NOTE_DEF.match(line)
            if mdef:
                flush_note()
                cur_note_no = int(mdef.group(1))
                first = mdef.group(2).strip()
                cur_note_buf = [first] if first else []
            else:
                # nối dòng chú thích (multi-line)
                if cur_note_no is not None:
                    cur_note_buf.append(line.strip())
            continue

        # 2) Detect bắt đầu notes block: gặp dòng "[n] ..." SAU khi đã có ít nhất 1 điều
        mdef = RE_NOTE_DEF.match(line)
        if mdef and cur_art is not None:
            in_notes_block = True
            cur_note_no = int(mdef.group(1))
            first = mdef.group(2).strip()
            cur_note_buf = [first] if first else []
            continue

        # 3) Parse cấu trúc chính
        m = RE_CHAPTER.match(line)
        if m:
            ensure_chapter(m.group(1), m.group(2))
            continue

        m = RE_SECTION.match(line)
        if m:
            ensure_section(m.group(1), m.group(2))
            continue

        m = RE_ARTICLE.match(line)
        if m:
            add_article(m.group(1), m.group(2))
            continue

        # Khoản (mạnh nhất): bắt được cả 4.[2]..., 2a.[19]..., 2.[13]...
        m = RE_CLAUSE.match(line)
        if m and cur_art is not None:
            cl_key = m.group(1)
            note_inline = m.group(2)  # có thể None
            cl_text = m.group(3) or ""
            add_clause(cl_key, note_inline, cl_text)
            continue

        # Điểm: d)[15]...
        m = RE_POINT.match(line)
        if m and cur_art is not None:
            pt_key = m.group(1)
            note_inline = m.group(2)
            pt_text = m.group(3) or ""
            add_point(pt_key, note_inline, pt_text)
            continue

        # 4) fallback: dòng tiếp theo của khoản/điều
        if cur_clause is not None:
            # nếu dòng lẻ có marker [n] thì cộng note_refs vào clause
            set_or_merge_note_refs(cur_clause, extract_note_refs_anywhere(line))
            cur_clause["raw_lines"].append(line if keep_note_markers else strip_note_markers(line))
        elif cur_art is not None:
            set_or_merge_note_refs(cur_art, extract_note_refs_anywhere(line))
            cur_art["raw_paragraphs"].append(line if keep_note_markers else strip_note_markers(line))

    # chốt notes cuối file
    if in_notes_block:
        flush_note()

    return data

def main():
    reg = yaml.safe_load(REGISTRY.read_text(encoding="utf-8"))
    items = reg.get("items", [])
    module = reg.get("module")
    schema_version = reg.get("schema_version")

    built = 0
    for item in items:
        if not item.get("enabled", False):
            continue

        doc_id = item["doc_id"]
        keep_note_markers = bool(item.get("keep_note_markers", False))
        parse_mode = item.get("parse_mode", "clauses_with_notes")

        lines, source = resolve_source(doc_id, item)

        structured = build_structure(lines, doc_id, keep_note_markers)
        structured["source"] = source
        structured["parse_mode"] = parse_mode
        if module:
            structured["module"] = module
        if schema_version:
            structured["schema_version"] = schema_version

        out_path = OUT_DIR / f"{doc_id}.json"
        out_path.write_text(json.dumps(structured, ensure_ascii=False, indent=2), encoding="utf-8")

        built += 1
        print(f"[OK] {doc_id} -> {out_path}")

    print(f"Done. Built {built} file(s).")

if __name__ == "__main__":
    main()