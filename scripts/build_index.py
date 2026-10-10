#!/usr/bin/env python3
"""Build INDEX.md from the front matter of every entry.

Usage:
    python scripts/build_index.py          # rewrite INDEX.md
    python scripts/build_index.py --check  # exit 1 if INDEX.md is stale or an entry is invalid

No third-party dependencies: front matter is a small YAML subset
(`key: value` lines, lists written inline as `[a, b, c]`).
"""

import sys
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
INDEX = ROOT / "INDEX.md"
SECTIONS = {"papers": "論文 Papers", "repos": "專案 Repos", "notes": "筆記 Notes"}
REQUIRED = ("title", "date", "tags", "summary")
STATUS_ICON = {"to-read": "📥", "reading": "📖", "read": "✅"}


def parse_value(raw):
    raw = raw.strip()
    if not raw.startswith(("'", '"')):
        raw = raw.split(" #", 1)[0].strip()  # inline comment
    if raw.startswith("[") and raw.endswith("]"):
        return [item.strip().strip("\"'") for item in raw[1:-1].split(",") if item.strip()]
    return raw.strip("\"'")


def parse_front_matter(path):
    lines = path.read_text(encoding="utf-8").splitlines()
    if not lines or lines[0].strip() != "---":
        raise ValueError("missing front matter (file must start with ---)")
    meta = {}
    for line in lines[1:]:
        if line.strip() == "---":
            return meta
        if not line.strip() or line.lstrip().startswith("#"):
            continue
        key, sep, value = line.partition(":")
        if not sep:
            raise ValueError(f"bad front matter line: {line!r}")
        meta[key.strip()] = parse_value(value)
    raise ValueError("front matter is not closed with ---")


def load_entries():
    entries, errors = [], []
    for section in SECTIONS:
        for path in sorted((ROOT / section).glob("*.md")):
            try:
                meta = parse_front_matter(path)
                missing = [k for k in REQUIRED if not meta.get(k)]
                if missing:
                    raise ValueError(f"missing fields: {', '.join(missing)}")
                if isinstance(meta["tags"], str):
                    meta["tags"] = [meta["tags"]]
            except ValueError as exc:
                errors.append(f"{path.relative_to(ROOT)}: {exc}")
                continue
            meta["section"] = section
            meta["path"] = path.relative_to(ROOT).as_posix()
            entries.append(meta)
    return entries, errors


def cell(text):
    return str(text).replace("|", "\\|").replace("\n", " ")


def render(entries):
    by_date = sorted(entries, key=lambda e: (e["date"], e["title"]), reverse=True)
    tags = defaultdict(list)
    for e in by_date:
        for tag in e["tags"]:
            tags[tag].append(e)

    out = [
        "# 索引 Index",
        "",
        "<!-- 自動產生，請勿手動編輯。執行 `python scripts/build_index.py` 重建。 -->",
        "",
        f"共 {len(entries)} 筆 · 📥 待讀 · 📖 閱讀中 · ✅ 已讀",
        "",
        "## 最近加入",
        "",
        "| 日期 | 標題 | 類型 | 標籤 | 摘要 |",
        "|---|---|---|---|---|",
    ]
    for e in by_date[:15]:
        out.append(
            f"| {e['date']} | [{cell(e['title'])}]({e['path']}) | {SECTIONS[e['section']].split()[0]} "
            f"| {' '.join(f'`{t}`' for t in e['tags'])} | {cell(e['summary'])} |"
        )

    for section, label in SECTIONS.items():
        items = [e for e in by_date if e["section"] == section]
        if not items:
            continue
        out += ["", f"## {label}", ""]
        for e in items:
            icon = STATUS_ICON.get(e.get("status", ""), "")
            source = f" · [來源]({e['source']})" if e.get("source") else ""
            out.append(f"- {icon} [{e['title']}]({e['path']}) — {e['summary']}{source}")

    out += ["", "## 依標籤", ""]
    for tag in sorted(tags, key=str.lower):
        links = ", ".join(f"[{e['title']}]({e['path']})" for e in tags[tag])
        out.append(f"- **{tag}** ({len(tags[tag])}): {links}")
    return "\n".join(out) + "\n"


def main():
    entries, errors = load_entries()
    for err in errors:
        print(f"error: {err}", file=sys.stderr)
    content = render(entries)
    if "--check" in sys.argv:
        current = INDEX.read_text(encoding="utf-8") if INDEX.exists() else ""
        if current != content:
            print("INDEX.md is out of date; run python scripts/build_index.py", file=sys.stderr)
            return 1
        return 1 if errors else 0
    INDEX.write_text(content, encoding="utf-8")
    print(f"INDEX.md: {len(entries)} entries")
    return 1 if errors else 0


if __name__ == "__main__":
    sys.exit(main())
