"""Regenerate the marked blocks of docs/status.md from code, config and (optionally) tickets.

Stdlib only and deterministic (no timestamps). Hand-written text outside the markers is never
touched. Blocks: built, adrs, evals (tracked files only) and tickets (needs --tickets and --prs).

    python scripts/status_doc.py --write
    python scripts/status_doc.py --check
    python scripts/status_doc.py --write --tickets <dir> --prs <gh-pr-list.json>
"""

# ruff: noqa: E501

from __future__ import annotations

import argparse
import ast
import json
import re
import sys
import tomllib
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DOC = Path("docs/status.md")
TRACKED_BLOCKS = ("built", "adrs", "evals")
DEFAULT_REPO = "arnavk2319/prod-rag-tool"
SECRET_HINTS = ("key", "password", "secret", "token_cap_none", "database_url", "redis_url")

_MARK = r"<!-- generated:{name} begin -->\n(.*?)<!-- generated:{name} end -->"


def replace_block(text: str, name: str, body: str) -> str:
    """Return text with the named block body replaced; raise KeyError if markers are absent."""
    pat = re.compile(_MARK.format(name=re.escape(name)), re.S)
    if not pat.search(text):
        raise KeyError(name)
    new = f"<!-- generated:{name} begin -->\n{body.rstrip()}\n<!-- generated:{name} end -->"
    return pat.sub(lambda _m: new, text, count=1)


def validate_markers(text: str) -> None:
    """Raise ValueError (one line) on unbalanced, nested or duplicate block markers."""
    seen: set[str] = set()
    open_name: str | None = None
    for m in re.finditer(r"<!-- generated:([\w-]+) (begin|end) -->", text):
        name, kind = m.group(1), m.group(2)
        if kind == "begin":
            if open_name is not None:
                raise ValueError(f"nested or unclosed marker: {open_name} (before {name})")
            if name in seen:
                raise ValueError(f"duplicate block markers: {name}")
            seen.add(name)
            open_name = name
        else:
            if open_name != name:
                raise ValueError(f"unbalanced marker: end of {name} without matching begin")
            open_name = None
    if open_name is not None:
        raise ValueError(f"unbalanced marker: begin of {open_name} has no end")


def get_block(text: str, name: str) -> str | None:
    m = re.search(_MARK.format(name=re.escape(name)), text, re.S)
    return m.group(1).rstrip("\n") if m else None


# ---------------------------------------------------------------- readers


def _read(root: Path, rel: str) -> str:
    p = root / rel
    return p.read_text(encoding="utf-8") if p.is_file() else ""


def module_constants(src: str) -> dict[str, object]:
    out: dict[str, object] = {}
    for node in ast.parse(src).body:
        if (
            isinstance(node, ast.Assign)
            and len(node.targets) == 1
            and isinstance(node.targets[0], ast.Name)
            and node.targets[0].id.isupper()
        ):
            try:
                out[node.targets[0].id] = ast.literal_eval(node.value)
            except ValueError:
                continue
    return out


def settings_defaults(src: str) -> dict[str, object]:
    out: dict[str, object] = {}
    for node in ast.parse(src).body:
        if not (isinstance(node, ast.ClassDef) and node.name == "Settings"):
            continue
        for st in node.body:
            if not (isinstance(st, ast.AnnAssign) and isinstance(st.target, ast.Name)):
                continue
            val = st.value
            if isinstance(val, ast.Call):
                kws = {k.arg: k.value for k in val.keywords}
                val = kws.get("default")
            if val is None:
                continue
            try:
                out[st.target.id] = ast.literal_eval(val)
            except ValueError:
                continue
    return out


def compose_info(text: str) -> list[tuple[str, str, str]]:
    """(service, image, ports) per service, in file order."""
    rows: list[tuple[str, str, str]] = []
    in_services = False
    cur = ""
    image: dict[str, str] = {}
    ports: dict[str, list[str]] = {}
    order: list[str] = []
    for line in text.splitlines():
        if re.match(r"^\S", line):
            in_services = line.startswith("services:")
            continue
        if not in_services:
            continue
        m = re.match(r"^  ([A-Za-z][\w-]*):\s*$", line)
        if m:
            cur = m.group(1)
            order.append(cur)
            continue
        m = re.match(r"^\s+image:\s*(\S+)", line)
        if m and cur:
            image[cur] = m.group(1)
        m = re.match(r'^\s+- "(?:\$\{BIND_ADDR:-[^}]*\}:)?\$\{(\w+):-(\d+)\}:(\d+)"', line)
        if m and cur:
            ports.setdefault(cur, []).append(f"{m.group(2)} (env {m.group(1)})")
    for s in order:
        if s in image or s in ports:
            rows.append(
                (s, image.get(s, "built from this repo"), ", ".join(ports.get(s, [])) or "-")
            )
    return rows


def migration_info(root: Path) -> tuple[str, list[str], list[str]]:
    revs: list[str] = []
    tables: list[str] = []
    for f in sorted((root / "migrations/versions").glob("*.py")):
        t = f.read_text(encoding="utf-8")
        m = re.search(r'^revision\s*=\s*"([^"]+)"', t, re.M)
        if m:
            revs.append(m.group(1))
        for tb in re.findall(r"CREATE TABLE (?:IF NOT EXISTS )?([a-z_]+) \(", t):
            if tb not in tables:
                tables.append(tb)
    roles = sorted(
        set(
            re.findall(
                r"CREATE ROLE (\w+)", _read(root, "migrations/versions/0001_initial_schema.py")
            )
        )
    )
    roles = sorted(
        set(roles) | set(re.findall(r"CREATE ROLE (\w+)", _read(root, "docker/postgres/init.sh")))
    )
    return (max(revs) if revs else "none", tables, roles)


def adr_titles(root: Path) -> list[tuple[str, str]]:
    out = []
    for f in sorted((root / "docs/decisions").glob("ADR-[0-9]*.md")):
        first = f.read_text(encoding="utf-8").splitlines()[0] if f.stat().st_size else ""
        out.append((f.name, first.lstrip("# ").strip()))
    return out


# ---------------------------------------------------------------- blocks


def block_built(root: Path) -> str:
    st = settings_defaults(_read(root, "src/rag_platform/settings.py"))
    ch = module_constants(_read(root, "src/rag_platform/services/ingest/chunker.py"))
    em = module_constants(_read(root, "src/rag_platform/adapters/embedders.py"))
    py = tomllib.loads(_read(root, "pyproject.toml") or "")
    deps = py.get("project", {}).get("dependencies", [])
    dev = py.get("dependency-groups", {}).get("dev", [])
    rev, tables, roles = migration_info(root)
    comp = compose_info(_read(root, "compose.yaml"))
    models = root / "config/models.yaml"
    L: list[str] = []
    L.append("**Models and retrieval pieces (from code)**")
    L.append("")
    L.append("| Item | Value |")
    L.append("| --- | --- |")
    L.append(
        f"| Chunking strategy | single-level `{ch.get('CHUNKER_VERSION')}`: {ch.get('MIN_TOKENS')}-{ch.get('MAX_TOKENS')} tokens (target {ch.get('TARGET_TOKENS')}), overlap ratio {ch.get('OVERLAP_RATIO')} |"
    )
    L.append(
        f"| Embedding model | `{em.get('FASTEMBED_EMBED_MODEL')}`, {em.get('HASH_DIM')} dims; backend default `{st.get('embedding_backend')}` (`hash` = deterministic test embedder) |"
    )
    L.append(
        f"| Reranker | `{em.get('FASTEMBED_RERANK_MODEL')}` adapter exists; not yet wired into any retrieval path |"
    )
    L.append(f"| Provider default | `{st.get('provider')}` (no real model is called by default) |")
    if models.is_file():
        names = [
            m.strip()
            for m in re.findall(
                r"^\s*-?\s*(?:id|model|name):\s*(.+)$", models.read_text(encoding="utf-8"), re.M
            )
        ]
        L.append(
            f"| LLM fallback chain | `config/models.yaml` entries: {', '.join(f'`{n}`' for n in names) or 'none'} |"
        )
    else:
        L.append("| LLM fallback chain | not built: `config/models.yaml` does not exist yet |")
    has_gate = any((root / "src/rag_platform").rglob("policy*.py"))
    L.append(
        f"| Pipeline policy gate | {'code present' if has_gate else 'not built (no policy module in `src/`)'} |"
    )
    L.append(
        f"| Vector store and index | Postgres with pgvector (`{dict((s, i) for s, i, _ in comp).get('postgres', '-')}`); tables `index_versions`, `index_alias`, `chunks` exist in the schema; the per-partition HNSW index DDL exists, but no index is populated or queried yet |"
    )
    L.append("| Registry | `registry_pointers` table exists in the schema; nothing writes it yet |")
    L.append(f"| Database roles | {', '.join(f'`{r}`' for r in roles) or 'none'} |")
    L.append(f"| Highest migration revision | `{rev}` |")
    L.append("")
    L.append(
        "**Settings defaults** (`src/rag_platform/settings.py`, secrets and credentialed URLs omitted)"
    )
    L.append("")
    L.append("| Setting | Default |")
    L.append("| --- | --- |")
    for k, v in st.items():
        if any(h in k for h in SECRET_HINTS):
            continue
        L.append(f"| `{k}` | `{v}` |")
    L.append("")
    L.append("**Compose services, images and host ports** (`compose.yaml`)")
    L.append("")
    L.append("| Service | Image | Host port (default) |")
    L.append("| --- | --- | --- |")
    for s, i, p in comp:
        L.append(f"| {s} | `{i}` | {p} |")
    L.append("")
    L.append("**Database tables** (all migrations): " + ", ".join(f"`{t}`" for t in tables))
    L.append("")
    L.append("**Pinned dependencies** (`pyproject.toml`)")
    L.append("")
    L.append("Runtime: " + ", ".join(f"`{d}`" for d in deps))
    L.append("")
    L.append("Dev: " + ", ".join(f"`{d}`" for d in dev))
    return "\n".join(L)


def block_adrs(root: Path) -> str:
    rows = adr_titles(root)
    L = ["| File | Title |", "| --- | --- |"]
    L += [f"| `{f}` | {t} |" for f, t in rows]
    return "\n".join(L)


def block_evals(root: Path) -> str:
    p = root / "evals/results/latest.json"
    if not p.is_file():
        return "No eval results are committed yet."
    res = json.loads(p.read_text(encoding="utf-8"))
    th_path = root / "evals/thresholds.json"
    th_flat: dict[str, float] = {}
    if th_path.is_file():
        for grp in json.loads(th_path.read_text(encoding="utf-8")).values():
            for k, v in grp.items():
                th_flat[k] = v
    L = [
        f"Suite `{res.get('suite')}`, served model `{res.get('served_model')}`. {res.get('header', '')}.",
        "",
        "| Metric | Value | Threshold | Result |",
        "| --- | --- | --- | --- |",
    ]
    for k, v in sorted(res.get("metrics", {}).items()):
        if k in th_flat:
            ok = v >= th_flat[k]
            L.append(f"| {k} | {v} | >= {th_flat[k]} | {'pass' if ok else 'FAIL'} |")
        else:
            L.append(f"| {k} | {v} | none set | n/a |")
    L.append("")
    hz = res.get("hard_zero", {})
    L.append("Hard-zero counters: " + ", ".join(f"{k} = {v}" for k, v in sorted(hz.items())) + ".")
    L.append("")
    L.append("| Class | Cases | Passed |")
    L.append("| --- | --- | --- |")
    for k, v in sorted(res.get("classes", {}).items()):
        L.append(f"| {k} | {v['n']} | {v['passed']} |")
    return "\n".join(L)


# ---------------------------------------------------------------- tickets


def parse_ticket(path: Path) -> dict[str, object] | None:
    """Read only id, title, phase, status and depends_on; ignore every other line."""
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except OSError:
        return None
    m = re.match(r"^# (T-\d+) (.+)$", lines[0]) if lines else None
    if not m:
        return None
    out: dict[str, object] = {
        "id": m.group(1),
        "title": m.group(2).strip(),
        "phase": "",
        "status": "",
        "depends": [],
    }
    for line in lines[1:25]:
        k = re.match(r"^(status|phase|depends_on):\s*(.*)$", line)
        if not k:
            continue
        if k.group(1) == "depends_on":
            out["depends"] = re.findall(r"T-\d+", k.group(2))
        else:
            out[k.group(1)] = k.group(2).strip()
    return out


def pr_map(prs: list[dict[str, object]]) -> dict[str, tuple[int, str]]:
    """ticket id -> (pr number, state); a branch t-0NN-* maps to T-0NN; highest number wins."""
    out: dict[str, tuple[int, str]] = {}
    for p in prs:
        m = re.match(r"^t-(\d+)-", str(p.get("headRefName", "")))
        if not m:
            continue
        num = int(str(p["number"]))
        tid = f"T-{m.group(1)}"
        if tid not in out or num > out[tid][0]:
            out[tid] = (num, str(p.get("state", "")))
    return out


def _public_title(title: str) -> str:
    title = re.sub(r"evidence map", "evidence table", title, flags=re.I)
    title = title.replace("REVIEW.md", "review document")
    title = title.replace("|", "\\|").replace("`", "'")
    return " ".join(title.split())


def block_tickets(
    tickets: list[dict[str, object]], prs: list[dict[str, object]], in_progress: set[str], repo: str
) -> str:
    pm = pr_map(prs)
    L = [
        "| Ticket | Title | Phase | State | Depends on | PR |",
        "| --- | --- | --- | --- | --- | --- |",
    ]
    for t in sorted(tickets, key=lambda x: str(x["id"])):
        tid = str(t["id"])
        pr = pm.get(tid)
        if pr and pr[1] == "MERGED":
            state = "merged"
        elif pr and pr[1] == "OPEN":
            state = "in review"
        elif tid in in_progress:
            state = "in progress"
        elif t["status"] == "done":
            state = "merged"
        else:
            state = "planned"
        link = f"[#{pr[0]}](https://github.com/{repo}/pull/{pr[0]})" if pr else "-"
        deps = ", ".join(str(d) for d in t["depends"]) if t["depends"] else "-"  # type: ignore[attr-defined]
        L.append(
            f"| {tid} | {_public_title(str(t['title']))} | {t['phase'] or '-'} | {state} | {deps} | {link} |"
        )
    return "\n".join(L)


# ---------------------------------------------------------------- driver


def compute_blocks(
    root: Path, tickets_dir: Path | None, prs_file: Path | None, in_progress: set[str], repo: str
) -> dict[str, str]:
    blocks = {"built": block_built(root), "adrs": block_adrs(root), "evals": block_evals(root)}
    if (
        tickets_dir is not None
        and tickets_dir.is_dir()
        and prs_file is not None
        and prs_file.is_file()
    ):
        tickets = [t for f in sorted(tickets_dir.glob("T-*.md")) if (t := parse_ticket(f))]
        prs = json.loads(prs_file.read_text(encoding="utf-8"))
        if tickets:
            blocks["tickets"] = block_tickets(tickets, prs, in_progress, repo)
    return blocks


def check(text: str, blocks: dict[str, str]) -> str | None:
    for name, body in blocks.items():
        cur = get_block(text, name)
        if cur is None or cur.strip() != body.strip():
            return name
    return None


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument(
        "--write", action="store_true", help="rewrite the generated blocks in docs/status.md"
    )
    ap.add_argument("--check", action="store_true", help="exit 1 naming the first stale block")
    ap.add_argument(
        "--tickets", default=".planning/tickets", help="directory of local ticket files (optional)"
    )
    ap.add_argument(
        "--prs",
        default=None,
        help="JSON from `gh pr list --state all --json number,title,headRefName,state,mergedAt`",
    )
    ap.add_argument(
        "--in-progress", default="", help="comma separated ticket ids to show as in progress"
    )
    ap.add_argument("--repo", default=DEFAULT_REPO)
    ap.add_argument("--root", default=str(ROOT), help=argparse.SUPPRESS)
    a = ap.parse_args(argv)
    root = Path(a.root)
    tdir = Path(a.tickets) if Path(a.tickets).is_absolute() else root / a.tickets
    prs = Path(a.prs) if a.prs else None
    ip = {s.strip() for s in a.in_progress.split(",") if s.strip()}
    blocks = compute_blocks(root, tdir, prs, ip, a.repo)
    doc = root / DOC
    text = doc.read_text(encoding="utf-8")
    try:
        validate_markers(text)
        missing = next((n for n in blocks if get_block(text, n) is None), None)
    except ValueError as e:
        print(f"status doc markers invalid: {e}", file=sys.stderr)
        return 1
    if missing and (a.check or a.write):
        print(f"missing marker for block: {missing}", file=sys.stderr)
        return 1
    if a.check:
        stale = check(text, blocks)
        if stale:
            print(f"stale block: {stale} (run: make status)", file=sys.stderr)
            return 1
        return 0
    if a.write:
        for name, body in blocks.items():
            text = replace_block(text, name, body)
        doc.write_text(text, encoding="utf-8")
        return 0
    ap.print_usage(sys.stderr)
    return 2


if __name__ == "__main__":
    sys.exit(main())
