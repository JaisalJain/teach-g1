"""Repo hygiene report from the git index (read only): tracked files over 10 MB, secret-looking strings in tracked text files,
links in README.md and docs/*.md that point at files git does not track (or that do not exist), whether a top-level LICENSE
exists, and the size of a fresh clone (a local `git clone` of HEAD into a temporary folder, measured, then deleted).
Run from the repo root: python3 scripts/hygiene.py  ->  results/final/hygiene_report.md"""

import re
import subprocess
import tempfile
from pathlib import Path

ROOT = Path.cwd().resolve()
tracked = subprocess.run(["git", "ls-files"], capture_output=True, text=True, check=True).stdout.splitlines()
tracked_set = set(tracked)
big = [(f, Path(f).stat().st_size) for f in tracked if Path(f).is_file() and Path(f).stat().st_size > 10_000_000]
SECRET = re.compile(r"(AKIA[0-9A-Z]{16}|ghp_[0-9A-Za-z]{30,}|sk-[0-9A-Za-z]{20,}|xox[bpa]-[0-9A-Za-z-]{10,}|-----BEGIN [A-Z ]*PRIVATE KEY-----|"
                    r"(?i:(api[_-]?key|secret|password|passwd|token)\s*[=:]\s*['\"][^'\"]{8,}['\"]))")
hits = []
for f in tracked:
    p = Path(f)
    if p.suffix.lower() in {".py", ".md", ".sh", ".txt", ".yaml", ".yml", ".json", ".csv", ".cfg", ".toml"} and p.is_file() and p.stat().st_size < 5_000_000:
        for i, line in enumerate(p.read_text(errors="ignore").splitlines(), 1):
            if SECRET.search(line):
                hits.append(f"{f}:{i}: {line.strip()[:120]}")

# links (markdown [text](target) and ![alt](target)) in README.md and docs/*.md, resolved relative to each file
docs = ["README.md"] + sorted(str(p) for p in Path("docs").glob("*.md"))
bad_links, n_links = [], 0
for doc in docs:
    text = Path(doc).read_text()
    for target in sorted(set(re.findall(r"\]\(([^)\s]+)\)", text))):
        target = target.split("#")[0]
        if not target or target.startswith(("http://", "https://", "mailto:")):
            continue
        n_links += 1
        path = (Path(doc).parent / target).resolve()
        try:
            rel = str(path.relative_to(ROOT))
        except ValueError:
            bad_links.append(f"{doc}: {target} (outside the repo)")
            continue
        if rel not in tracked_set and not any(t.startswith(rel + "/") for t in tracked):
            bad_links.append(f"{doc}: {target} -> {rel} ({'exists, but gitignored / untracked' if path.exists() else 'missing'})")

# size of a fresh clone of HEAD
with tempfile.TemporaryDirectory() as tmp:
    subprocess.run(["git", "clone", "-q", "--no-hardlinks", str(ROOT), f"{tmp}/clone"], check=True)
    du = lambda p: int(subprocess.run(["du", "-sb", p], capture_output=True, text=True, check=True).stdout.split()[0])
    clone_total, clone_git = du(f"{tmp}/clone"), du(f"{tmp}/clone/.git")
head = subprocess.run(["git", "rev-parse", "--short", "HEAD"], capture_output=True, text=True, check=True).stdout.strip()

lines = ["# Repo hygiene report (scripts/hygiene.py, read only)", "",
         f"- Commit checked: `{head}`. Tracked files: {len(tracked)}.",
         "- Tracked files over 10 MB: " + ("; ".join(f"`{f}` {s / 1e6:.2f} MB" for f, s in big) or "none"),
         "- The four raw clips are committed on purpose (data/raw/<id>/, rgb.mp4 included; every other *.mp4 stays gitignored). "
         "GitHub's hard limit is 100 MB per file.",
         "- Secret-looking strings in tracked text files: " + ("\n  - " + "\n  - ".join(hits) if hits else "none found"),
         f"- Links in {', '.join(docs)} ({n_links} relative links) to files git does not track or that do not exist: "
         + ("\n  - " + "\n  - ".join(bad_links) if bad_links else "none"),
         f"- Top-level LICENSE: {'present' if Path('LICENSE').exists() and 'LICENSE' in tracked_set else 'absent or untracked'}",
         f"- Fresh clone of HEAD (local `git clone --no-hardlinks`): {clone_total / 1e6:.1f} MB in total, of which .git {clone_git / 1e6:.1f} MB"]
Path("results/final/hygiene_report.md").write_text("\n".join(lines) + "\n")
print("\n".join(lines))
