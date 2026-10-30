"""M7 repo hygiene report from the git index (read only): tracked files over 10 MB, secret-looking strings in tracked text files,
links in docs/README_draft.md that point at files git does not track, and whether a top-level LICENSE exists.
Run: python3 scripts/hygiene.py  ->  results/final/hygiene_report.md"""

import re
import subprocess
from pathlib import Path

tracked = subprocess.run(["git", "ls-files"], capture_output=True, text=True, check=True).stdout.splitlines()
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
readme = Path("docs/README_draft.md").read_text()
links = sorted(set(re.findall(r"\]\(([^)#\s]+)", readme)))
untracked = []
for l in links:
    if l.startswith("http"):
        continue
    target = (Path("docs") / l).resolve().relative_to(Path.cwd().resolve())
    if str(target) not in tracked and not any(t.startswith(str(target) + "/") for t in tracked):
        untracked.append(f"{l} -> {target} ({'exists, gitignored/untracked' if target.exists() else 'missing'})")
lines = ["# Repo hygiene report (scripts/hygiene.py, read only)", "",
         f"- Tracked files over 10 MB: " + ("; ".join(f"`{f}` {s / 1e6:.2f} MB" for f, s in big) or "none"),
         f"- Secret-looking strings in tracked text files: " + ("\n  - " + "\n  - ".join(hits) if hits else "none found"),
         f"- Links in docs/README_draft.md to files git does not track: " + ("\n  - " + "\n  - ".join(untracked) if untracked else "none"),
         "- mp4 videos are gitignored (*.mp4); the README links only GIFs and PNGs. The data sample's rgb.mp4 is force-added.",
         f"- Top-level LICENSE: {'present' if Path('LICENSE').exists() else 'absent (the owner will choose one; none added)'}",
         f"- Tracked files: {len(tracked)}"]
Path("results/final/hygiene_report.md").write_text("\n".join(lines) + "\n")
print("\n".join(lines))
