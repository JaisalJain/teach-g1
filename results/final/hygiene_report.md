# Repo hygiene report (scripts/hygiene.py, read only)

- Commit checked: `7035420`. Tracked files: 8493.
- Tracked files over 10 MB: `data/raw/4e103843db/rgb.mp4` 10.68 MB
- The four raw clips are committed on purpose (data/raw/<id>/, rgb.mp4 included; every other *.mp4 stays gitignored). GitHub's hard limit is 100 MB per file.
- Secret-looking strings in tracked text files: none found
- Links in README.md, docs/INSTALL.md, docs/details.md, docs/log.md (49 relative links) to files git does not track or that do not exist: none
- Top-level LICENSE: present
- Fresh clone of HEAD (local `git clone --no-hardlinks`): 342.2 MB in total, of which .git 143.7 MB
