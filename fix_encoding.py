"""Converts text files written in Windows cp1252 to UTF-8 and checks the notebooks. Run from the heatwave-delhi folder:  python fix_encoding.py"""
import json
from pathlib import Path

root = Path.cwd()
patterns = ["src/*.py", "config/*.json", "reports/*.md", "reports/*/*.md", "models/*.json", "data/processed/*.json", "README.md"]
fixed = checked = 0
for pat in patterns:
    for p in root.glob(pat):
        checked += 1
        raw = p.read_bytes()
        try:
            raw.decode("utf-8")
        except UnicodeDecodeError:
            p.write_text(raw.decode("cp1252", errors="replace"), encoding="utf-8")
            fixed += 1
            print("converted to UTF-8:", p.relative_to(root))
print(f"checked {checked} files, converted {fixed}")

for nbname in ("02_model_training.ipynb", "03_xgboost_training.ipynb"):
    nb = root / "notebooks" / nbname
    if nb.exists():
        n = nb.read_text(encoding="utf-8").count('encoding=\\"utf-8\\"')
        print(f"{nbname}: {'UPDATED version' if n else 'OLD version -> replace it with the new file'} ({n} utf-8 markers)")
for p in (root / "src").glob("*.py"):
    compile(p.read_bytes(), str(p), "exec")
print("src/*.py all compile OK")
