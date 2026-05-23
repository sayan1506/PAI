"""Create a clean source zip for uploading to AI — includes .env, excludes packages."""
import zipfile
import os

SKIP_DIRS = {'.git', '.hypothesis', '__pycache__', 'node_modules',
             '.venv', 'venv', 'env', 'site-packages', '.kiro'}
SKIP_FILES = {'_make_zip.py', 'PAI-source.zip'}
SKIP_EXT = {'.pyc', '.pyo', '.zip'}

zf = zipfile.ZipFile('PAI-source.zip', 'w', zipfile.ZIP_DEFLATED)
count = 0

for root, dirs, files in os.walk('.'):
    dirs[:] = [d for d in dirs if d not in SKIP_DIRS]
    for f in files:
        if f in SKIP_FILES:
            continue
        if any(f.endswith(ext) for ext in SKIP_EXT):
            continue
        filepath = os.path.join(root, f)
        arcname = os.path.relpath(filepath, '.')
        zf.write(filepath, arcname)
        count += 1

zf.close()
size_kb = os.path.getsize('PAI-source.zip') / 1024
print(f"Done! PAI-source.zip — {count} files, {size_kb:.1f} KB")
