"""Task 1.2 - VisionGuard dataset audit.

Walks data/raw and records, for every file: class, file size, width, height,
colour mode, real format and SHA-256 hash. Flags unreadable files and exact
duplicates, prints the report tables and saves reports/audit.csv.
Run from the project root:  python src/audit.py
"""
import hashlib
from pathlib import Path

import pandas as pd
from PIL import Image

RAW = Path("data/raw")
OUT = Path("reports/audit.csv")
EXPECTED_EXT = {"JPEG": {".jpg", ".jpeg"}, "PNG": {".png"},
                "WEBP": {".webp"}, "GIF": {".gif"}}
pd.set_option("display.width", 200)


def sha256(path):
    """Content fingerprint: identical bytes give an identical hash."""
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def inspect(path):
    rec = {
        "path": path.as_posix(),                   # forward slashes: works on Linux too
        "cls": path.relative_to(RAW).parts[0],     # label = first folder under data/raw
        "extension": path.suffix.lower() or "(none)",
        "size_bytes": path.stat().st_size,
        "sha256": sha256(path),
        "width": None, "height": None, "mode": None, "format": None,
        "corrupt": False, "error": "",
    }
    try:
        with Image.open(path) as im:
            im.verify()                            # checks the file structure
        with Image.open(path) as im:               # verify() closes the image, so reopen
            im.load()                              # fully decodes the pixels
            rec.update(width=im.width, height=im.height,
                       mode=im.mode, format=im.format)
    except Exception as e:
        rec["corrupt"] = True
        rec["error"] = f"{type(e).__name__}: {e}"
    return rec


def main():
    files = sorted(p for p in RAW.rglob("*") if p.is_file())
    df = pd.DataFrame([inspect(p) for p in files])
    OUT.parent.mkdir(exist_ok=True)
    df.to_csv(OUT, index=False)
    ok = df[~df.corrupt]

    print(f"Files scanned: {len(df)}   (table saved to {OUT})\n")
    print("== Images per class ==")
    print(df.cls.value_counts().to_string(), "\n")

    print("== Width, height (px) and file size (bytes): min / median / max ==")
    stats = ok.groupby("cls")[["width", "height", "size_bytes"]].agg(["min", "median", "max"])
    print(stats.T.to_string(), "\n")

    print(f"== Unreadable (corrupt) files: {int(df.corrupt.sum())} ==")
    if df.corrupt.any():
        print(df[df.corrupt][["cls", "path", "error"]].to_string(index=False))
    print()

    dup = df[df.duplicated("sha256", keep=False)].sort_values("sha256")
    n_groups = dup.sha256.nunique()
    cross = int((dup.groupby("sha256").cls.nunique() > 1).sum()) if n_groups else 0
    print(f"== Exact duplicates: {n_groups} groups, {len(dup)} files, "
          f"{len(dup) - n_groups} extra copies, {cross} groups span both classes ==")
    if n_groups:
        print(dup[["sha256", "cls", "path"]].assign(sha256=dup.sha256.str[:12])
              .to_string(index=False))
    print()

    print("== Colour modes per class ==")
    print(pd.crosstab(ok.cls, ok["mode"]).to_string(), "\n")
    print("== Real format (from file content) per class ==")
    print(pd.crosstab(ok.cls, ok.format).to_string(), "\n")

    mismatch = ok[[e not in EXPECTED_EXT.get(f, set()) for e, f in zip(ok.extension, ok.format)]]
    print(f"== Extension does not match real format: {len(mismatch)} ==")
    if len(mismatch):
        print(mismatch[["cls", "extension", "format", "path"]].to_string(index=False))


if __name__ == "__main__":
    main()
