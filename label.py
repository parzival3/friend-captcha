#!/usr/bin/env python3
"""Label clustered faces: build contact sheets, then name each cluster."""
import argparse
import math
import subprocess
from pathlib import Path

from PIL import Image, ImageDraw

IMG_EXTS = {".jpg", ".jpeg", ".png"}


def make_sheet(cluster_dir, out_path, thumb=128, cols=5, max_faces=25):
    files = sorted(p for p in cluster_dir.iterdir()
                   if p.suffix.lower() in IMG_EXTS)[:max_faces]
    if not files:
        return False
    rows = math.ceil(len(files) / cols)
    sheet = Image.new("RGB", (cols * thumb, rows * thumb + 24), "white")
    ImageDraw.Draw(sheet).text((4, 4), cluster_dir.name, fill="black")
    for i, f in enumerate(files):
        im = Image.open(f).convert("RGB")
        im.thumbnail((thumb, thumb))
        x, y = (i % cols) * thumb, 24 + (i // cols) * thumb
        sheet.paste(im, (x + (thumb - im.width) // 2,
                         y + (thumb - im.height) // 2))
    sheet.save(out_path)
    return True


def rename_cluster(people_dir, cluster, name):
    src = people_dir / cluster
    dst = people_dir / name
    if not src.is_dir():
        raise SystemExit(f"no such cluster: {src}")
    if dst.exists():
        raise SystemExit(f"target already exists: {dst}")
    src.rename(dst)
    print(f"{cluster} -> {name}")


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--people", type=Path, default=Path("output/people"))
    ap.add_argument("--sheets", type=Path, default=Path("output/sheets"))
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("sheets", help="Build a contact sheet per cluster")
    p_name = sub.add_parser("name", help="Rename a cluster folder")
    p_name.add_argument("cluster")
    p_name.add_argument("name")
    p_inter = sub.add_parser("interactive", help="Sheet + prompt per cluster")
    p_inter.add_argument("--open", action="store_true",
                         help="Open each sheet with xdg-open")
    args = ap.parse_args()

    clusters = sorted(d for d in args.people.iterdir()
                      if d.is_dir() and d.name.startswith(("cluster_", "unknown")))

    if args.cmd == "sheets":
        args.sheets.mkdir(parents=True, exist_ok=True)
        for d in clusters:
            if make_sheet(d, args.sheets / f"{d.name}.jpg"):
                print(args.sheets / f"{d.name}.jpg")
    elif args.cmd == "name":
        rename_cluster(args.people, args.cluster, args.name)
    else:
        args.sheets.mkdir(parents=True, exist_ok=True)
        for d in clusters:
            sheet = args.sheets / f"{d.name}.jpg"
            if not make_sheet(d, sheet):
                continue
            print(f"\n{d.name} ({sum(1 for _ in d.iterdir())} faces): {sheet}")
            if args.open:
                subprocess.Popen(["xdg-open", str(sheet)],
                                 stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            name = input("name (empty = skip): ").strip()
            if name:
                rename_cluster(args.people, d.name, name)


if __name__ == "__main__":
    main()
