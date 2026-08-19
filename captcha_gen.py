#!/usr/bin/env python3
"""Generate a fake CAPTCHA page: 'Select all images with <name>'."""
import argparse
import html
import json
import random
import shutil
from pathlib import Path

IMG_EXTS = {".jpg", ".jpeg", ".png"}

PAGE = """<!doctype html>
<html><head><meta charset="utf-8"><title>CAPTCHA</title>
<style>
  body {{ font-family: Roboto, Arial, sans-serif; background:#f9f9f9;
         display:flex; justify-content:center; padding-top:40px; }}
  .captcha {{ width:480px; background:#fff; border-radius:8px;
             box-shadow:0 2px 10px rgba(0,0,0,.2); overflow:hidden; }}
  .header {{ background:#1a73e8; color:#fff; padding:20px; }}
  .header small {{ display:block; opacity:.85; }}
  .header b {{ font-size:22px; }}
  .grid {{ display:grid; grid-template-columns:repeat({grid},1fr);
          gap:4px; padding:8px; }}
  .tile {{ cursor:pointer; position:relative; }}
  .tile img {{ width:100%; aspect-ratio:1; object-fit:cover; display:block;
              border-radius:4px; transition:transform .1s; }}
  .tile.sel img {{ transform:scale(.85); outline:4px solid #1a73e8; }}
  .tile.sel::after {{ content:"\\2713"; position:absolute; top:6px; left:6px;
     background:#1a73e8; color:#fff; width:24px; height:24px;
     border-radius:50%; text-align:center; line-height:24px; }}
  .footer {{ display:flex; justify-content:space-between; align-items:center;
            padding:12px 16px; border-top:1px solid #eee; }}
  #result {{ font-weight:bold; min-height:1.2em; }}
  button {{ background:#1a73e8; color:#fff; border:0; border-radius:4px;
           padding:10px 24px; font-size:14px; cursor:pointer; }}
  button:hover {{ background:#1765cc; }}
</style></head><body>
<div class="captcha">
  <div class="header"><small>Select all images with</small><b>{name}</b>
    <small>Click verify once there are none left.</small></div>
  <div class="grid">{tiles}</div>
  <div class="footer"><span id="result"></span>
    <button onclick="verify()">Verify</button></div>
</div>
<script>
const targets = new Set({targets});
document.querySelectorAll('.tile').forEach(t =>
  t.onclick = () => t.classList.toggle('sel'));
function verify() {{
  const sel = new Set([...document.querySelectorAll('.tile.sel')]
    .map(t => +t.dataset.i));
  const ok = sel.size === targets.size && [...sel].every(i => targets.has(i));
  const r = document.getElementById('result');
  r.textContent = ok ? "\\u2705 Correct!" : "\\u274c Try again";
  r.style.color = ok ? "#188038" : "#d93025";
}}
</script></body></html>
"""


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--people", type=Path, default=Path("output/people"))
    ap.add_argument("--person", required=True, help="Name of the target friend")
    ap.add_argument("--grid", type=int, default=3)
    ap.add_argument("--targets", type=int, default=3,
                    help="Number of correct tiles in the grid")
    ap.add_argument("--out", type=Path, default=Path("site"))
    ap.add_argument("--seed", type=int, default=None)
    args = ap.parse_args()

    rng = random.Random(args.seed)
    target_dir = args.people / args.person
    if not target_dir.is_dir():
        avail = ", ".join(sorted(d.name for d in args.people.iterdir() if d.is_dir()))
        raise SystemExit(f"unknown person '{args.person}'. Available: {avail}")

    n_tiles = args.grid * args.grid
    targets = sorted(p for p in target_dir.iterdir() if p.suffix.lower() in IMG_EXTS)
    others = [p for d in args.people.iterdir()
              if d.is_dir() and d != target_dir
              for p in sorted(d.iterdir()) if p.suffix.lower() in IMG_EXTS]

    n_targets = min(args.targets, len(targets))
    if len(others) < n_tiles - n_targets:
        raise SystemExit("not enough distractor faces; label more friends first")

    picks = ([(p, True) for p in rng.sample(targets, n_targets)] +
             [(p, False) for p in rng.sample(others, n_tiles - n_targets)])
    rng.shuffle(picks)

    img_dir = args.out / "img"
    img_dir.mkdir(parents=True, exist_ok=True)
    tiles, target_idx = [], []
    for i, (src, is_target) in enumerate(picks):
        dst = img_dir / f"{i:02d}{src.suffix}"
        shutil.copy2(src, dst)
        if is_target:
            target_idx.append(i)
        tiles.append(f'<div class="tile" data-i="{i}">'
                     f'<img src="img/{dst.name}"></div>')

    page = PAGE.format(grid=args.grid, name=html.escape(args.person),
                       tiles="\n".join(tiles), targets=json.dumps(target_idx))
    (args.out / "index.html").write_text(page)
    print(f"-> {args.out / 'index.html'} ({n_targets} targets among {n_tiles} tiles)")


if __name__ == "__main__":
    main()
