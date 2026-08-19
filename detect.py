#!/usr/bin/env python3
"""Detect faces in photos with a Coral Edge TPU and save cropped faces.

Resume-safe: already-processed images are skipped via the JSONL manifest.
"""
import argparse
import json
import sys
from pathlib import Path

from PIL import Image
from pycoral.adapters import common, detect
from pycoral.utils.edgetpu import make_interpreter

IMG_EXTS = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}


def iter_images(root):
    for p in sorted(root.rglob("*")):
        if p.is_file() and p.suffix.lower() in IMG_EXTS:
            yield p


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("input", type=Path,
                    help="Folder of photos (e.g. Google Takeout export)")
    ap.add_argument("--model", type=Path, default=Path(
        "models/ssd_mobilenet_v2_face_quant_postprocess_edgetpu.tflite"))
    ap.add_argument("--out", type=Path, default=Path("output"))
    ap.add_argument("--threshold", type=float, default=0.6)
    ap.add_argument("--margin", type=float, default=0.35,
                    help="Pad each side of the box by this fraction of its size")
    ap.add_argument("--min-face", type=int, default=40,
                    help="Minimum crop side in pixels")
    args = ap.parse_args()

    crops_dir = args.out / "crops"
    crops_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = args.out / "detections.jsonl"

    done = set()
    if manifest_path.exists():
        with manifest_path.open() as f:
            for line in f:
                done.add(json.loads(line)["image"])

    interpreter = make_interpreter(str(args.model))
    interpreter.allocate_tensors()

    todo = [p for p in iter_images(args.input) if str(p) not in done]
    print(f"{len(todo)} images to process, {len(done)} already done")

    n_faces = 0
    with manifest_path.open("a") as manifest:
        for i, img_path in enumerate(todo, 1):
            try:
                img = Image.open(img_path).convert("RGB")
            except Exception as e:
                print(f"skip {img_path}: {e}", file=sys.stderr)
                continue

            scale = common.set_resized_input(
                interpreter, img.size,
                lambda size: img.resize(size, Image.BILINEAR))
            interpreter.invoke()
            objs = detect.get_objects(interpreter, args.threshold, scale)

            faces = []
            for j, o in enumerate(objs):
                b = o.bbox
                mx, my = b.width * args.margin, b.height * args.margin
                box = (max(0, int(b.xmin - mx)), max(0, int(b.ymin - my)),
                       min(img.width, int(b.xmax + mx)),
                       min(img.height, int(b.ymax + my)))
                if box[2] - box[0] < args.min_face or box[3] - box[1] < args.min_face:
                    continue
                name = f"{img_path.stem}_{j}.jpg"
                img.crop(box).save(crops_dir / name, quality=95)
                faces.append({"crop": name, "bbox": list(box),
                              "score": round(float(o.score), 3)})

            n_faces += len(faces)
            manifest.write(json.dumps({"image": str(img_path), "faces": faces}) + "\n")
            if i % 100 == 0:
                print(f"{i}/{len(todo)} images, {n_faces} faces so far")
                manifest.flush()

    print(f"done: {n_faces} faces from {len(todo)} images -> {crops_dir}")


if __name__ == "__main__":
    main()
