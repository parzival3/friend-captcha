#!/usr/bin/env python3
"""Compute face embeddings and cluster crops into per-person folders.

Embeddings are cached in embeddings.npz, so re-clustering with different
DBSCAN parameters (--recluster) doesn't re-run the model.
"""
import argparse
import shutil
from pathlib import Path

import numpy as np
from PIL import Image


def load_interpreter(model_path, use_edgetpu):
    if use_edgetpu:
        from pycoral.utils.edgetpu import make_interpreter
        return make_interpreter(str(model_path))
    import tflite_runtime.interpreter as tflite
    return tflite.Interpreter(str(model_path))


def embed(interpreter, img):
    in_det = interpreter.get_input_details()[0]
    out_det = interpreter.get_output_details()[0]
    _, h, w, _ = in_det["shape"]

    x = np.asarray(img.resize((w, h), Image.BILINEAR), dtype=np.float32)
    x = x / 127.5 - 1.0  # [-1, 1], standard for FaceNet-style models
    if in_det["dtype"] == np.uint8:
        scale, zero = in_det["quantization"]
        if scale == 0.0:
            x = (x + 1.0) * 127.5
        else:
            x = x / scale + zero
        x = np.round(x).clip(0, 255).astype(np.uint8)

    interpreter.set_tensor(in_det["index"], x[None])
    interpreter.invoke()

    y = interpreter.get_tensor(out_det["index"])[0].astype(np.float32)
    if out_det["dtype"] == np.uint8:
        scale, zero = out_det["quantization"]
        if scale != 0.0:
            y = (y - zero) * scale
    y = y.ravel()
    return y / (np.linalg.norm(y) + 1e-10)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--crops", type=Path, default=Path("output/crops"))
    ap.add_argument("--model", type=Path, default=Path("models/facenet_edgetpu.tflite"))
    ap.add_argument("--cpu", action="store_true",
                    help="Run the embedding model on CPU instead of Edge TPU")
    ap.add_argument("--out", type=Path, default=Path("output"))
    ap.add_argument("--eps", type=float, default=0.35,
                    help="DBSCAN eps on cosine distance (lower = stricter)")
    ap.add_argument("--min-samples", type=int, default=2)
    ap.add_argument("--recluster", action="store_true",
                    help="Reuse cached embeddings.npz, only re-run clustering")
    args = ap.parse_args()

    from sklearn.cluster import DBSCAN

    npz_path = args.out / "embeddings.npz"
    if args.recluster and npz_path.exists():
        data = np.load(npz_path, allow_pickle=True)
        files, embs = data["files"], data["embs"]
        print(f"loaded {len(files)} cached embeddings")
    else:
        crops = sorted(p for p in args.crops.iterdir()
                       if p.suffix.lower() in {".jpg", ".jpeg", ".png"})
        print(f"embedding {len(crops)} crops...")
        interpreter = load_interpreter(args.model, not args.cpu)
        interpreter.allocate_tensors()
        vecs = []
        for i, p in enumerate(crops, 1):
            try:
                vecs.append(embed(interpreter, Image.open(p).convert("RGB")))
            except Exception as e:
                print(f"skip {p.name}: {e}")
                crops[i - 1] = None
            if i % 500 == 0:
                print(f"{i}/{len(crops)}")
        pairs = [(p, v) for p, v in zip(crops, vecs) if p is not None]
        files = np.array([p.name for p, _ in pairs])
        embs = np.stack([v for _, v in pairs])
        np.savez_compressed(npz_path, files=files, embs=embs)

    labels = DBSCAN(eps=args.eps, min_samples=args.min_samples,
                    metric="cosine").fit_predict(embs)

    people_dir = args.out / "people"
    if people_dir.exists():
        shutil.rmtree(people_dir)
    people_dir.mkdir(parents=True)

    counts = {}
    for name, lab in zip(files, labels):
        folder = "unknown" if lab == -1 else f"cluster_{lab:03d}"
        d = people_dir / folder
        d.mkdir(exist_ok=True)
        shutil.copy2(args.crops / name, d / name)
        counts[folder] = counts.get(folder, 0) + 1

    for folder, n in sorted(counts.items(), key=lambda kv: -kv[1]):
        print(f"{folder}: {n} faces")
    print(f"-> {people_dir} (now run label.py to name the clusters)")


if __name__ == "__main__":
    main()
