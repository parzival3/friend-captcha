# friend-captcha

Build a fun fake CAPTCHA ("Select all images with **Alice**") from your own
photo library, using a Google Coral Edge TPU to detect and recognize your
friends' faces.

## Pipeline

```
photos/  -->  detect.py  -->  output/crops/  -->  embed_cluster.py
  -->  output/people/cluster_000/ ...  -->  label.py
  -->  output/people/Alice/ Bob/ ...  -->  captcha_gen.py  -->  site/index.html
```

## Setup

1. Export your Google Photos via [Google Takeout](https://takeout.google.com)
   and unpack them somewhere, e.g. `~/photos/`.
2. Install the [Coral runtime](https://coral.ai/docs/accelerator/get-started/)
   (pycoral + tflite-runtime), then:
   ```sh
   pip install -r requirements.txt
   ./download_models.sh
   ```
3. Provide a quantized FaceNet/MobileFaceNet TFLite embedding model
   (see the notes printed by `download_models.sh`).

## Usage

```sh
# 1. Detect faces on the Coral and crop them (resume-safe)
python3 detect.py ~/photos/

# 2. Embed crops and cluster by identity
python3 embed_cluster.py                 # Edge TPU
python3 embed_cluster.py --recluster --eps 0.30   # tune clustering, no recompute

# 3. Look at each cluster and give it a name
python3 label.py sheets                  # contact sheets -> output/sheets/
python3 label.py name cluster_000 Alice
python3 label.py interactive --open      # or do it interactively

# 4. Generate the fake CAPTCHA
python3 captcha_gen.py --person Alice
xdg-open site/index.html
```

Re-run `captcha_gen.py` (optionally with `--seed`) for a fresh grid.

## Notes

- `detect.py` writes `output/detections.jsonl`; delete it to reprocess everything.
- `embed_cluster.py` caches embeddings in `output/embeddings.npz`.
- Faces that don't cluster cleanly land in `output/people/unknown/`.
- **Privacy:** keep this repo private and never commit `output/` — it
  contains your friends' faces (`.gitignore` already covers it).
