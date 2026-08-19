#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"
mkdir -p models

echo ">> Face detection model (Edge TPU)"
curl -fSL -o models/ssd_mobilenet_v2_face_quant_postprocess_edgetpu.tflite \
  https://github.com/google-coral/test_data/raw/master/ssd_mobilenet_v2_face_quant_postprocess_edgetpu.tflite

cat <<'EOF'

>> Face embedding model
The Coral project does not ship an official face-embedding model.
Get a quantized FaceNet / MobileFaceNet TFLite (112x112 or 160x160 input),
then compile it for the Edge TPU:

    edgetpu_compiler -s your_facenet_model.tflite
    mv your_facenet_model_edgetpu.tflite models/facenet_edgetpu.tflite

Or skip compilation and run embeddings on CPU instead:

    python3 embed_cluster.py --cpu --model your_facenet_model.tflite
EOF
