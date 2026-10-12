ONNX files for [Clef-Flash-ONNX-CPU](https://github.com/Kazuhito00/Clef-Flash-ONNX-CPU): Cloudflare clef-flash (`Cloudflare/clef-flash`, 9B, Apache-2.0) on onnxruntime + numpy.

Fetch them with `uv run download_model.py` (SHA-256 checked against `manifest.json`; split files are joined). Do not download the assets by hand: each safetensors shard is split into 2 to 3 parts.

These are converted files (see "About the converted files" in the README): the weight-free ONNX graph published by ollaya (`ollaya-dev/clef`, Apache-2.0 per its model card) with its weight references rewritten to the file names below; the weights are Cloudflare's own bf16 safetensors at revision `17f0b0ad64efb65d273590632833508766b2aae6`, byte for byte, split into parts to stay under GitHub's 2 GiB asset limit; the Qwen3.5 vision tower exported as a separate graph, and the text graph with three extra image inputs (graph surgery only). Decisions match Cloudflare's reference code (text and images: 0 differing decisions).

Licences: `LICENSE` (Apache-2.0, the file in Cloudflare/clef-flash, unchanged).

| Asset | Content |
|---|---|
| `model-0000{1..4}-of-00004.safetensors.part000..` | Cloudflare's bf16 weights (4 shards, 2 to 3 parts each), 18 GB |
| `joint_head.safetensors` | Cloudflare's joint schema head (244 MB) |
| `model-fp32.onnx`, `model-fp32-vl.onnx` | weight-free graphs (text; with image inputs), 12 MB each |
| `vision_encoder.onnx`, `vision_encoder.onnx.data` | vision tower (1.8 GB of weights) |
| `flash__decision.json`, `flash__calibration.json` | sequence layout and calibration (stored under `flash/`) |
| `tokenizer.json`, `tokenizer_config.json`, `config.json`, `joint_head_config.json`, `processor_config.json`, `generation_config.json`, `chat_template.jinja`, `model.safetensors.index.json`, `joint_schema_model.py`, `LICENSE` | Cloudflare's files, unchanged |
| `manifest.json`, `SHA256SUMS` | file list with sizes and SHA-256, and how the parts are joined |

Memory: about 14.5 GB at peak on the CPU. The model does not fit on a 4 GB GPU (CUDA out of memory); GPU speed is unverified.
