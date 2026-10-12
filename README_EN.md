[[Japanese](README.md)/[English](README_EN.md)]

# Clef-Flash-ONNX-CPU

A minimal-dependency inference implementation of [`Cloudflare/clef-flash`](https://huggingface.co/Cloudflare/clef-flash) (a 9B structured decision model; text and images) that needs only onnxruntime and numpy. The device can be switched to GPU (see "Running on a GPU"; this 9B model does not fit on common GPUs, though).<br>
It does not use torch or transformers, which the official `joint_schema_model.py` requires.

# Features
- Minimal dependencies: onnxruntime and numpy at run time (plus pillow for images)
- Two scripts are enough: `download_model.py` (fetch the model) and `demo_inference_text.py` (inference). Images: `demo_inference_image.py`; camera: `demo_inference_webcam.py`
- Decisions match the reference (Cloudflare's own code, bf16, CPU): 0 differing decisions on text and on images, identical token ids
- All questions are decided in one forward pass (nothing is generated). `noul` (yes/no), `choice` and `score` questions can be mixed in one request
- Image input (up to 4 per request; each is scaled down to at most 448 px on its longer side)
- Pure-Python ports of the preprocessing (Qwen byte-level BPE, prompt assembly, 3-axis M-RoPE, image patching and positions), with cross-checks against the reference
- `--device cuda` switches to a GPU (with `--require-gpu`, a silent fallback to the CPU is an error)

# Purpose of This Repository
- How far can clef-flash decisions go with 2 to 3 runtime dependencies?
- Can the reference preprocessing be ported to pure Python and produce identical inputs?
- How far can a 9B model be handled on a CPU (memory)?

# Requirements
```
Python 3.10 or later

numpy        1.26 or later
onnxruntime  1.20 or later
pillow       10.0 or later   (images only)
```
That is all that is needed to run. Fetching the model uses the standard library only. Environments are managed with [uv](https://docs.astral.sh/uv/).<br>
**Memory: on the CPU the process peaks at about 14.5 GB** (bf16 weights are kept as bf16 and widened to fp32 at run time; measured with `verify/bench.py` on a 32 GB machine). The whole model is about 21 GB on disk.<br>
Optional groups:
- `verify`: `tokenizers`, for the tokenizer cross-check (`check_tokenizer_fuzz`)
- `webcam`: OpenCV, for `demo_inference_webcam.py`
- `export`: torch, transformers etc. for exporting or running the reference (never used at run time)
- `test`: pytest for `tests/` (no model needed)
- extra `gpu`: `onnxruntime-gpu` (see "Running on a GPU"; it cannot coexist with `onnxruntime`)

# Installation

```bash
git clone https://github.com/Kazuhito00/Clef-Flash-ONNX-CPU
cd Clef-Flash-ONNX-CPU

# runtime dependencies only (onnxruntime, numpy, pillow)
uv sync
```

# Download Model
clef-flash has no published ONNX. The files assembled from ollaya's published weight-free graph and Cloudflare's official bf16 safetensors are to be placed on the [Releases](https://github.com/Kazuhito00/Clef-Flash-ONNX-CPU/releases) page (they are not in the repository; the Release is planned as v0.0.0 and has not been uploaded yet).<br>
The script below fetches them into `models/onnx/`. Standard-library HTTPS only; no git or git-lfs.
```bash
uv run download_model.py
```
- **With no arguments it fetches every file of the Release (graphs, vision tower, Cloudflare's 5 safetensors files, tokenizer and configs; about 21 GB)**
- The source is the latest Release of `Kazuhito00/Clef-Flash-ONNX-CPU` by default (`--repo OWNER/NAME` or the environment variable `CLEF_ONNX_REPO` choose another; priority: argument > environment > default)
- Every file is checked against the SHA-256 in the Release's `manifest.json` (a mismatch deletes the file and stops)
- Files split to stay under GitHub's 2 GiB limit (the 4 safetensors shards, 2 to 3 parts each) are joined after download and checked
- Files already in place are skipped, and an interrupted download resumes with the same command

```bash
uv run download_model.py --list                                               # list the files
uv run download_model.py --exclude "*vision_encoder*" --exclude "*-vl.onnx"   # text only (skips the vision tower, about 1.8 GB less)
uv run download_model.py --tag v0.0.0                                         # a given tag (default: latest)
```

### Files for the Release (v0.0.0)
`release/make_release.py` stages them (`--dry-run` only lists them). Each asset is at most 1.9 GiB (about 2040 MB); 28 assets in total (plus `manifest.json` and `SHA256SUMS`), about 20.9 GB.

| Asset | Size | Content |
|---|---:|---|
| `model-0000{1,2,3}-of-00004.safetensors.part000..002` | 2040 / 2040 / 863-908 MB each | Cloudflare's bf16 weights, byte for byte (the graph reads them by byte offset) |
| `model-00004-of-00004.safetensors.part000, 001` | 2040 / 1894 MB | same (4th shard) |
| `joint_head.safetensors` | 244 MB | Cloudflare's joint schema head |
| `vision_encoder.onnx.data` | 1825 MB | weights of the vision tower (one file under 2 GiB, not split) |
| `model-fp32.onnx`, `model-fp32-vl.onnx`, `vision_encoder.onnx` | 12 / 12 / 6 MB | weight-free graphs (text, with image inputs, vision tower) |
| `flash__decision.json`, `flash__calibration.json` | a few KB | sequence layout, calibration (temperatures); stored under `flash/` after download |
| `tokenizer.json`, other configs, `joint_schema_model.py`, `LICENSE` | 20 MB etc. | Cloudflare's files, unchanged |

The weights are Cloudflare's own files, pinned to revision `17f0b0ad...`. Hugging Face's `main` has moved on from that revision, so only this pinned revision matches the offsets in the graph. Without the Release you can fetch the same bytes from Hugging Face and assemble the model yourself (`export/prepare_model.py`, checked against the LFS sha256).
```bash
uv sync --group export
uv run export/prepare_model.py                   # 19 GB of weights + the graph; revision pinned, sha256 checked
```

# Usage

### Examples (Python)
```bash
uv run demo_inference_text.py
uv run demo_inference_image.py
```
The text demo answers a built-in example (a support ticket with a noul, a choice and a score question). To give your own text and questions:
```bash
uv run demo_inference_text.py --state "My card was charged twice." --choice "team=billing,technical" --noul "angry=The customer is angry"
```
The image demo uses `artifacts/images/sample_chart.png` by default (a bar chart; it asks which bar is the tallest and whether the picture is a bar chart). The sample was drawn by a script: it is not a photograph and contains no third-party material. Pass your own pictures (up to 4) as positional arguments:
```bash
uv run demo_inference_image.py photo.jpg --noul "dog=There is a dog in the picture"
```
`artifacts/images/cat.jpg` (Wikimedia Commons, CC0, by AdinaVoicu; see `artifacts/images/LICENSES.json`) is bundled too. Each picture is scaled down to at most 448 px on its longer side before it reaches the model (the original implementation accepts up to about 16.7 Mpixel, which is not workable on a CPU). The reference (golden data) is given the same scaled pictures.

```python
from clef_decide import Decider, choice, noul, score

d = Decider()                  # models/onnx (CLEF_MODEL_DIR points elsewhere)
d.decide("My card was charged twice.", {
    "angry": noul("The customer is angry"),
    "team": choice("Which team?", {"billing": "charges", "tech": "bugs"}),
    "urgency": score("How urgent?", ["low", "medium", "high", "critical"]),
})
```

For pictures, open the model with `Decider(load_vision=True)` and add them to the request (up to 4):
```python
d = Decider(load_vision=True)
d.decide("Look at the picture.", {"dog": noul("There is a dog")}, images=["photo.jpg"])
```
`images` takes file paths, bytes, Pillow images or base64. With `load_vision=True`, text-only requests also go through the image-capable graph (`model-fp32-vl.onnx`; with no image it gives the text graph's result, checked by `verify/check_vl_equivalence.py`). The two graphs read the same 18 GB of weights, so only one is opened at a time (opening both would double the memory).

**All questions are decided in one forward pass.** Nothing is generated: the backbone (Qwen3.5-9B) reads the state and every question, and the joint schema head scores all options of all questions together.

### Is there a person in the camera image?
Grabs frames with OpenCV, asks whether a person is in the newest frame and draws the answer on the live image.
```bash
uv run --group webcam demo_inference_webcam.py                        # camera 0
uv run --group webcam demo_inference_webcam.py --camera 1 --mirror
uv run --group webcam demo_inference_webcam.py --question "Is the person wearing glasses?"
uv run --group webcam demo_inference_webcam.py --camera artifacts/images/cat.jpg   # an image or video file instead of a camera
```
- One decision takes several seconds to tens of seconds on the CPU (9B), so it runs in its own thread and the live image never stalls. The banner (green = yes, red = no) and the probability of "yes" update whenever a decision is ready
- Change the yes/no question with `--question`. The picture given to the model is scaled to 224 px on its long side (`--size`)
- OpenCV is for this demo only (the `webcam` group); it is not a runtime dependency

### Running on a GPU
Every demo and `verify/bench.py` accept `--device` and `--require-gpu`.

| Option | Behavior |
|---|---|
| `--device cpu` (default) | CPU only |
| `--device cuda` | CUDAExecutionProvider, then CPUExecutionProvider. Without CUDA it warns and runs on the CPU |
| `--device auto` | CUDA if this onnxruntime build has it, else the CPU |
| `--require-gpu` | an error when the model does not end up on a GPU (prevents silently running on the CPU while thinking it is the GPU) |

The providers really used are printed on the last line of the demos (`providers used -> graph: ...; vision: ...`). onnxruntime can drop CUDA without an error when it cannot load it, so after every session is created `get_providers()` is checked, and a mismatch is a warning (an error with `--require-gpu`). From Python: `Decider(device="cuda", require_gpu=True)`, then `d.providers_used`.

```bash
# 1. Install the GPU build of onnxruntime (it writes into the same folder as onnxruntime, so the two cannot coexist: swap them)
uv sync --extra gpu
uv pip uninstall onnxruntime
uv pip install --force-reinstall --no-deps onnxruntime-gpu

# 2. Run (--no-sync is required; without it uv puts the CPU build back)
uv run --no-sync demo_inference_text.py --device cuda --require-gpu
```
- Requirements: an NVIDIA GPU, CUDA 13 and cuDNN 9. `onnxruntime-gpu` does not bundle the CUDA DLLs and looks them up on PATH, so put both `bin` folders on PATH (Windows example: `C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v13.4\bin\x64`, `C:\Program Files\NVIDIA\CUDNN\v9.20\bin\13.2\x64`; adjust to your versions). If they are missing, a warning about a missing DLL appears
- To go back to the CPU build: `uv pip uninstall onnxruntime-gpu`, then `uv sync --reinstall-package onnxruntime`
- **This model is large (about 20 GB of weights) and does not fit on common GPUs.** Measured (RTX 3050 Ti Laptop 4 GB, CUDA 13.4, cuDNN 9.20, onnxruntime-gpu 1.31): loading on CUDA failed with out of memory (`CUDA failure 2`) after 62.7 s, and the run fell back to the CPU with a warning (with `--require-gpu` it is an error at that point). It is not a PATH problem: the CUDA DLLs loaded and the memory allocation failed
- VRAM guideline: it does not fit in 4 GB. How much VRAM is enough, and whether it is then faster than the CPU, is **unverified** (no GPU speed measurement exists for this model)
- **On an A100 40 GB (Colab, onnxruntime-gpu 1.31.0, CUDA 13) the model loads, but with graph optimization on every request failed** (`MemcpyToHost ... CUDA failure 1: invalid argument`; not out of memory): the copy of the Memcpy node onnxruntime inserts for the nodes left on the CPU fails. With graph optimization disabled (`ORT_DISABLE_ALL`) it runs, and the output of one question matched the CPU exactly (max difference 0, same choice). Memory limits, the arena strategy, cuDNN, TF32 and the copy stream settings do not help. So when CUDA is requested, the main graph (only) is created with graph optimization disabled (CPU is unchanged). Other GPUs / other onnxruntime versions are untested
- For context: in the same measurements some smaller models were much faster on the GPU (about 14x for OneJev 0.8B, about 10x for JEVision, for example), but the effect differs a lot between models (some show no gain or are slower) and does not carry over to this 9B
- Probabilities on a GPU do not match the CPU exactly (different order of operations)

# Verification
```bash
uv run verify/run_all.py --skip-model                    # only the gates that do not load the 9B model
uv run verify/run_all.py                                 # everything (each gate that loads the 9B model needs about 14.5 GB of RAM and minutes)
uv run --group verify verify/run_all.py --full-fuzz      # run the fuzz at full size when tokenizers is present
uv run --group test pytest                               # unit tests (no model)
```

| Gate | What it checks | Result (run during development) |
|---|---|---|
| check_prompt | prompt strings, `input_ids` and the spans of every question and option equal the reference | 10/10 exact |
| verify | option logits and decisions equal Cloudflare's code (bf16 reference) | 0 decisions differ, max abs logit difference 0.037 |
| check_invariance | the padding length does not move the result | 0 |
| check_tokenizer_fuzz | pure-Python BPE vs transformers (Rust) | 0 mismatches in 5159 cases |
| check_vision_inputs | image pixels, bilinear taps and position ids equal transformers (PIL backend) | 18/18 |
| check_vl_equivalence | with no image the VL graph answers like the text graph (10 cases) | 0 decisions differ, max abs logit difference 0.037 |
| verify_image | vision tower + prompt ids + logits + decisions equal the reference (5 picture requests) | ids exact, 0 decisions differ, max abs logit difference 0.072 |

The reference runs in bf16 (fp32 would need about 36 GB); the graph computes in fp32 from the same bf16 weights, so logits are close, not equal. The gates therefore demand the same decision for every question and report near-ties separately.<br>
`artifacts/golden.json` and `golden_image.json` are golden data made by actually running Cloudflare's own code (`joint_schema_model.py`). A difference can therefore be pinned on the port (the image processor is pinned to the PIL backend; transformers 5.10.2 is assumed).

### Accuracy
Accuracy comes from the Wikipedia image bench (`wikipedia-vision-decision-bench`, "comparison of all models"), on its legacy 1875 questions with pictures (8 threads, 224 px, `REV=0`; see there for the other models).

| Commons en | Commons ja | synthetic en | synthetic ja | reasoning en | reasoning ja |
|---:|---:|---:|---:|---:|---:|
| 98.3 | 98.2 | 97.4 | 96.1 | 76.4 | 80.0 |

(In %. Commons 1268 questions, synthetic 382, reasoning 225. The reasoning set includes formats with few options where guessing already scores well.)

### Speed and memory
- Memory: about 14.5 GB peak on the CPU (above)
- Speed: **no timing is published for this model.** In the bench, the measurements of this model on that machine did not meet the calibration rule ("the machine's reference timing within about 3x of the quiet value"), so the times were not adopted. The first call reads about 18 GB of weights from disk and is much slower. The GPU is unmeasured because the model does not fit (see above)
- To measure on your machine: `uv run verify/bench.py` (prints peak memory, the median and the providers used)

# Limitations
- It uses a lot of memory (about 14.5 GB peak). Do not run it next to another heavy job
- Up to 4 pictures, each scaled to at most 448 px on its longer side. Video is not supported
- The runtime does not state an upper limit on the number of options (unconfirmed)
- The graph's sequence length is rounded up to a multiple of 64 (the linear-attention layers loop over 64-token chunks); padding does not change the result (`check_invariance`)
- `tokenizer.py` is specific to this checkpoint and rejects unexpected input with an exception
- The weights in the Release are those of revision `17f0b0ad...`. Weights of another revision cannot be used (the graph reads them by byte offset)
- GPU: as above, it does not fit in 4 GB of VRAM

# Project Structure

```text
README.md                # README (Japanese)
README_EN.md             # README (English)
LICENSE                  # Apache-2.0
pyproject.toml           # dependencies (runtime: onnxruntime, numpy, pillow; extra: gpu; groups: verify / test / webcam / export)
download_model.py        # fetch the model (GitHub Releases, SHA-256 check, joins split files)
demo_inference_text.py   # text inference demo
demo_inference_image.py  # image inference demo
demo_inference_webcam.py # is there a person in the camera image (OpenCV)
clef_decide/             # the inference code (light dependencies only; does not import export/)
  runtime.py             #   ORT session (Decider) + answer arithmetic
  device.py              #   --device / --require-gpu provider choice, and the check of what was really used
  tokenizer.py           #   pure-Python Qwen byte-level BPE
  protocol.py            #   prompt assembly, question validation, spans
  image.py               #   Pillow decode + smart_resize + patching + positions + vision tower
  _regex_translate.py    #   Rust regex to Python re translation
  _uniprops.py           #   generated: Unicode range tables
export/                  # heavy dependencies (torch / transformers / onnx); never used at run time
  prepare_model.py       #   fetch the weights (Cloudflare) and the graph (ollaya), rewrite the weight references
  export_vision.py       #   export the vision tower to ONNX
  add_image_inputs.py    #   add image inputs to the text graph (graph surgery only)
  make_golden.py         #   make artifacts/golden.json (with Cloudflare's code)
  make_golden_image.py   #   make artifacts/golden_image.json
  cases.py, cases_image.py, download.py, gen_uniprops.py, vision_export_patch.py
release/
  make_release.py        # stage the Release files from models/onnx (split, SHA-256, manifest.json; has --dry-run)
  notes.md               # the Release description
verify/                  # verification gates (run_all.py runs them all), bench.py, smoke.py
tests/                   # unit tests (no model)
artifacts/               # golden.json, golden_image.json (golden data), sample pictures
licenses/                # the upstream license text, unchanged
models/                  # ONNX, weights, tokenizer, configs (gitignored)
```

# Reference
- [Cloudflare/clef-flash](https://huggingface.co/Cloudflare/clef-flash) (the original model; Cloudflare; Apache-2.0)
- [Qwen/Qwen3.5-9B](https://huggingface.co/Qwen/Qwen3.5-9B) (the base of the original model; Alibaba Cloud; Apache-2.0)
- [ollaya-dev/clef](https://huggingface.co/ollaya-dev/clef) (the weight-free ONNX graph; ollaya; its model card metadata says Apache-2.0)

# License
Clef-Flash-ONNX-CPU is under [Apache-2.0 license](LICENSE).<br>
The model weights follow the license of the original model ([Cloudflare/clef-flash](https://huggingface.co/Cloudflare/clef-flash), Apache-2.0; the base Qwen3.5-9B is Apache-2.0 too). `licenses/LICENSE-UPSTREAM` is a verbatim copy of the `LICENSE` file in the Cloudflare/clef-flash repository (the full Apache-2.0 text with the notice "Copyright 2026 Alibaba Cloud", identical to Qwen's license text). The Release carries the same `LICENSE` unchanged.

## About the converted files
The ONNX files in the Release are made from [Cloudflare/clef-flash](https://huggingface.co/Cloudflare/clef-flash) (Cloudflare; Apache-2.0), which has no official ONNX, as follows (the changes are listed in the spirit of Apache-2.0 section 4(b)):
1. **Graph**: ollaya's published ONNX graph (`flash/model-fp32.onnx` of [ollaya-dev/clef](https://huggingface.co/ollaya-dev/clef), 12 MB, no weights). Its weights are references to byte ranges of Cloudflare's official bf16 safetensors, widened to fp32 by Cast nodes inside the graph. `export/prepare_model.py` rewrites the references from `sha256-<hash>` to the real file names (nothing else is changed)
2. **Weights**: Cloudflare's safetensors (revision `17f0b0ad...`) are used as they are, neither copied nor converted. The Release carries them split by bytes because of the 2 GiB limit; the parts are joined on download and checked by sha256
3. **Image input**: the vision tower (`vision_encoder.onnx`) was exported with `export/export_vision.py`. The text graph with extra image inputs (`image_embeds`, `image_positions`, `mrope_positions`; `model-fp32-vl.onnx`) was made by graph surgery only (`export/add_image_inputs.py`); with no image it gives the text graph's result
4. **Runtime**: the tokenizer, prompt assembly, image preprocessing and answer arithmetic are ported to pure Python and checked against the original implementation (see "Verification"). Note that pictures are scaled to at most 448 px on the long side (a deviation from the original; the golden data uses the same scaled pictures)

**License check result**: Cloudflare/clef-flash (model card metadata and the repository's `LICENSE`) and the base Qwen/Qwen3.5-9B are both Apache-2.0, and no restriction such as non-commercial use was found for redistribution. Neither repository has a `NOTICE` file. The original model card does not describe the training data, so this could not be checked. The ollaya-dev/clef repository has no `LICENSE` file: the basis for calling it Apache-2.0 is its model card metadata and text only (the Release contains a rewritten version of this graph). The providers of the original model, ollaya, and this converted work give no warranty on the converted files.
