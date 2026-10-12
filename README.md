[[Japanese](README.md)/[English](README_EN.md)]

# Clef-Flash-ONNX-CPU

[`Cloudflare/clef-flash`](https://huggingface.co/Cloudflare/clef-flash)（9B の構造化判定モデル。テキストと画像）を、onnxruntime と numpy だけで推論する最小依存の推論実装です。 ※GPU への切り替えも可（下記「GPUで動かす」。ただしこの 9B は一般的な GPU には載りません）<br>
公式の `joint_schema_model.py` が要求する torch・transformers は使いません。

# Features
以下の特徴があります。
- 最小依存：実行時に必要なパッケージは onnxruntime と numpy の2つだけ（画像を使う場合は pillow を足した3つ）
- 動かすのに必要なスクリプトは `download_model.py`（モデル取得）と `demo_inference_text.py`（推論）の2つ。画像は `demo_inference_image.py`、カメラは `demo_inference_webcam.py`
- 判定は参照実装（Cloudflare 自身のコード、bf16、CPU）と一致（テキスト・画像とも判定の相違 0 件、トークン列は完全一致）
- 全質問が1回の順伝播で決まる（生成は走らない）。`noul`（はい／いいえ）、`choice`（選択）、`score`（段階）の3種類を1つのリクエストに混ぜられる
- 画像対応（1回に最大4枚。各画像は長辺 448 px 以下に縮小）
- 前処理（Qwen byte-level BPE、プロンプトの組み立て、3次元 M-RoPE、画像のパッチ化と位置情報）を純Pythonに移植し、参照実装との突き合わせ検証を同梱
- CPU のほか、`--device cuda` で GPU に切り替えられる（`--require-gpu` を付ければ、CPU への黙ったフォールバックはエラーになる）

# Purpose of This Repository
以下の検証を目的としています。
- clef-flash の判定を、実行時依存2〜3つだけでどこまで動かせるか
- 参照実装の前処理を純Pythonに移植して、同じ入力が作れるか
- 9B のモデルを CPU でどこまで扱えるか（メモリ）

# Requirements
```
Python 3.10 or later

numpy        1.26 or later
onnxruntime  1.20 or later
pillow       10.0 or later   （画像を使う場合のみ）
```
実行に必要なのはこれだけです。モデルの取得も標準ライブラリのみで動きます。<br>
環境構築は[uv](https://docs.astral.sh/uv/)を使います。<br>
**メモリ：CPU で動かすとき、プロセスのピークは約 14.5 GB です**（bf16 の重みを bf16 のまま持ち、実行時に fp32 へ展開するため。`verify/bench.py` での値で、32 GB の機械での測定）。ディスクは、モデル全体で約 21 GB です。<br>
オプションのグループがあります。
- `verify`：トークナイザの突き合わせ検証（`check_tokenizer_fuzz`）用の `tokenizers`
- `webcam`：`demo_inference_webcam.py` 用の OpenCV
- `export`：ONNX を自分で書き出す／検証用の参照実装を動かすときの torch・transformers など（実行時には使いません）
- `test`：`tests/` 用の pytest（モデルは要りません）
- extra `gpu`：`onnxruntime-gpu`（下記「GPUで動かす」。`onnxruntime` とは同居できません）

# Installation

```bash
git clone https://github.com/Kazuhito00/Clef-Flash-ONNX-CPU
cd Clef-Flash-ONNX-CPU

# 実行時の依存（onnxruntime・numpy・pillow）のみインストール
uv sync
```

# Download Model
clef-flash には公開されている ONNX がありません。ollaya が公開している「重みを含まないグラフ」と Cloudflare の公式 safetensors（bf16）から組み立てたものを、[Releases](https://github.com/Kazuhito00/Clef-Flash-ONNX-CPU/releases)に置く予定です（リポジトリには含めていません。Release は v0.0.0 を想定しており、まだアップロードしていません）。<br>
以下のスクリプトで `models/onnx/` に取得します。標準ライブラリのHTTPSのみで動き、gitもgit-lfsも不要です。
```bash
uv run download_model.py
```
- **引数なしでは、Release の全ファイル（グラフ、視覚エンコーダ、Cloudflare の safetensors 5 本、トークナイザと設定。約 21 GB）を取得します**
- 取得元は、既定で `Kazuhito00/Clef-Flash-ONNX-CPU` の最新の Release です（`--repo OWNER/NAME` か、環境変数 `CLEF_ONNX_REPO` で別のリポジトリにできます。優先順位は 引数 ＞ 環境変数 ＞ 既定）
- 取得したファイルは、Release の `manifest.json` に書かれた SHA-256 と照合します（不一致ならファイルを削除して停止）
- GitHub の制限（1ファイル 2 GiB 未満）で分割されたファイル（safetensors の 4 本。各 2〜3 パート）は、取得後に結合して照合します
- 取得済みなら何もせず、中断した場合は同じコマンドで続きから再開します

```bash
uv run download_model.py --list                                               # ファイル一覧
uv run download_model.py --exclude "*vision_encoder*" --exclude "*-vl.onnx"   # テキスト専用（視覚エンコーダを取らない。約 1.8 GB 少ない）
uv run download_model.py --tag v0.0.0                                         # タグを指定（既定は latest）
```

### Release に置くファイル（v0.0.0）
`release/make_release.py` が作ります（`--dry-run` で一覧だけ確認できます）。1 アセット 1.9 GiB（約 2040 MB）以下に分割し、合計 28 アセット（＋ `manifest.json`、`SHA256SUMS`）、約 20.9 GB です。

| アセット | サイズ | 内容 |
|---|---:|---|
| `model-0000{1,2,3}-of-00004.safetensors.part000〜002` | 各 2040 / 2040 / 863〜908 MB | Cloudflare の bf16 重み（元のバイト列のまま。グラフがバイトオフセットで参照する） |
| `model-00004-of-00004.safetensors.part000, 001` | 2040 / 1894 MB | 同上（第 4 シャード） |
| `joint_head.safetensors` | 244 MB | Cloudflare の joint schema head |
| `vision_encoder.onnx.data` | 1825 MB | 視覚エンコーダの重み（1 ファイルで 2 GiB 未満なので分割しません） |
| `model-fp32.onnx`、`model-fp32-vl.onnx`、`vision_encoder.onnx` | 12 / 12 / 6 MB | 重みを持たないグラフ（テキスト用、画像入力つき、視覚エンコーダ） |
| `flash__decision.json`、`flash__calibration.json` | 数 KB | 系列の配置、較正（温度）。取得すると `flash/` 以下に置かれます |
| `tokenizer.json` ほか設定ファイル、`joint_schema_model.py`、`LICENSE` | 20 MB ほか | Cloudflare のファイル（元のまま） |

重みは Cloudflare の公式ファイルそのものです（リビジョン `17f0b0ad…` に固定）。Hugging Face の最新版（`main`）はこのリビジョンから進んでいるため、グラフの参照先と合うのはこの固定リビジョンだけです。Release を使わず、Hugging Face から同じバイト列を取って自分で組み立てることもできます（`export/prepare_model.py`。LFS の sha256 と照合します）。
```bash
uv sync --group export
uv run export/prepare_model.py                   # 19 GB の重み + グラフ。リビジョン固定、sha256 照合
```

# Usage

### 実行例(Python)
```bash
uv run demo_inference_text.py
uv run demo_inference_image.py
```
テキストのデモは、組み込みの例（チケットの文面に対し noul・choice・score の 3 問）に答えます。自分の文面と質問は次のように渡します。
```bash
uv run demo_inference_text.py --state "My card was charged twice." --choice "team=billing,technical" --noul "angry=The customer is angry"
```
画像のデモは、既定で `artifacts/images/sample_chart.png`（棒グラフ。「どの棒が最も高いか」「棒グラフか」を聞きます）を使います。このサンプルはスクリプトで描いた画像で、写真ではなく、第三者の素材も含みません。自分の画像（最大 4 枚）は位置引数で渡します。
```bash
uv run demo_inference_image.py photo.jpg --noul "dog=There is a dog in the picture"
```
`artifacts/images/cat.jpg`（Wikimedia Commons、CC0、AdinaVoicu 撮影。出典は `artifacts/images/LICENSES.json`）も付属しています。各画像は、モデルに渡す前に長辺 448 px 以下に縮小します（元の実装は最大約 1670 万画素まで受け付けますが、CPU では現実的でないため）。参照実装（正解データ）にも同じ縮小後の画像を渡して突き合わせています。

```python
from clef_decide import Decider, choice, noul, score

d = Decider()                  # models/onnx（環境変数 CLEF_MODEL_DIR で別の場所も指せます）
d.decide("My card was charged twice.", {
    "angry": noul("The customer is angry"),
    "team": choice("Which team?", {"billing": "charges", "tech": "bugs"}),
    "urgency": score("How urgent?", ["low", "medium", "high", "critical"]),
})
```

画像は `Decider(load_vision=True)` で開き、リクエストに混ぜます（最大 4 枚）。
```python
d = Decider(load_vision=True)
d.decide("Look at the picture.", {"dog": noul("There is a dog")}, images=["photo.jpg"])
```
`images` には、ファイルのパス、バイト列、Pillow の画像、base64 を渡せます。`load_vision=True` のときは、テキストだけの質問も画像対応のグラフ（`model-fp32-vl.onnx`）を通ります（画像なしの結果はテキスト用グラフと同じであることを `verify/check_vl_equivalence.py` で確認しています）。2 つのグラフは同じ 18 GB の重みを読むので、同時には開きません（開くとメモリが倍になります）。

**全質問が1回の順伝播で決まります。** 生成は走りません。バックボーン（Qwen3.5-9B）が状態と全質問を読み、joint schema head がすべての質問のすべての選択肢を一緒に採点します。

### カメラで「人が映っているか」を判定する
OpenCV でカメラの映像を取得し、最新のフレームについて「人が映っているか」を判定して、結果を映像に重ねて表示します。
```bash
uv run --group webcam demo_inference_webcam.py                        # カメラ 0
uv run --group webcam demo_inference_webcam.py --camera 1 --mirror
uv run --group webcam demo_inference_webcam.py --question "Is the person wearing glasses?"
uv run --group webcam demo_inference_webcam.py --camera artifacts/images/cat.jpg   # カメラの代わりに画像・動画ファイル
```
- 1回の判定は CPU で数秒から数十秒かかる（9B）ため、判定は別スレッドで動かし、映像の表示は止めません。判定が出るたびに、上部の帯（緑＝はい／赤＝いいえ）と「はい」の確率が更新されます
- 質問は `--question`（はい・いいえの質問）で変えられます。モデルに渡す画像は長辺 224 px に縮小します（`--size`）
- OpenCV はこのデモ専用です（`webcam` グループ）。実行時の依存には入りません

### GPUで動かす
どのデモ・`verify/bench.py` も `--device` と `--require-gpu` を受け付けます。

| 指定 | 動作 |
|---|---|
| `--device cpu`（既定） | CPU のみ |
| `--device cuda` | CUDAExecutionProvider → CPUExecutionProvider の順。CUDA が使えなければ警告を出して CPU で動く |
| `--device auto` | この onnxruntime に CUDA があれば CUDA、なければ CPU |
| `--require-gpu` | GPU で動かないときエラーにする（CPU に静かに切り替わって GPU のつもりになる事故を防ぐ） |

実際に使われたプロバイダは、デモの最終行（`providers used -> graph: ...; vision: ...`）で確認できます。onnxruntime は、CUDA を読み込めないときにエラーなしで CPU に切り替えることがあるため、セッションを作るたびに `get_providers()` を調べ、食い違いは警告（`--require-gpu` ならエラー）にします。Python からは `Decider(device="cuda", require_gpu=True)` で指定し、`d.providers_used` で確認します。

```bash
# 1. GPU 版 onnxruntime を入れる（onnxruntime とは同じフォルダに書き込むため、同居できません。入れ替えます）
uv sync --extra gpu
uv pip uninstall onnxruntime
uv pip install --force-reinstall --no-deps onnxruntime-gpu

# 2. 実行する（--no-sync が必須。付けないと uv が CPU 版に戻します）
uv run --no-sync demo_inference_text.py --device cuda --require-gpu
```
- 要件：NVIDIA GPU、CUDA 13 と cuDNN 9。`onnxruntime-gpu` は CUDA の DLL を同梱せず PATH から探すので、両方の `bin` フォルダを PATH に入れてください（Windows の例：`C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v13.4\bin\x64`、`C:\Program Files\NVIDIA\CUDNN\v9.20\bin\13.2\x64`。バージョンは環境に合わせます）。見つからないと DLL が無いという警告が出ます
- CPU 版に戻すには `uv pip uninstall onnxruntime-gpu` のあと `uv sync --reinstall-package onnxruntime`
- **このモデルは大きく（重みは約 20 GB）、一般的な GPU には載りません。** 実測（RTX 3050 Ti Laptop 4 GB、CUDA 13.4、cuDNN 9.20、onnxruntime-gpu 1.31）では、CUDA でのロードが 62.7 秒後に out of memory（`CUDA failure 2`）で失敗し、警告を出して CPU で動きました（`--require-gpu` ならこの時点でエラーです）。PATH の問題ではなく、CUDA の DLL は読めていてメモリ確保で失敗しています
- VRAM の目安：4 GB では載りません。何 GB あれば載るか、載った場合に CPU より速いかは**未検証**です（このモデルでは GPU の速度測定ができていません）
- **A100 40 GB（Colab、onnxruntime-gpu 1.31.0、CUDA 13）では、ロードはできるが、グラフの最適化が有効だと全リクエストが失敗しました**（`MemcpyToHost ... CUDA failure 1: invalid argument`。メモリ不足ではありません）。CPU に残るノードのために onnxruntime が入れる Memcpy ノードのコピーが失敗します。グラフの最適化を無効（`ORT_DISABLE_ALL`）にすると動き、1 問の出力は CPU と完全に一致しました（最大差 0、選択も同じ）。メモリの上限・arena・cuDNN・TF32・コピー用ストリームなどの設定では直りません。そこで、CUDA を指定したときはメイン部のグラフだけ最適化を無効にしています（CPU は変更なし）。他の GPU・他の onnxruntime での動作は未検証です
- 参考：同じ測定で、小さい別のモデルでは GPU で大きく速くなったものがあります（OneJev 0.8B で約 14 倍、JEVision で約 10 倍など）が、モデルによって効果は大きく異なり（効果なしや遅くなるものもあります）、この 9B には当てはまりません
- GPU での確率は、CPU と完全には一致しません（演算の順序の違い）

# Verification
```bash
uv run verify/run_all.py --skip-model                    # 9B を読み込まないゲートだけ
uv run verify/run_all.py                                 # すべて（9B を読み込むゲートは、各々 約 14.5 GB のメモリと数分）
uv run --group verify verify/run_all.py --full-fuzz      # tokenizers があればファズも本気で回す
uv run --group test pytest                               # 単体テスト（モデル不要）
```

| ゲート | 内容 | 結果（開発時に実行したもの） |
|---|---|---|
| check_prompt | プロンプト文字列・`input_ids`・各質問と選択肢の範囲が参照実装と一致 | 10/10 完全一致 |
| verify | 選択肢ごとの logit と判定が Cloudflare のコード（bf16 の参照）と一致 | 判定の相違 0 件、logit の差は最大 0.037 |
| check_invariance | パディング長が結果を動かさない | 差 0 |
| check_tokenizer_fuzz | 純Python の BPE と transformers（Rust）の突合 | 5159 件で不一致 0 |
| check_vision_inputs | 画像のピクセル、双線形補間の参照点、位置 ID が transformers（PIL バックエンド）と一致 | 18/18 |
| check_vl_equivalence | 画像なしで VL グラフがテキスト用グラフと同じ答えを出す（10 件） | 判定の相違 0 件、logit の差は最大 0.037 |
| verify_image | 視覚エンコーダ＋プロンプトの ID＋logit＋判定が参照実装と一致（画像 5 リクエスト） | ID 完全一致、判定の相違 0 件、logit の差は最大 0.072 |

参照実装は bf16 で動かしています（fp32 では約 36 GB 要るため）。グラフは同じ bf16 の重みから fp32 で計算するので、logit は「近い」が「同じ」ではありません。このため、ゲートは「すべての質問で同じ判定になること」を求め、僅差のものは別に報告します。<br>
`artifacts/golden.json` と `golden_image.json` は、Cloudflare 自身のコード（`joint_schema_model.py`）を実際に動かして作った正解データです。差が出たら、移植のバグと断定できます（画像の前処理は PIL バックエンドに固定。transformers 5.10.2 を前提にしています）。

### 精度（正答率）
このモデルの正答率は、Wikipedia 画像ベンチ（`wikipedia-vision-decision-bench`。「全モデルの比較表」）の legacy の画像つき 1875 問での実測です（8 スレッド、224 px、`REV=0`。ほかのモデルとの比較はそちらを見てください）。

| Commons 英 | Commons 日 | synthetic 英 | synthetic 日 | reasoning 英 | reasoning 日 |
|---:|---:|---:|---:|---:|---:|
| 98.3 | 98.2 | 97.4 | 96.1 | 76.4 | 80.0 |

（単位は %。Commons 1268 問、synthetic 382 問、reasoning 225 問。reasoning は選択肢が少なく当てずっぽうでも当たりやすい形式を含みます。）

### 速度とメモリ
- メモリ：CPU でピーク約 14.5 GB（上記）
- 速度：**このモデルの時間は載せません。** ベンチでは、この機械での測定が「機械の基準測定が静かな時の約 3 倍以内」という較正を満たさなかったため、時間を採用していません。初回は約 18 GB の重みをディスクから読むので、特に遅くなります。GPU は上記のとおり、載らないため未測定です
- 手元で測るには `uv run verify/bench.py`（ピークメモリと中央値、使われたプロバイダを表示します）

# Limitations
- メモリを大量に使います（ピーク約 14.5 GB）。ほかの重い処理と同時に動かさないでください
- 画像は最大 4 枚。1 枚ずつ長辺 448 px 以下に縮小します。動画は未対応です
- 選択肢の数の上限は、このランタイムでは明示していません（未確認）
- グラフの系列長は 64 の倍数に切り上げます（linear attention の層が 64 トークンずつ処理するため）。パディングは結果を変えません（`check_invariance`）
- `tokenizer.py` はこのチェックポイント専用で、想定外の入力は例外で弾きます
- Release に置く重みはリビジョン `17f0b0ad…` のものです。別のリビジョンの重みとは使えません（グラフがバイトオフセットで参照するため）
- GPU：上記のとおり、4 GB の VRAM では載りません

# Project Structure

```text
README.md                # README（日本語）
README_EN.md             # README（英語）
LICENSE                  # Apache-2.0
pyproject.toml           # 依存定義（実行時は onnxruntime・numpy・pillow。extra: gpu、グループ: verify / test / webcam / export）
download_model.py        # モデル取得（GitHub Releases、SHA-256照合、分割ファイルの結合）
demo_inference_text.py   # テキストの推論デモ
demo_inference_image.py  # 画像の推論デモ
demo_inference_webcam.py # カメラで「人が映っているか」を判定（OpenCV）
clef_decide/             # 推論実装（軽い依存のみ。export/ を import しない）
  runtime.py             #   ORT セッション（Decider）+ 回答の計算
  device.py              #   --device / --require-gpu のプロバイダ選択と、実際に使われたプロバイダの確認
  tokenizer.py           #   Qwen byte-level BPE の純Python実装
  protocol.py            #   プロンプトの組み立て、質問の検証、スパン
  image.py               #   Pillow デコード + smart_resize + パッチ化 + 位置情報 + 視覚エンコーダ
  _regex_translate.py    #   Rust の正規表現から Python の re への変換
  _uniprops.py           #   生成物: Unicode範囲表
export/                  # 重い依存（torch / transformers / onnx）。実行時には使わない
  prepare_model.py       #   重み（Cloudflare）とグラフ（ollaya）を取得し、重みの参照名を書き換える
  export_vision.py       #   視覚エンコーダを ONNX に書き出す
  add_image_inputs.py    #   テキスト用グラフに画像用の入力を足す（グラフの書き換えのみ）
  make_golden.py         #   artifacts/golden.json を作る（Cloudflare のコードで）
  make_golden_image.py   #   artifacts/golden_image.json を作る
  cases.py, cases_image.py, download.py, gen_uniprops.py, vision_export_patch.py
release/
  make_release.py        # models/onnx から Release 用のファイルを作る（分割・SHA-256・manifest.json。--dry-run あり）
  notes.md               # Release の説明文
verify/                  # 検証ゲート（run_all.pyで一括）と bench.py、smoke.py
tests/                   # 単体テスト（モデル不要）
artifacts/               # golden.json、golden_image.json（検証の正解データ）、サンプル画像
licenses/                # 元のモデルのライセンス文（そのまま）
models/                  # ONNX・重み・トークナイザ・設定（gitignore）
```

# Reference
- [Cloudflare/clef-flash](https://huggingface.co/Cloudflare/clef-flash)（元のモデル。Cloudflare。Apache-2.0）
- [Qwen/Qwen3.5-9B](https://huggingface.co/Qwen/Qwen3.5-9B)（元のモデルのベース。Alibaba Cloud。Apache-2.0）
- [ollaya-dev/clef](https://huggingface.co/ollaya-dev/clef)（重みを含まない ONNX グラフ。ollaya。モデルカードのメタデータは Apache-2.0）

# License
Clef-Flash-ONNX-CPU is under [Apache-2.0 license](LICENSE).<br>
モデルの重みは、元のモデル（[Cloudflare/clef-flash](https://huggingface.co/Cloudflare/clef-flash)、Apache-2.0。ベースの Qwen3.5-9B も Apache-2.0）のライセンスに従います。`licenses/LICENSE-UPSTREAM` は、Cloudflare/clef-flash のリポジトリにある `LICENSE` を元のまま写したものです（Apache-2.0 の全文で、著作権表示は「Copyright 2026 Alibaba Cloud」。Qwen のライセンス文と同じです）。Release にも、同じ `LICENSE` を元のまま同梱します。

## 変換物であることについて
Release に含まれる ONNX ファイルは、公式には ONNX が公開されていない [Cloudflare/clef-flash](https://huggingface.co/Cloudflare/clef-flash)（Cloudflare。Apache-2.0）から、次の手順で作ったものです（Apache-2.0 第 4 条(b) に従い、変更点を書きます）。
1. **グラフ**：ollaya が公開している ONNX グラフ（[ollaya-dev/clef](https://huggingface.co/ollaya-dev/clef) の `flash/model-fp32.onnx`、12 MB、重みなし）を使います。このグラフの重みは、Cloudflare の公式 safetensors（bf16）のバイト範囲への参照で、fp32 への展開はグラフ内の Cast で行います。`export/prepare_model.py` が、参照名を `sha256-<ハッシュ>` から実際のファイル名に書き換えています（それ以外は変更していません）
2. **重み**：Cloudflare の safetensors（リビジョン `17f0b0ad…`）を、コピーも変換もせず、そのままのバイト列で使います。Release には、2 GiB の制限のためバイト列で分割して置き、取得時に結合して sha256 を照合します
3. **画像入力**：視覚エンコーダ（`vision_encoder.onnx`）を `export/export_vision.py` で書き出しました。テキスト用グラフに画像用の入力（`image_embeds`、`image_positions`、`mrope_positions`）を足したもの（`model-fp32-vl.onnx`）は、グラフの書き換えだけで作っています（`export/add_image_inputs.py`）。画像なしの結果は、テキスト用グラフと同じです
4. **ランタイム**：トークナイザ、プロンプトの組み立て、画像の前処理、回答の計算を純Pythonで移植しました。元の実装との判定の一致を確認しています（上の「Verification」）。なお、画像は長辺 448 px 以下に縮小します（元の実装との差。正解データにも同じ縮小後の画像を使っています）

**ライセンス確認の結果**：Cloudflare/clef-flash（モデルカードのメタデータとリポジトリの `LICENSE`）、ベースの Qwen/Qwen3.5-9B はいずれも Apache-2.0 で、再配布に非商用などの制限は見つかりませんでした。両者のリポジトリに `NOTICE` ファイルはありません。学習データについて、元のモデルカードは記載しておらず、確認できていません。ollaya-dev/clef のリポジトリには `LICENSE` ファイルがなく、Apache-2.0 とする根拠はモデルカードのメタデータと本文の記述のみです（Release にはこのグラフを書き換えたものを含みます）。元のモデルの提供元、ollaya、およびこの変換物の動作を、提供元は保証していません。
