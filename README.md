# BathysRDT-722M

> **準備中** — 重みの配布（Hugging Face）は準備中です。

BathysRDT-722M は、重みを共有した再帰ブロックを深さ方向に繰り返し適用する Transformer（recurrent-depth Transformer）の 722M パラメータ版です。日本語テキストで、大規模な教師モデルからの知識蒸留により事前学習しました。

このリポジトリは、**研究者がモデルを実行して評価できる最低限**（推論コード、推論設定、重みの識別情報）を提供します。学習コードと学習時の制御方法は含みません。

BathysRDT-722M is a 722M-parameter recurrent-depth Transformer (a weight-shared block applied repeatedly along depth), pre-trained on Japanese text by knowledge distillation. This repository provides the minimum needed to run and evaluate the model. Training code and training-time control methods are not included.

## Models

| Checkpoint | Training tokens | Parameters | dtype | File | Size (bytes) | SHA-256 |
|---|---|---|---|---|---|---|
| 700M | 700,006,400 | 721,631,233 | bfloat16 | `model.safetensors` | 1,443,273,890 | `b82b27ce3962c819a8f6d668e15fbc4237e880ec454a87b41b04cd3567ef6a66` |
| 1400M | 1,400,012,800 | 721,631,233 | bfloat16 | `model.safetensors` | 1,443,273,890 | `0e763df83d6173e6d0ceed05b61a572b3db8ad70873ab838b57a3a83cff57b3d` |

- 重みファイルには 116 テンソルが入っています。出力層（`head.weight`）は埋め込み（`embed.weight`）と共有（tie）しており、読み込み時に復元します。
- SHA-256 は、公開時点のモデルファイルとの同一性を確認するための識別子です（`SHA256SUMS` を参照）。
- 重みの配布先: Hugging Face（準備中）。

## Validation

公開した2つのチェックポイントを、同じ評価条件で測定しました。

- 評価データ: 検証用に分けた日本語テキスト 50 系列 × 512 トークン
- 条件: 同梱の推論設定（`configs/`）、seed 0・1・2 の平均。seed による差は 0.002 以下でした
- CE: 正解トークンに対する交差エントロピー（nats/token）。KL: 教師モデルの出力分布に対する KL ダイバージェンス

| Checkpoint | Depth | CE | KL |
|---|---|---|---|
| 700M | 11（公開推論設定） | 9.640 | 4.764 |
| 1400M | 11（公開推論設定） | 10.151 | 4.684 |

- 公開推論設定（depth=11）では、CE と KL で異なる傾向が観測されました（CE は 700M、KL は 1400M が低い値）。
- 深さ1では、1400M の CE が 700M より 0.147 低い値でした（標準誤差 0.011、50 系列中 49 系列で低い）。
- これらはチェックポイント間の特性の違いを示す測定結果であり、単一の指標による総合的な優劣を示すものではありません。
- 本モデルは研究用の成果物です。実用的な文章生成の品質は想定していません。

The two checkpoints were evaluated under identical conditions (50 held-out Japanese sequences × 512 tokens; bundled inference settings; mean over seeds 0–2). At the released setting (depth 11), CE is lower for 700M and KL is lower for 1400M. At depth 1, the CE of 1400M is 0.147 lower than that of 700M (SE 0.011; lower in 49 of 50 sequences). These are measurements of differences between checkpoints, not an overall ranking. This model is a research artifact.

## Usage

推論コードは `inference/`、推論設定は `configs/` にあります。重み（`model.safetensors`、`config.json`）は配布元から入手してください。

```bash
# 自己テスト（CPU、重み不要）
python3 inference/bathysrdt_722m_infer_v111_20261002_0525.py --self-test

# テキストの CE と参考用の greedy 生成
python3 inference/bathysrdt_722m_infer_v111_20261002_0525.py \
    --weights <dir>/1400M/model.safetensors \
    --config <dir>/1400M/config.json \
    --settings configs/infer_settings_722m_public_v110.json \
    --tag 1400M \
    --text "日本の首都は" \
    --generate 32
```

- 重みファイルの SHA-256 は、推論設定に書かれた値と照合され、一致しなければ停止します。
- 推論には bfloat16 と CUDA の autocast を使います（推論設定どおり）。評価値を再現するには、推論設定を変更しないでください。
- 必要なもの: Python 3、PyTorch、safetensors、numpy。テキスト入力には transformers（`trust_remote_code=True`）。

## Tokenizer

`llm-jp/llm-jp-4-32b-a3b-base` のトークナイザー（語彙 196,608、Apache-2.0）を使います。リビジョンと語彙のハッシュは推論設定に記載します。トークナイザーは同梱しません。

## Training data and teacher

- 教師モデル: [llm-jp/llm-jp-4-32b-a3b-base](https://huggingface.co/llm-jp/llm-jp-4-32b-a3b-base)（Apache-2.0）
- 学習データ: [CulturaX](https://huggingface.co/datasets/uonlp/CulturaX)（日本語）。CulturaX の利用条件は mC4 と OSCAR の条件に従います。mC4 は ODC-BY で、Common Crawl の利用規約にも従います。OSCAR は、テキストそのものの権利を OSCAR 側が持たない旨を明記しています。
- CulturaX の利用条件を、このモデルの重みのライセンスとして読み替えることはしません。

## Not included

学習コード、学習時の制御器と制御状態、optimizer、学習 checkpoint 全体は含みません。

## License

コード・文書とモデルの重みは、**BathysRDT Research License Version 1.0**（[LICENSE](LICENSE)、日本語が正文。英語の参考訳は [LICENSE_EN.md](LICENSE_EN.md)）で提供します。

- 非営利の研究・教育目的に限り利用できます（所属ではなく、利用行為の目的で判断します）。
- 原重みの再配布はできません。重みは指定の配布元から入手してください。追加学習した派生重みは、同じライセンスで公開できます。
- 特許その他の産業財産権についてのライセンスは許諾しません。
- 商用利用には別途の書面契約が必要です。

Code, documents and model weights are provided under the **BathysRDT Research License Version 1.0** for non-commercial research and educational use only. No patent license is granted. Commercial use requires a separate written agreement.

## Related publication

中村忠行「精度差分観測による重み共有再帰Transformerの学習ダイナミクス解明: Gate Decay振幅制御、FP32崩壊境界、および多制御器協調学習」Jxiv（JST プレプリントサーバ）, 2026. DOI: [10.51094/jxiv.6476](https://doi.org/10.51094/jxiv.6476)

## Patents

本モデルに関係する技術について、日本で特許出願をしています（特願2026-201413 ほか）。
本リポジトリおよび BathysRDT Research License は、これらの特許出願・特許についての
いかなる権利も許諾するものではありません（LICENSE 第5条）。

Patent applications related to this model have been filed in Japan (including JP 2026-201413).
Neither this repository nor the BathysRDT Research License grants any rights under these
applications or any resulting patents (see LICENSE, Section 5).

## Contact

- 技術的な質問: GitHub Issues
- ライセンス・商用利用の問い合わせ: 中村忠行 <tada2503@yahoo.co.jp>
