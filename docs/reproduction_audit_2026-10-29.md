# 論文全体の再現性監査（2026-10-29）

これは修正前の監査記録です。現在の実装と未確認範囲は [coverage.md](coverage.md) を参照してください。監査時のFigure6/7の集合の記述は、その後の照合で訂正しました。Figure6は従来verbalizationと同じ100件、Figure7とは18件だけ共通です。

## 判定

**現状のREADME通りに実行しても、論文全体は再現できない。verbalize実験も未実装。**

確認済みなのは、HF形式のローカルデータ変換と、一部モデルの基本的な視覚指示編集・評価・FSR集計の模擬ワークフローである。前回の13件のテスト成功は、論文全実験の網羅性やGPU実行を証明していない。

今回はコードを変更せず、実在するファイル、CLI、設定、論文の実験項目との対応を監査した。READMEと検証文書には、この未完了状態への参照を追加した。

## 照合対象

- ユーザー提示の現行本文（主結果、Figure 4–9、Appendix Fのverbalization）。
- `diagram/paper/pdfs/Camera-ready_20261017.pdf`（本文と既存Appendixの実験項目）。
- `diagram/paper/figures5_9_large_text_20261028/README.md`（最新の図と元データ）。
- `diagram/paper/figure9_results_20261029/README.md`、`diagram/paper/figure40_no_qwen3vl_20261024/provenance.json`（現行Figure 9の16設定・4条件）。
- `VisEditBench/` のREADME、全設定、CLI、生成・評価・集計実装、テスト。

最新の論文TeX全体はローカルで見つかっていない。旧PDFではFigure 9がADDの例なので、現行Figure 9はユーザー指定と更新済み図の出典を基準に照合した。Appendixの図番号の対応は旧PDFから推定しない。

## 実行を阻害する問題

| 項目 | 確認結果 | 根拠 |
| --- | --- | --- |
| 編集用vLLM def | 欠落 | `editing/vllm/vllm.def` が存在しない。`editing/vllm/apptainer/build.sh` はこのファイルを参照する |
| 評価用vLLM def | ファイルはあるがビルド入力のパスが不正 | `evaluation_image-level/vllm/vllm.def` の `%files` が `./evaluation_image-based/vllm/entrypoint.sh` を参照。この経路は公開リポジトリに存在しない |
| verbalize入口 | なし | `python edit.py verbalize --help` が引数エラーで終了（exit 2） |
| verbalize生成・抽出 | なし | 公開用のPython/shell/TOMLに `verbalize_first`、`<instruction>`、`gold_text_visual` の処理がない |

defの存在を先に確認せず「2か所にある」と回答したのは誤り。存在するのは評価用の1ファイルだけであり、そのファイルも現在の公開用ディレクトリに対応していない。

## 論文項目ごとの対応

「基本部あり」は、論文に掲載されたモデル・標本・設定・数値の再現を完了した意味ではない。

| 論文項目 | 公開用コードの状態 | 不足 |
| --- | --- | --- |
| データ取得・展開 | ローカル694件の変換・検証済み。HF呼出しはmockで確認 | ライブHFダウンロードと公開revisionの内容照合は未実行 |
| データ構築・人手検証の統計 | 未移植 | 元の2,819件の検証記録、2,148件の選別、694件への抽出と統計を作る処理 |
| Figure 3：データ分布 | type/countは保持 | 掲載ドーナツ図の作成処理 |
| Table 1：人手評価との相関 | judge要求生成の基本部あり | 人手評点、300出力との対応、評価者除外規則、Spearman/Kendall、Krippendorff alphaの集計 |
| Table 2：主結果 | 視覚指示編集・画像採点・FSRの基本部あり | 論文全32モデルの実行設定（TikZ編集29＋直接画像編集3）、モデル別の採用設定、コード指標・CFRを含む表の集計 |
| Figure 4：編集タイプ別FSR | `by_edit_type` の集計あり | 掲載図は1操作の236件に限定。現行集計は全件であり、その対象選択・モデル一覧・作図処理がない |
| Figure 5：操作数別FSR | `by_edit_operation` の集計あり | 掲載全モデルの実行と系列順・凡例・作図処理 |
| Figure 6：入力モダリティ | 4種類のvisual入力promptは存在 | 固定100件の抽出、各モデル×条件の設定と一括実行、図の分母規則・作図 |
| Figure 7：text/visual比較 | gold textを渡すpromptと補助JSONL結合は存在 | goldテキスト、固定100件のID、全13モデルの採用設定、比較・作図 |
| Figure 8：コード編集と直接画像編集 | Qwen Diffusersクライアントあり | Gemini 3 Pro Image / GPT Image 1.5の直接画像編集実装・設定、6条件の比較・作図 |
| Figure 9 / Appendix F：4条件 | 未移植 | 下記verbalizeの不足一式。現行Figure 9は16設定×4条件×100件 |
| コード評価と人手評価の相関 | コードjudge・評点抽出の基本部あり | 人手データとのID対応、相関・表の作成 |
| 主結果の標準偏差 | 未移植 | 各モデルの画像/コード指標の標準偏差を作る処理 |
| 代替judgeでの全694件評価 | 個別judgeを設定する基盤あり | 全29モデルの設定、採点済み/未採点の管理、対応表の作成 |
| judge間のモデル順位相関 | 未移植 | モデルID対応、同順位・丸めの扱い、Spearman/Kendallとp値の集計 |
| 自動評価の反復安定性 | 未移植 | 固定サンプル、5反復の設定と集計 |
| 直接画像編集のタイプ/操作数別分析 | 未移植 | 全3モデルの出力、条件別集計・表・図 |
| LAYOUT/STRUCTURE等の定性例 | 未移植 | 事例ID・元/指示/出力画像の選択、コード差分、今回選んだ誤接続例を出す処理 |
| 人手評価UI・一致度 | 未移植 | UI、評価者別評点・除外後の対応表・一致度集計 |

Figure 1・2等の説明図は実験実行の結果ではないが、掲載図の元ソース・再出力手順もこのリポジトリには入っていない。

## verbalizeで不足している内容

論文で採用した条件は次の4つ。

1. `direct`：visual指示から編集。
2. `verbalize_first`：**1回の応答内**で `<instruction>...</instruction>` を出してからTikZを出力。編集時にもvisual指示を参照する。
3. `gold_text`：正解テキスト指示から編集。
4. `gold_text_visual`：正解テキスト指示とvisual指示を併用して編集。

現在の公開コードには1と3の基本promptしかなく、2・4のprompt、モデル別要求、言語化文とTikZを分離する抽出、4条件の集計、Figure 9の出力がない。補助JSONLでgold textを渡せるだけではverbalize実験にはならない。

現行Figure 9は新実験12設定と従来実験4設定を組み合わせている。古い10設定の実験フォルダだけを移植しても16設定版とは一致しない。judgeはGemini 3.6 Flashであり、公開例のGemini 3.1 Proと異なる。採用対象100件・seed・トークン設定・失敗時処理を設定ごとに固定する必要がある。

## データと設定の照合結果

| 対象 | 確認した件数・状態 |
| --- | --- |
| ローカルHF形式 | 694件、gold `text_instruction` は0件 |
| Figure 6/7のローカル対象データ | 100件、gold textは100件、HFの694 IDに全件含まれる |
| 従来verbalizationのローカル対象データ | 別の100件、gold textは100件、HFの694 IDに全件含まれる |
| 上の2つの100件集合の共通部分 | 18件。先頭100件や任意抽出に置き換えられない |
| 公開用`configs/editing.toml` | 3モデル、全て同じvisual条件のみ |
| 公開用に含まれる図・集計CSV・notebook | なし |
| 環境 | SDK依存は範囲指定。完全固定lockfile、全モデルのrevision・vLLM実行環境の固定はなし |

HFのリモート上の最新内容は今回確認していない。「HFに必要な追加データが絶対に存在しない」とは結論していない。現行の取得・変換・READMEがこれらの追加データを提供していないことは確認できる。

## diagram側にある移植候補

以下は公開用リポジトリの外側にあり、現状のREADMEからは利用できない。内容を無条件にコピーすれば再現可能になるという意味ではなく、データ経路・機密情報・実行環境・採用条件の整合確認が必要。

- `figure7_four_conditions_13models_20261023/`：新12設定のverbalize/gold+visualと既存baselineをつなぐ処理・モデル別要求・judge・集計。
- `experiments/matched_open_generation_20261013/`：従来verbalization実験。
- `experiments/gold_text_vs_verbalize_first/`：過去のverbalize実装・prompt・指示抽出。
- `data/ablation_input_modality_100/metadata.jsonl`、`data/ablation_100/metadata.jsonl`：固定100件とgold text。
- `paper/figure40_no_qwen3vl_20261024/`、`paper/figure9_results_20261029/`：現行16設定の採用元・FSR・本文用集計。
- `paper/figure4_fsr_heatmap_20261020/`、`paper/figure5_operation_count_20261020/`、`paper/figure7_text_visual_20261020/`、`paper/figures5_9_large_text_20261028/`：掲載図の再計算と作図。
- `paper/ablation_human_vs_auto/`、`paper/human_evaluation/`：人手相関・一致度。
- `paper/ablation_gemini_judge_bias/`：代替judge表・モデル順位相関。
- `paper/ablation_layout_structure_failure_taxonomy/code_level_analysis_20261029/`：LAYOUT/STRUCTUREのコード差分・誤接続例。

## 検証の限界

今回の検証は、ファイル実在、defの入力経路、設定の列挙、実装・CLIの有無、データ集合・論文項目の対応の監査である。実API・GPU・コンテナbuild・モデル取得・リモート操作・新規課金実験は行っていない。

不足が明確に存在するため、現時点では「README通りで論文の全てを再現できる」という確認結果にはできない。
