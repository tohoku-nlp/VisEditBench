# Hugging Face追加公開用データ

`diagram/VisEditBench_hf_release_20261030/` に追加公開用データを準備しました。`reproduction/` を、既存の `tohoku-nlp/VisEditBench` データセットのルートへ追加します。コードのGitHubリポジトリには含めません。

含まれるもの:

- 694件のgoldテキスト指示。
- Figure 6、Figure 7、従来verbalizationの固定100件ID。Figure 6と従来verbalizationは同じ集合、Figure 7とは18件だけ共通です。
- Figure 9の6,400評点、32モデル計22,208件の主実験画像評点。
- 人手評価300出力・1,350評点と自動judge評点。評価者IDは独立した匿名ラベルへ置換。自由記述、連絡先、作業者/割当/HIT ID、時刻、支払情報、画像URLは含めていません。
- 掲載表・図の入力値、LAYOUT/STRUCTUREの比較用画像・コード。
- 全追加ファイルと元入力のSHA-256、対象ベースmetadataのSHA-256。

ユーザーが追加公開用フォルダで実行するコマンド:

```bash
cd /Users/akatsuki/projects/2026/diagram/VisEditBench_hf_release_20261030
hf upload tohoku-nlp/VisEditBench ./reproduction reproduction --repo-type dataset
hf upload tohoku-nlp/VisEditBench ./README.md README.md --repo-type dataset
```

エージェントはアップロードしていません。追加後のHFコミットを記録し、コード側で `python dataset.py download --revision COMMIT --force` を実行します。

同梱のREADMEは元のデータカードを保持し、defaultの対象を `dataset/metadata.jsonl` に限定する設定を追加しています。再現用のJSONLを694件のbenchmark行へ混在させないための変更です。

公開前のローカル確認:

```bash
python dataset.py prepare --snapshot ../dataset/VisEditBench
python dataset.py supplement --source ../VisEditBench_hf_release_20261030/reproduction
python paper.py reproduce --output runs/paper_reproduction
```

著者による追加データの再作成:

```bash
python scripts/build_hf_release.py --workspace .. --output ../VisEditBench_hf_release_NEW_VERSION
```

公開済み入力を変えず、新版を別ディレクトリへ作ります。`--overwrite-unpublished` は未公開の作業一式を修正するためのオプションです。

元の50件×5回の安定性評価データと、構築時の全2,819件の検証ラベルは未確認です。`manifest.json` と再現レポートの `coverage_gaps` へ記録しています。
