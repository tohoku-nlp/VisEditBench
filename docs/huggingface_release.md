# 再現用データの配布

再現用追加データは公開コードの`reproduction/`に同梱しています。HF上の追加公開を前提とせず、`dataset.py download`または`dataset.py prepare`で自動的に検証・配置されます。

含まれるデータ:

- 694件のgoldテキスト指示と固定subsetのID。
- Figure 9の6,400評点、32モデルの主実験画像評点、コード評点。
- 匿名化した人手評価1,350評点・300出力と自動judge評点。
- 掲載表・図の入力値と、LAYOUT/STRUCTUREの比較用画像・コード。
- 2026-11-07のClaude再評価とTable 2/5/6の更新入力。
- 全追加ファイルのSHA-256と、対象ベンチマーク内容のhash。

```bash
python dataset.py download --revision main
python paper.py reproduce --output runs/paper_reproduction
```

既存の準備済みデータへ追加する場合:

```bash
python dataset.py supplement --source reproduction
```

古い準備済みデータに内容hashがない場合は、`dataset.py prepare --snapshot PATH --force`で再準備してください。列順・行順・licenseの追加だけでは対象データの同一性は変わりません。ID・コード・画像などが異なる場合は結合を拒否します。

## 任意のHFミラー

同じ追加データをHFにも置く場合は、著者が以下を実行できます。再現の必須手順ではありません。ベンチマークのREADMEやsplit設定の変更も不要です。

```bash
hf upload tohoku-nlp/VisEditBench ./reproduction reproduction --repo-type dataset
```

公開済みsnapshotを変更する場合は、新しいコミットとmanifestを保存してください。`scripts/build_hf_release.py`は著者workspaceから別の公開候補を作成するためのツールです。同梱の公開snapshotの内容は`reproduction/manifest.json`が固定しています。
