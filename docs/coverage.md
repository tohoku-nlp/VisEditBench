# 再現経路と検証範囲

2026-11-07更新。再現用データはGitHubの`reproduction/`に同梱しており、Hugging Faceへの追加公開を待つ必要はありません。

## 保存済み結果の再分析

```bash
python dataset.py download --revision main
python dataset.py verify
python paper.py reproduce --output runs/paper_reproduction
```

ベンチマーク本体をHFから取得し、同梱の694件のgold指示・固定subset・匿名化評点・保存済みモデル評点を自動結合します。入力データのコミットとhash、追加データ全ファイルのhashを記録します。既存スナップショットには`dataset.py prepare --snapshot PATH`も使えます。

| 対象 | 公開入力と実行内容 |
| --- | --- |
| gold・固定subset | 694件のテキスト指示とFigure 6/7・従来verbalizationの固定IDを同梱 |
| 主実験 | 32モデルの画像評点、コード評点、全件分母FSR・平均・標準偏差を再集計 |
| Figure 3–9 | データ分布と保存済みの図入力から作図。対象集合と分母を保持 |
| 人手評価 | 匿名化した1,350評点・300出力から相関とordinal alphaを再計算 |
| 代替judge・順位相関 | 同じ29モデルを対応付け、同順位を考慮して再計算 |
| 定性例 | LAYOUT・STRUCTUREの元画像・指示・出力・コードとdiffを同梱 |
| 2026-11-07更新 | 新Qwen3.5非Thinking4モデルのClaude再評価、訂正Table 2・更新Table 5からTable 6を再計算 |

`runs/paper_reproduction/index.html`が結果の索引です。`reproduction_report.json`の`archived_analysis_verified`は同梱入力を使う再分析の検証、`fresh_inference_executed`はその実行で新しいモデル推論を行ったかを表します。保存済み評点の再分析を、新しいAPI/GPU実験の成功とは扱いません。

掲載表のスナップショットと、raw評点から計算した`_recomputed`は別名で出力します。コード評点の欠測・不正値がある群では実際の`n_code`を併記し、完全な694件の集計として扱いません。11月7日の更新値も別名で保存し、従来版の入力を上書きしません。

## 新しい推論と評価

`paper.py prepare/submit/collect/chat/image-run/render/judge-prepare/report`で、新しい生成・描画・評価を実行できます。29コードモデルと3画像モデルを登録しています。API呼び出しには各自の認証情報、ローカルモデルには対応するGPU環境が必要です。要求・応答・入力画像・モデル設定・対象ID・失敗状態を保存します。

同じ入力・設定でも、提供モデルの変更や推論の非決定性により歴史的な出力と数値の完全一致は保証しません。コンテナ定義・各providerの要求生成・再試行・集計はオフラインで検証し、実際のAPI/GPU実行とは区別しています。

## 安定性評価：50件×5回

公開ベンチマークから生成した完了済みrunを使い、描画成功出力からseedで50件を固定し、同じ画像を5回採点できます。

```bash
python paper.py stability-prepare \
  --generation-job runs/main/MODEL__direct__main__full/job.json \
  --seed 42 --output runs/stability
```

選択ID・画像hash・seedを保存します。250評点が揃ったときだけ平均と反復間の標準偏差を確定します。5回の間に図を再生成しません。新しく選んだ対象は論文当時の対象と同一とは称さず、実行した選択と設定を記録します。詳細は[安定性評価](stability.md)を参照してください。
