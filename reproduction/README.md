# Reproduction data

This directory is distributed with the GitHub code. No separate HF supplement release is required. The benchmark images and metadata are downloaded from `tohoku-nlp/VisEditBench`; `dataset.py download` and `dataset.py prepare` verify and install this data automatically.

- `gold_text.jsonl`: 694 benchmark gold instructions, joined only by sample ID.
- `cohorts.json`: fixed main/ablation sample IDs.
- `scores/`: archived image/code model judgments and Figure 9 inputs.
- `human/`: 1,350 anonymized ratings for 300 outputs and corresponding automatic scores. Worker identities, assignments, free-text comments, payment details and source URLs are excluded.
- `published/`: versioned input tables for the earlier archived analyses.
- `cases/`: selected LAYOUT/STRUCTURE examples, PNGs and TikZ code.
- `updates/20261107/`: new Qwen3.5 nonthinking Claude evaluations, corrected Table 2 (9B CP = 2.736), updated Table 5, and recomputed Table 6. The earlier archived source versions are not overwritten.
- `dataset_construction.json`: published dataset counts.
- `manifest.json`: file hashes and a content identity for the benchmark, independent of JSON field/row order and the license column.

After preparing the benchmark:

```bash
python paper.py reproduce --output runs/paper_reproduction
```

This is archived-score analysis; it does not call a model API. To repeat judge stability, follow `docs/stability.md`: choose 50 completed generated outputs with a fixed seed and evaluate those same images five times. The public workflow does not require a private historical selection file.
