# Local validation — 2026-10-30

These results cover offline code/data verification. No live model inference, paid judging, remote operation, HF upload, or container build was performed.

| Check | Result |
| --- | --- |
| `python -m unittest discover -s tests -v` | 32 tests passed |
| All code-model request conditions | 29 models × 4 conditions = 116 request shapes materialized; sampling options preserved within each configuration |
| Figure 9 grid | 16 configurations × 4 conditions = 64 groups, with the 12/4 cohort/profile distinction preserved |
| Both vLLM definitions | Definitions and `%files` inputs exist and point to the correct backend |
| Local benchmark conversion | All 694 records verified; images copied unchanged; gold text joined for all 694 records with the supplement |
| Standalone CLI validation | Temporary repository copy, network blocked: data preparation, legacy basic workflow, all four Gemini conditions, mock response import, judge preparation, aggregation, and archived table/figure generation passed |
| Real TikZ rendering | Synthetic valid document rendered; invalid overwrite removed stale success output; rendering failure stayed in the FSR denominator |
| Input/journal protection | Duplicate submission, concurrent submission, unknown outcome, changed input, retry subset and merging checks passed |
| Python/shell syntax | 63 Python files parsed; 11 shell scripts passed `bash -n` |
| Source provenance | All 87 recorded source SHA-256 values still matched |
| Publication audit | 100 text/source files; no benchmark data, raw ratings, model weights, `.env`, or virtual environment included |

The standalone command was run from the repository root with the existing local Python environment:

```bash
python scripts/validate_local_snapshot.py \
  --snapshot ../dataset/VisEditBench \
  --supplement ../VisEditBench_hf_release_20261030/reproduction
```

Mock scores/images are software fixtures and are not reported as model results. Real rendering is tested separately in the unit suite.

## Archived numerical checks

- Main image records: 32 × 694 = **22,208**.
- Figure 9 records: 16 × 4 × 100 = **6,400**.
- Figure 9 verbalization improves over visual-only in **15/16** configurations; gold text exceeds verbalization in **14/16**.
- Figure 9 mean FSR: **39.5**, **48.3125**, **55.0625**, **55.4375** percent for direct, verbalization, gold text, and both modalities.
- Human evaluation: **300** outputs, **1,350** filtered ratings, with anonymized rater labels.
- Ordinal alpha: IA **0.8044935428**, CP **0.6665221038**, DR **0.6488031918**, matching the manuscript's rounded 0.80 / 0.67 / 0.65.
- Main code scores: **40,175** parsed records across 58 model/judge groups. 32 groups contain all 694 scores; other groups' actual sample counts remain visible in the recomputed tables.
- Figure 6 / previous verbalization and Figure 7 use different 100-record cohorts with **18** overlapping IDs.

Figure PDFs/PNGs were generated, and Figures 6 and 9 were visually inspected for readable labels, legend placement, and clipping. Figure 5 retains the legend on the right in one column.

Primary package versions used for these checks are recorded in [constraints-tested.txt](../constraints-tested.txt). This is not a transitive lockfile or a historical environment reconstruction. No clean-environment installation was performed.

## Remaining verification

The HF supplement is prepared locally but not uploaded. Live HF download, provider access, GPU/container execution, and exact historical model outputs remain unverified. Original 50-case five-repeat stability assets, raw construction-validation labels, explanatory Figure 1/2 assets, and the complete latest paper TeX have not been located. See [coverage](coverage.md); the report intentionally leaves `all_paper_artifacts_verified` false.
