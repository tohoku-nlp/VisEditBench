# Local validation — 2026-11-07

These results cover offline code/data verification. The checks below use local data and mocked responses with network access blocked; they do not perform live model inference, paid judging, HF upload, or container builds.

| Check | Result |
| --- | --- |
| `python -m unittest discover -s tests -v` | 40 tests passed |
| All code-model request conditions | 29 models × 4 conditions = 116 request shapes materialized; sampling options preserved within each configuration |
| Figure 9 grid | 16 configurations × 4 conditions = 64 groups, with the 12/4 cohort/profile distinction preserved |
| Both vLLM definitions | Definitions and `%files` inputs exist and point to the correct backend |
| Local benchmark conversion | All 694 records verified; images copied unchanged; gold text joined for all 694 records with the supplement |
| Standalone CLI validation | Temporary repository copy, network blocked: data preparation, legacy basic workflow, all four Gemini conditions, mock response import, judge preparation, aggregation, and archived table/figure generation passed |
| Real TikZ rendering | Synthetic valid document rendered; invalid overwrite removed stale success output; rendering failure stayed in the FSR denominator |
| Input/journal protection | Duplicate submission, concurrent submission, unknown outcome, changed input, retry subset and merging checks passed |
| Python syntax | All repository Python files parsed; new functions have Japanese docstrings |
| Public reproduction bundle | Gold text, cohorts, filtered/anonymized human ratings, archived judgments and updated Table 2/5/6 inputs included; all file hashes verified |
| Metadata compatibility | License additions and JSON key/row reordering accepted; modified code or PNG bytes rejected |
| Stability sampling | Fixed seed chooses 50 rendered outputs independently of row order; all five judge requests use the same image bytes |
| Updated Table 6 | Four image metrics for 29 models independently match the November 7 correlation snapshot |

The standalone command was run from the repository root with the existing local Python environment:

```bash
python scripts/validate_local_snapshot.py \
  --snapshot ../dataset/VisEditBench \
  --supplement reproduction
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

Figure PDFs/PNGs were regenerated in the offline CLI checks. PDF export now embeds TrueType fonts (`pdf.fonttype=42`) to avoid introducing Type 3 fonts.

Primary package versions used for these checks are recorded in [constraints-tested.txt](../constraints-tested.txt). This is not a transitive lockfile or a historical environment reconstruction. No clean-environment installation was performed.

## Remaining verification

The reproduction data is bundled in this Git repository, so a separate HF supplement upload is not required. The same offline checks are run against both the original local benchmark snapshot and the licensed/reordered metadata used for HF. New model/API execution and container builds are separate from archived-score analysis; run them in the intended environment with the documented settings. The public stability workflow uses a fixed recorded selection from completed outputs and does not require the historical private selection. See [coverage](coverage.md).
