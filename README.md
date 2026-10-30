# VisEditBench

Code for **VisEditBench: A Benchmark for Vector-Format Diagram Editing with Visual Instructions**.

- Dataset: [tohoku-nlp/VisEditBench](https://huggingface.co/datasets/tohoku-nlp/VisEditBench)
- Model registry: [configs/models.json](configs/models.json), 29 TikZ-editing models and 3 direct-image models
- Four-condition prompts: [configs/prompts.json](configs/prompts.json)
- Coverage and validation: [docs/coverage.md](docs/coverage.md), [docs/validation.md](docs/validation.md)

The repository supports archived-result analysis and new inference runs. Archived analysis reconstructs tables/figures from saved ratings and published values. New inference calls the selected providers or a user-managed GPU server. These are separate workflows; offline tests do not demonstrate exact regeneration of historical API/GPU outputs.

**Release status:** the reproduction supplement is prepared locally and must be added to Hugging Face by the dataset owner. Until then, gold-text comparisons and archived-result analysis require the local supplement. The original 50-output, five-repeat stability data and raw dataset-construction validation labels have not been located. Programs report these gaps and do not claim complete paper reproduction.

## Setup

Use Python 3.11 or later and run commands from the repository root.

```bash
python3.11 -m venv .venv
source .venv/bin/activate
python -m pip install -c constraints-tested.txt -e '.[analysis]'
cp .env.example .env
```

For inference, fill the necessary `GOOGLE_API_KEY`, `OPENAI_API_KEY`, `ANTHROPIC_API_KEY`, or `OPENROUTER_API_KEY` in this repository's `.env`. Local vLLM authentication can use `VLLM_API_KEY`. Offline preparation and archived analysis do not need credentials.

`constraints-tested.txt` records primary dependency versions used for offline checks. It is not a full transitive lockfile or the historical paper environment.

TikZ rendering requires `pdflatex`, `pdfcrop`, diagram packages, and Poppler's `pdftoppm`. A full TeX Live installation is recommended. On Ubuntu, for example:

```bash
sudo apt-get install texlive-full poppler-utils
```

## Download data

```bash
python dataset.py download --revision main
python dataset.py verify
```

The downloader resolves the revision to a Hugging Face commit, downloads metadata/PNGs and `reproduction/` when present, verifies inputs, extracts source/reference TikZ, and joins gold text by ID. Data and results are Git-ignored. Preserve `data/dataset_manifest.json` and use its resolved commit for subsequent runs.

The supplement supplies exact cohort IDs, gold instructions, anonymized human ratings, archived judgments, published values, and qualitative cases. File hashes and the base-metadata hash are checked before use. No benchmark data are embedded in this code repository. See the [HF release guide](docs/huggingface_release.md) for the owner's upload command.

For an already downloaded snapshot:

```bash
python dataset.py prepare --snapshot /path/to/dataset-snapshot
```

For the author-only local supplement prepared alongside this repository:

```bash
python dataset.py supplement --source ../VisEditBench_hf_release_20261030/reproduction
```

Use all 694 records for paper experiments. `--limit` creates an explicitly labeled software trial, not the paper's 100-example subset. Figure 6 uses `figure6`, Figure 7 uses `figure7`, and Figure 9 uses `figure7` for 12 configurations and `legacy_verbalization` for 4. The latter two cohorts share only 18 IDs.

## Reconstruct saved results without inference

After obtaining the supplement:

```bash
python paper.py reproduce --output runs/paper_reproduction
```

Open `runs/paper_reproduction/index.html`. Outputs include:

- Figure 3–9 PDFs/PNGs, including the one-operation Figure 4 heatmap and four-condition Figure 9.
- CSV/LaTeX tables for archived published inputs and separately named recomputed metrics.
- FSR, means, standard deviations, direct-image type/count analyses, human correlations and ordinal agreement, and judge-rank correlations.
- Original/instruction/output assets and code differences for the selected LAYOUT and STRUCTURE cases.
- `reproduction_report.json`, containing provenance, reconstructed counts, and unresolved gaps.

Published snapshots retain their recorded source versions. Files ending in `_recomputed` are calculated from distributed ratings. Only one code judge's complete question-ID-based raw correlation scores was located; published snapshots for the other code judges are retained separately. See [coverage](docs/coverage.md).

## Run new experiments

```bash
python paper.py models
python paper.py plan --experiment verbalization
```

| Experiment | Groups | Cohort / condition |
| --- | ---: | --- |
| `main` | 29 | 694 diagrams, visual instruction + original image/code |
| `direct_image` | 3 | 694 diagrams, direct image editing |
| `modality` | 16 | 4 models × 4 input variants, Figure 6 cohort |
| `text_visual` | 26 | 13 models × visual/gold text, Figure 7 cohort |
| `verbalization` | 64 | 16 configurations × 4 conditions, per-configuration cohort/profile |

Recorded unavailable models remain identifiable; submission stops unless `--include-blocked` is explicitly supplied after checking access. No replacement is silently selected. Hosted versions, unrecorded historical revisions, and runtime changes limit exact numerical reproduction.

### Four-condition preparation

```bash
python edit.py verbalize --models gemini-3.1-pro-preview --output runs/verbalization
```

This is equivalent to `python paper.py prepare --experiment verbalization ...`. It creates four jobs for the selected model:

1. `direct`: visual instruction editing.
2. `verbalize_first`: **one response** containing `<instruction>...</instruction>` followed by TikZ. The visual instruction remains available while editing.
3. `gold_text`: gold textual instruction editing.
4. `gold_text_visual`: editing with both instruction modalities.

Templates retain each configuration's sampling options, system/user arrangement, image order, and explicit thinking setting. Local images replace historical GCS URLs. The old four configurations keep their separate seed-42 / 50,000-token profile. Generated instructions, edited code, and raw responses are saved separately.

For a small software trial, add `--limit 1`. Use a new output directory for the full run; prepared requests are not overwritten.

### Submit, collect, render, and judge

For the Gemini verbalization job:

```bash
JOB=runs/verbalization/gemini-3.1-pro-preview__verbalize_first__main__full/job.json
python paper.py submit --job "$JOB"
python paper.py collect --job "$JOB"
```

`submit` contacts the provider and incurs charges. `collect` checks the existing job once; repeat until all chunks are collected. IDs/states are saved; submissions with unknown outcomes are not resent automatically. Request chunks are kept under 30 MB.

```bash
python paper.py render --job "$JOB" --historical
python paper.py judge-prepare --job "$JOB" --kind image
```

`--historical` uses the paper's PDF-exists compilation rule with fresh temporary files. Without it, rendering also requires a successful compiler exit. Both disable shell escape and prevent stale-output reuse. Record the chosen policy.

Figure 9 uses Gemini 3.6 Flash; main and Figure 7 runs default to Gemini 3.1 Pro.

```bash
JUDGE=runs/verbalization/gemini-3.1-pro-preview__verbalize_first__main__full/judges/gemini-3.6-flash/image/repeat_00/job.json
python paper.py submit --job "$JUDGE"
python paper.py collect --job "$JUDGE"
python paper.py report --job "$JUDGE"
```

Reports contain per-case scores, overall/edit-type/count metrics, standard deviations, CFR, and one-operation edit-type metrics. FSR requires IA = CP = DR = 5. Outputs failing to render receive 1; API/judge failures remain pending and stop final reporting.

For code judging, use `judge-prepare --kind code`. For alternative judges, use `judge-prepare --judge-model claude-opus-4-6` on the **same generation job**; do not regenerate diagrams. For repeated judging, use `--repeats 5`; see [stability instructions](docs/stability.md) for the original 50-output protocol.

`python paper.py retry --job "$JOB"` isolates failed responses. Submit/collect the returned retry job; successes merge by ID and request fingerprint. Existing successes are not resent. If submission was interrupted before saving its ID, use `recover --chunk N --job-id CONFIRMED_PROVIDER_JOB_ID` after checking the provider's job; do not delete the journal.

### vLLM and direct-image models

Both container definitions are present:

- [editing/vllm/vllm.def](editing/vllm/vllm.def)
- [evaluation_image-level/vllm/vllm.def](evaluation_image-level/vllm/vllm.def)

Run their `apptainer/build.sh` scripts from the repository root. The recipes use `vllm/vllm-openai:v0.19.1`; actual builds/GPU runs have not been tested locally.

```bash
python paper.py server-command --model qwen3.5-4b-nothink
```

This prints the recorded command without starting a server. After the user starts it, send a prepared vLLM job with:

```bash
python paper.py chat --job /path/to/job.json --endpoint http://127.0.0.1:8000/v1
```

OpenRouter uses `https://openrouter.ai/api/v1`. Direct-image jobs prepared with `--experiment direct_image` use `paper.py image-run --job ...`. Qwen additionally needs `python -m pip install -e '.[diffusers]'` and a compatible CUDA environment. See [backend details](docs/backends.md).

## Offline checks

```bash
python -m unittest discover -s tests -v
```

Tests use synthetic inputs and mocked APIs, plus real TikZ rendering when available. They check all 29 code models' four conditions, the 64-group plan, exact cohorts, both def inputs, thinking/seed settings, duplicate-submission protection, extraction, retry merging, image processing, statistics, and reporting.

The owner can also use `scripts/validate_local_snapshot.py`. Actual HF download, clean-environment installation, container building, provider availability, and paid API/GPU inference remain untested in the target environment. Unavailable original assets are reported in the coverage report.
