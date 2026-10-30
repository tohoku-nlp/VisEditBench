<div align="center">

# VisEditBench: A Benchmark for Vector-Format Diagram Editing with Visual Instructions

[![Project Page](https://img.shields.io/badge/VisEditBench-Project%20Page-green?logo=googlechrome&logoColor=white)](https://viseditbench.github.io/)
[![Dataset](https://img.shields.io/badge/%F0%9F%A4%97%20Dataset-VisEditBench-yellow)](https://huggingface.co/datasets/tohoku-nlp/VisEditBench)
[![Code](https://img.shields.io/badge/GitHub-VisEditBench-black?logo=github&logoColor=white)](https://github.com/tohoku-nlp/VisEditBench)

</div>

This repository contains the official implementation of **VisEditBench: A Benchmark for Vector-Format Diagram Editing with Visual Instructions**.

## Contents

- [Overview](#overview)
- [Requirements](#requirements)
- [Installation](#installation)
- [Data Preparation](#data-preparation)
- [Reproducing Experiments](#reproducing-experiments)
- [Verbalization Experiments](#verbalization-experiments)
- [Local and Direct Image Models](#local-and-direct-image-models)
- [Archived Results and Figures](#archived-results-and-figures)
- [Public Human Annotation Data](#public-human-annotation-data)
- [Repository Structure](#repository-structure)
- [Verification](#verification)
- [Citation](#citation)

## Overview

**VisEditBench** evaluates models' ability to edit vector-format diagrams from **visual instructions**. Given an original diagram image, its TikZ source, and an annotated instruction image, the model produces edited TikZ code and a rendered diagram that reflect the requested changes.

The benchmark contains **694 human-curated examples** spanning **five edit types** and **one to three edit operations** per instruction. The repository registers **32 models**: 29 TikZ-editing models and 3 models that directly generate edited images.

| Edit type | Description |
| --- | --- |
| `ADD` | Add diagram elements |
| `REMOVE` | Remove diagram elements |
| `UPDATE` | Modify properties of existing elements |
| `LAYOUT` | Change spatial arrangement |
| `STRUCTURE` | Modify structural relationships and connections |

Edited diagrams are evaluated along three image-level aspects:

| Metric | Description |
| --- | --- |
| IA — Instruction Adherence | How faithfully the output follows the instruction; 1–5 |
| CP — Content Preservation | How well content outside the requested edits is preserved; 1–5 |
| DR — Diagram Readability | How clearly the edited diagram can be read; 1–5 |
| FSR — Full-score Rate | Percentage of examples receiving IA = CP = DR = 5 |

Code-level evaluation additionally reports IA, CP, Overall Edit Score (OES), and Compile Failure Rate (CFR). See the [model registry](configs/models.json), [prompt definitions](configs/prompts.json), and [reproduction coverage](docs/coverage.md) for configuration details.

## Requirements

- Python **3.11 or later**.
- **TeX Live** with `pdflatex`, `pdfcrop`, and the required TikZ packages, plus **Poppler** (`pdftoppm`), for rendering generated TikZ.
- Provider API keys for the selected API-based generation and evaluation models.
- A compatible **NVIDIA GPU / CUDA environment** for local vLLM or Diffusers inference.
- **Apptainer or Singularity** if using the supplied container definitions.
- A Hugging Face token when authentication is required for dataset or model access.

API-based experiments and archived-result analysis can run without a local GPU.

## Installation

### 1. Clone the repository

```bash
git clone https://github.com/tohoku-nlp/VisEditBench.git
cd VisEditBench
```

### 2. Install Python dependencies

```bash
python3.11 -m venv .venv
source .venv/bin/activate
python -m pip install -c constraints-tested.txt -e '.[analysis]'
```

`constraints-tested.txt` records the primary dependency versions used for offline verification. It is not a full transitive lockfile or a reconstruction of the historical paper environment.

For Qwen direct-image editing, install the optional dependencies in a compatible GPU environment:

```bash
python -m pip install -e '.[diffusers]'
```

### 3. Configure API keys

```bash
cp .env.example .env
```

Set the keys needed for the providers you use:

```dotenv
GOOGLE_API_KEY=
OPENAI_API_KEY=
ANTHROPIC_API_KEY=
OPENROUTER_API_KEY=
VLLM_API_KEY=
```

Offline data preparation, request construction, and archived-result analysis do not require provider credentials. `.env` is excluded from Git.

### 4. Install rendering tools

A full TeX Live installation is recommended. For example, on Ubuntu:

```bash
sudo apt-get install texlive-full poppler-utils
```

Run subsequent commands from the repository root with the Python environment activated.

## Data Preparation

### Download from Hugging Face

The benchmark is hosted at [tohoku-nlp/VisEditBench](https://huggingface.co/datasets/tohoku-nlp/VisEditBench). Dataset images and annotations are downloaded separately from the code.

```bash
python dataset.py download --revision main
python dataset.py verify
```

The downloader resolves the revision to a commit, verifies the inputs, preserves PNG image bytes, extracts source/reference TikZ, and creates shared experiment metadata:

```text
data/
├── raw_hf/                     # Downloaded Hugging Face snapshot
├── source_images/              # Original diagrams
├── visual_instructions/        # Annotated instruction images
├── target_images/              # Reference edited diagrams
├── source_tikz/                # Original TikZ code
├── target_tikz/                # Reference edited TikZ code
├── metadata/edit_metadata.jsonl
├── image_metadata.jsonl
├── code_metadata.jsonl
├── dataset_manifest.json       # Dataset commit and file hashes
└── reproduction/               # Reproduction supplement, when available
```

Keep `data/dataset_manifest.json` with your results. To use the same dataset version again, pass its resolved commit with `--revision`.

For a previously downloaded snapshot:

```bash
python dataset.py prepare --snapshot /path/to/dataset-snapshot
```

### Reproduction supplement

Gold-text comparisons, fixed ablation subsets, public human annotations, and archived-result analysis use the `reproduction/` supplement. When present in the snapshot, it is downloaded and verified automatically, and gold text is joined by sample ID.

Some supplemental assets are pending release or have not been located. The base visual-instruction benchmark can be prepared independently; the additional workflows require their corresponding supplement files. See [release instructions](docs/huggingface_release.md) and [coverage](docs/coverage.md) for availability.

An existing local supplement can be installed with:

```bash
python dataset.py supplement --source /path/to/reproduction
```

The paper's subsets are selected by recorded IDs rather than arbitrary sampling. Figure 6 and Figure 7 use different 100-example cohorts. Figure 9 uses the Figure 7 cohort for 12 configurations and the earlier verbalization cohort for 4 configurations; these cohorts share 18 IDs.

## Reproducing Experiments

### 1. Prepare generation requests

The main setting provides the original image, original TikZ code, and visual instruction. The following example prepares Gemini 3.1 Pro:

```bash
python paper.py prepare --experiment main \
  --models gemini-3.1-pro-preview \
  --output runs/main
```

Use `python paper.py models` to inspect registered models. Adding `--limit 1` creates an explicitly labeled software trial. Use a separate output directory for a subsequent full run.

### 2. Submit and collect generation results

```bash
VEB_JOB_DIR=runs/main/gemini-3.1-pro-preview__direct__main__full
VEB_JOB="$VEB_JOB_DIR/job.json"

python paper.py submit --job "$VEB_JOB"
python paper.py collect --job "$VEB_JOB"
```

`submit` sends requests to the provider and incurs usage charges. `collect` checks the existing batch once; repeat it until every chunk has been collected before continuing. Batch IDs, request hashes, states, and raw responses are saved in the run directory.

### 3. Render edited TikZ

```bash
python paper.py render --job "$VEB_JOB" --historical
```

`--historical` uses the paper's PDF-exists compilation success rule with fresh temporary files. Without it, rendering additionally requires a successful compiler exit status. Both modes disable shell escape and prevent stale-output reuse. Record the selected policy.

### 4. Evaluate and aggregate

```bash
python paper.py judge-prepare --job "$VEB_JOB" --kind image

VEB_JUDGE_JOB="$VEB_JOB_DIR/judges/gemini-3.1-pro-preview/image/repeat_00/job.json"
python paper.py submit --job "$VEB_JUDGE_JOB"
python paper.py collect --job "$VEB_JUDGE_JOB"
```

After all judgments have been collected:

```bash
python paper.py report --job "$VEB_JUDGE_JOB"
```

Results are saved next to the judge job as `cases.jsonl` and `summary.json`. The summary includes FSR, mean scores, standard deviations, CFR, edit-type/count breakdowns, and the one-operation edit-type analysis used by Figure 4.

Rendering failures receive the minimum image scores. Missing generation responses and judge errors remain pending and stop final reporting; they are not silently counted as model failures.

For code-level evaluation, prepare a separate judge job with `--kind code`. To compare judges, use `--judge-model claude-opus-4-6` on the **same generation job**. Existing diagrams are reused for judge comparison.

### Experiment configurations

```bash
python paper.py plan --experiment verbalization
```

| Experiment | Groups | Setting |
| --- | ---: | --- |
| `main` | 29 | Visual instructions with original image/code; 694 examples |
| `direct_image` | 3 | Direct image editing; 694 examples |
| `modality` | 16 | 4 models × 4 input variants; Figure 6 cohort |
| `text_visual` | 26 | 13 models × visual/gold text; Figure 7 cohort |
| `verbalization` | 64 | 16 configurations × 4 conditions; recorded cohort/profile per configuration |

Unavailable models are retained in the registry and stop at submission unless explicitly reviewed with `--include-blocked`. No alternative model is selected automatically. Hosted model revisions and unrecorded historical runtime settings can affect numerical reproduction.

Failed responses can be isolated with `python paper.py retry --job "$VEB_JOB"`. Successful retries are merged by ID and request fingerprint. Existing successes are not resent. See [workflow details](docs/reproducibility.md) for the retained backend implementations.

## Verbalization Experiments

Prepare the four prompting conditions for a selected model:

```bash
python edit.py verbalize \
  --models gemini-3.1-pro-preview \
  --output runs/verbalization
```

This is equivalent to `paper.py prepare --experiment verbalization`. Every condition receives the original diagram image and TikZ code.

| Condition | Instruction setting |
| --- | --- |
| `direct` | Edit from the visual instruction |
| `verbalize_first` | Verbalize the visual instruction, then edit within the same response |
| `gold_text` | Edit from the gold textual instruction |
| `gold_text_visual` | Edit using both visual and gold textual instructions |

In `verbalize_first`, the model returns `<instruction>...</instruction>` followed by the edited TikZ. The visual instruction remains available while editing. Generated instructions, TikZ, and raw responses are stored separately.

Submit, collect, and render each prepared job using the workflow above. Figure 9 uses **Gemini 3.6 Flash** for judging, so its judge jobs are under `judges/gemini-3.6-flash/`. Main and Figure 7 experiments default to Gemini 3.1 Pro.

Request templates preserve sampling options, explicit thinking settings, image order, and system/user arrangements. The four earlier configurations retain their separate seed-42 / 50,000-token profile. See [prompt definitions](configs/prompts.json) and [coverage](docs/coverage.md).

## Local and Direct Image Models

### Local vLLM inference

Use a compatible vLLM environment or the supplied container definitions:

| Purpose | Definition |
| --- | --- |
| Editing | [editing/vllm/vllm.def](editing/vllm/vllm.def) |
| Image evaluation | [evaluation_image-level/vllm/vllm.def](evaluation_image-level/vllm/vllm.def) |

The corresponding `apptainer/build.sh` scripts are run from the repository root. Container builds and GPU execution require validation in the target environment.

```bash
python paper.py prepare --experiment main \
  --models qwen3.5-4b-nothink \
  --output runs/qwen4b

python paper.py server-command --model qwen3.5-4b-nothink
```

`server-command` prints the configured launch command. Start that server in your GPU environment, then run:

```bash
python paper.py chat \
  --job runs/qwen4b/qwen3.5-4b-nothink__direct__main__full/job.json \
  --endpoint http://127.0.0.1:8000/v1
```

OpenRouter jobs use the same `chat` command with `--endpoint https://openrouter.ai/api/v1`.

### Direct image editing

```bash
python paper.py prepare --experiment direct_image \
  --models gemini-3-pro-image-preview \
  --output runs/direct_image

python paper.py image-run \
  --job runs/direct_image/gemini-3-pro-image-preview__direct__main__full/job.json
```

The registered backends include Gemini 3 Pro Image, GPT Image 1.5, and Qwen Image Edit. Their outputs are evaluated with image-level judges; code-level evaluation does not apply. See [backend details](docs/backends.md).

## Archived Results and Figures

With the reproduction supplement installed, regenerate saved analyses without model inference:

```bash
python paper.py reproduce --output runs/paper_reproduction
```

Open `runs/paper_reproduction/index.html`. Outputs include:

- Figure 3–9 PDFs/PNGs.
- Published-input tables and separately named recomputed CSV/LaTeX tables.
- FSR, mean/std, direct-image edit-type/count analyses, human correlations and ordinal agreement, and judge-rank correlations.
- Selected LAYOUT/STRUCTURE image comparisons and code differences.
- `reproduction_report.json`, with input provenance and remaining coverage gaps.

Archived-result analysis and new inference are separate workflows. Published snapshots keep their recorded versions; `_recomputed` outputs use the distributed ratings. See [coverage](docs/coverage.md) for missing original assets and differences between historical and new runs.

## Public Human Annotation Data

The reproduction supplement provides filtered human ratings and corresponding automatic judgments:

```text
data/reproduction/human/
├── ratings.jsonl
├── image_judges.jsonl
└── code_judges.jsonl
```

| Rating field | Description |
| --- | --- |
| `question_id` | Identifier used to match ratings to an evaluated output |
| `generated_model` | Model that generated the evaluated diagram |
| `rater` | Anonymized rater identifier |
| `instruction_adherence` | IA rating; 1–5 |
| `content_preservation` | CP rating; 1–5 |
| `diagram_readability` | DR rating; 1–5 |

The prepared release contains 1,350 filtered ratings for 300 outputs. Rater labels are anonymized; worker/assignment IDs and free-text comments are excluded. Original five-repeat stability assets remain unavailable; the [stability workflow](docs/stability.md) accepts the original 50-output selection once provided.

## Repository Structure

```text
VisEditBench/
├── configs/                    # Model registry, prompts, request templates, TOML examples
├── editing/
│   ├── batch_api/              # Batch API editing
│   ├── vllm/                   # Local vLLM clients and container definition
│   ├── openrouter/             # OpenRouter client
│   ├── diffusers/              # Qwen image editing
│   └── compile_tikz_batch.py
├── evaluation_image-level/     # Image-level judging backends
├── evaluation_code-level/      # Code-level judging backends
├── viseditbench/               # Data, workflow, provider, and analysis modules
├── scripts/                    # Offline validation and release-preparation utilities
├── tests/                      # Synthetic-input and mocked-API tests
├── docs/                       # Protocol details, coverage, and validation
├── dataset.py                  # HF download and metadata preparation
├── edit.py                     # Editing and verbalization entry point
├── eval.py                     # Retained evaluation entry point
├── paper.py                    # Experiment planning, inference, judging, and analysis
├── results.py                  # Image-score aggregation
├── constraints-tested.txt
└── pyproject.toml
```

## Verification

```bash
python -m unittest discover -s tests -v
```

The offline tests cover four-condition requests for all 29 code models, the 64-group Figure 9 plan, fixed cohorts, def-file inputs, retry/duplicate-submission handling, extraction, image processing, statistics, and reporting. Real TikZ rendering is checked when TeX/Poppler are available.

Offline tests do not establish successful execution on every provider/GPU environment or exact regeneration of historical outputs. Current verification results and remaining data gaps are documented in [validation](docs/validation.md) and [coverage](docs/coverage.md).

## Citation

For project and publication information, see the [project page](https://viseditbench.github.io/).

```bibtex
@misc{taneguchi2026viseditbench,
  title  = {VisEditBench: A Benchmark for Vector-Format Diagram Editing with Visual Instructions},
  author = {Akito Taneguchi and Haruto Yoshida and Jun Suzuki and Itsumi Saito},
  year   = {2026},
  url    = {https://viseditbench.github.io/},
}
```
