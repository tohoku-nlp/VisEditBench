# Reproducibility details

The integrated paper workflow is documented in the [README](../README.md) and [current coverage](coverage.md). The legacy clients below remain available; the new `paper.py` workflow adds the four prompting conditions, fixed cohorts, archived statistics, and figure generation.

## Sources and integration changes

The editing/evaluation backends originate from the local `NeurIPS2026_submit` submission copy. The code-level score extractor comes from the current `diagram/batch_api_pipeline/evaluation_code/lib/extract.py`. [implementation_sources.json](implementation_sources.json) records the original source path and SHA-256 for every imported file. These hashes describe the input sources before the integration changes below.

This public directory is self-contained: no import or launcher depends on the parent `diagram` directory. Dependencies are declared in `pyproject.toml`. Its ranges specify supported SDK dependencies, not a frozen reconstruction of the authors' historical environment. Save `python -m pip freeze` for a completed run.

Integration changes:

- Hugging Face snapshots are converted into common metadata and materialized source/reference `.tex` files. Images are copied byte-for-byte and not resized.
- Dataset commit and file hashes are saved in a manifest. Preparation checks unique IDs, type/count consistency, code fields, and PNG integrity.
- Generated output paths are attached without dropping failed samples. Image metric aggregation retains the expected denominator and distinguishes missing judgments from generation/rendering failure.
- Claude's shortened API IDs are mapped back to complete sample IDs before saving `.tex` files.
- Image and code score parsing rejects fractional, boolean, or out-of-range scores. Invalid judgments must be retried.
- Rendering checks the compiler exit status, removes stale outputs on overwrite, and disables shell escape. A rendering failure cannot reuse an earlier PDF/PNG as a new success. Documents requiring shell escape will fail under this configuration; report that setting with CFR.
- `.env` loading is limited to this repository's own `.env`. Previewing requests does not overwrite saved request JSONL. `--preview` is restricted to request generation.
- Local vLLM readiness checks use `/health`.
- Code evaluation skips explicitly recorded missing generations and reports them in the log. Non-compiling generated code is still available for code evaluation.

## Settings retained from the submission

Prompt definitions remain in each backend rather than being silently unified; different backends originally used different wording. The main visual Batch API prompt is in `editing/batch_api/lib/generate.py`. Image judge prompts and important-note behavior are in `evaluation_image-level/batch_api/lib/generate.py`; code judge prompts are in `evaluation_code-level/batch_api/lib/generate.py`.

The batch generation library retains the submission's provider-specific token/reasoning defaults. In particular, Claude generation uses an 8,192-token cap; the local vLLM and OpenRouter clients default to 128,000 tokens. These defaults must not be described as a uniform 128,000-token setting. Inspect and preserve the generated request JSONL to establish what was actually sent. A provider may impose a different effective limit or change defaults. Image/code judge configs explicitly set temperature to 0.0.

The supplied TOML files demonstrate complete runs for Gemini, GPT, and Claude; they are not a manifest of every model/configuration and every updated result in the paper. All model IDs/settings and the target sample list must be recorded for any expanded reproduction. No historical provider result or seed-free Qwen3.5 experiment is claimed to be reproduced by these examples.

## Input variants

The Batch API editing backend supports the following `system_instruction_type` values (see its `SYSTEM_PROMPTS` dictionary for all variants):

| Value | Inputs |
| --- | --- |
| `visual_instruction` | Visual instruction image |
| `visual_instruction_w_org_image` | Original image + visual instruction |
| `visual_instruction_w_org_tikz` | Original code + visual instruction |
| `visual_instruction_w_org_image_w_org_tikz` | Original image/code + visual instruction |
| `text_instruction_w_org_image_w_org_tikz` | Original image/code + gold textual instruction |

Use separate experiment names and output directories for every condition. The `edited_input_type` field in the evaluation config and `--input-type` in attachment/aggregation must agree. They are identifiers for joining results, not automatic switches of the generation prompt.

## Gold textual instructions

The validated local snapshot has `id`, `source_image`, `visual_instruction`, `target_image`, `uri`, `source_tikz`, `target_tikz`, `edit_type`, and `edit_operation`. It has no gold text field. `*_file_name` image-field variants are also supported. If a future dataset revision includes `text_instruction`, preparation preserves it.

To supply independently obtained gold text for the existing schema, create a UTF-8 JSONL file with one unique sample ID and instruction per line:

```json
{"id":"00000069_STRUCTURE_1_edit_0","text_instruction":"The gold instruction for this sample."}
```

```bash
python dataset.py prepare --snapshot data/raw_hf --force \
  --text-instructions /path/to/gold_text.jsonl
```

This example describes the file format; its sentence is not a benchmark annotation. Supplemental IDs must belong to the dataset. Samples without supplied text remain without text. Gold instructions are never inferred from target image/code. The new reproduction supplement supplies the actual annotations, cohort IDs, and archived results; see the [HF release guide](huggingface_release.md). The four-condition implementation is in `viseditbench/workflow.py`.

## Metrics and reporting

Image IA, CP, and DR are integer scores in [1,5]. FSR counts a sample only if all three are 5. The image summary uses the full prepared metadata as the denominator. `missing_generation` and `compile_failed` counts are separate; missing output counts can include interrupted inference, so finish generation before interpreting them as model failures. The rendering failure count is operational and is not by itself a full code-quality assessment.

The optional code judge outputs IA, CP, and OES; code-only scores must not be mixed with image-level IA/CP/DR. Human-rating correlation analyses and classical image-metric comparisons require their own evaluation annotations and are outside this pipeline.

Keep the dataset manifest, input metadata, TOML settings, request/response JSONL, batch identifiers, compiler logs, SDK/environment versions, and final score files together. Hosted model drift and nondeterministic inference prevent a guarantee of exact historical numerical results.
