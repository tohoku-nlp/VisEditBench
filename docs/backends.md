# Additional backends

The paper workflow uses the model-specific templates in `configs/request_templates.json`; `paper.py server-command` prints its server settings. The commands below describe the retained submission clients. Their defaults differ from the new paper workflow. Container definitions now exist for both editing and evaluation, with corrected local input paths; building and GPU execution remain untested.

Run commands from the repository root after preparing the data. The commands here are examples for the user to run; no model downloads, server launches, or API calls are performed by the offline tests.

## Local vLLM

Install vLLM in a separate CUDA-compatible environment following the [official installation guide](https://docs.vllm.ai/en/latest/getting_started/installation/). The optional Apptainer recipes and launch scripts are in `editing/vllm/` and `evaluation_image-level/vllm/`. Adjust GPU allocation, model paths, and context length for your hardware. They are single-host scripts, not cluster job submissions.

Start a server, for example:

```bash
vllm serve Qwen/Qwen3-VL-8B-Instruct --host 127.0.0.1 --port 8000
```

In the repository's Python environment, run:

```bash
python edit.py vllm --mode visual_w_org_tikz \
  --input-path data/metadata/edit_metadata.jsonl \
  --output-path runs/qwen3vl8b/outputs.jsonl \
  --hosts 127.0.0.1 --ports 8000 \
  --model Qwen/Qwen3-VL-8B-Instruct --seed 42 --max-tokens 128000 \
  --tex-output-dir work/qwen3vl8b/tikz
```

The server must have enough context for the requested output budget plus image/code input; reduce `--max-tokens` if needed and report the changed setting. `--enable-thinking` passes the corresponding chat-template option; model support is required. The submission client defaults to seed 42. This is separate from later seed-free Qwen experiments and should not be presented as the same setting.

Render the TikZ files, then attach them with `dataset.py attach --model Qwen/Qwen3-VL-8B-Instruct --tex-dir work/qwen3vl8b/tikz --image-dir work/qwen3vl8b/images --output runs/qwen3vl8b/metadata.jsonl`. Set matching `edited_model_type` and `metadata_path` in a new image-evaluation config.

## OpenRouter

```bash
python edit.py openrouter \
  --mode visual_instruction_w_org_image_w_org_tikz \
  --jsonl-path data/metadata/edit_metadata.jsonl \
  --model qwen/qwen3-vl-235b-a22b-instruct \
  --save-dir work/qwen3vl235b/tikz --max-tokens 128000
```

The retained client passes `enable_thinking=true` only when `--enable-thinking` is selected. When omitted, it does not explicitly disable provider-side reasoning. Model selection and provider routing also affect reasoning behavior. To reproduce later Qwen3.5 non-thinking experiments, use an explicitly recorded provider-specific setting rather than equating omission of this flag with non-thinking. Preserve provider responses and record the actual route/settings.

## Qwen image editing with Diffusers

Install the optional dependencies into a compatible GPU environment:

```bash
python -m pip install -e '.[diffusers]'
python edit.py diffusers --mode visual \
  --metadata-path data/metadata/edit_metadata.jsonl \
  --model Qwen/Qwen-Image-Edit-2511 --device cuda:0 --seed 0 \
  --save-dir work/qwen_image_edit/images --skip-existing
python dataset.py attach --model Qwen/Qwen-Image-Edit-2511 \
  --image-dir work/qwen_image_edit/images \
  --output runs/qwen_image_edit/metadata.jsonl
```

This client produces PNGs directly and needs no TikZ rendering step. Use image evaluation with `judge_input_type = "w_org_image"`; code-level evaluation does not apply.

## Alternative evaluation clients

`python eval.py image-vllm --help` and `python eval.py image-openrouter --help` expose the submission clients for alternative judges. The main README uses the Batch API judge and its flat IA/CP/DR CSV. Other response formats must be converted to that flat schema before using `results.py`; never merge judge failures into model failures.
