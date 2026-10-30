# Automatic-judge stability

The paper evaluates the same 50 generated outputs five times. It does not regenerate diagrams five times. The original selection/repeated scores have not yet been found locally; no replacement sample or Table 11 values are invented.

Provide the original selection as JSONL with exactly 50 unique question IDs:

```json
{"question_id":"main_v2_000","source_key":"00023142_REMOVE_1_edit_0","generated_model":"gemini-3.1-pro-preview","edited_image_path":"outputs/main_v2_000.png"}
```

Relative image paths resolve against the selection file's directory. `source_key` identifies a prepared benchmark sample. Images must be the original human-evaluation outputs.

```bash
python paper.py stability-prepare --selection /path/to/original_50_selection.jsonl --output runs/stability
```

This copies the images unchanged and prepares five judge jobs under `runs/stability/judges/gemini-3.1-pro-preview/image/repeat_00/` through `repeat_04/`. Submit/collect each with `paper.py submit` / `paper.py collect`. After all 250 judgments are complete:

```bash
python paper.py stability-report --run runs/stability
```

The result contains the mean score and mean sample standard deviation over 50 outputs. Its score file can be added to HF as `human/stability_scores.jsonl`. The historical standard-deviation convention must still be compared against the original implementation once available.
