# Automatic-judge stability

This experiment evaluates the **same 50 generated diagram outputs five times**. It does not regenerate diagrams between repetitions. Public benchmark inputs and the generation workflow are sufficient; a private historical selection file is not required.

## 1. Generate and render benchmark outputs

Reuse a completed main experiment with at least 50 rendered outputs. If one is not available, prepare a new run, execute generation with the selected backend, and render it using the main workflow in the README. Record the model, dataset commit, generation settings, and compiler environment.

## 2. Freeze 50 outputs and prepare five judge jobs

```bash
python paper.py stability-prepare \
  --generation-job runs/main/MODEL__direct__main__full/job.json \
  --seed 42 --output runs/stability
```

The command checks that generation and rendering cover the complete run, selects 50 rendered outputs using a fixed seed and sorted IDs, copies images byte-for-byte, and saves `selection.jsonl`. The selection, seed, source run, and image hashes are recorded. API failures must be resolved before sampling. At least 50 rendered outputs are required.

To reuse an already recorded selection, `--selection` can be used instead of `--generation-job`:

```json
{"question_id":"evaluation_000","source_key":"00023142_REMOVE_1_edit_0","generated_model":"gemini-3.1-pro-preview","edited_image_path":"outputs/evaluation_000.png"}
```

```bash
python paper.py stability-prepare \
  --selection /path/to/selection.jsonl --output runs/stability
```

Supply exactly 50 unique question IDs. Relative image paths resolve against the selection file's directory. `source_key` identifies a prepared benchmark sample.

## 3. Evaluate the fixed images five times

Five judge jobs are created under `runs/stability/judges/gemini-3.1-pro-preview/image/repeat_00/` through `repeat_04/`. Run the following commands for each job:

```bash
python paper.py submit --job runs/stability/judges/gemini-3.1-pro-preview/image/repeat_00/job.json
python paper.py collect --job runs/stability/judges/gemini-3.1-pro-preview/image/repeat_00/job.json
```

Repeat for `repeat_01` through `repeat_04`. `collect` can be rerun while a batch is still processing. Resolve failed judgments with the documented retry workflow.

## 4. Report the 250 judgments

```bash
python paper.py stability-report --run runs/stability
```

Reporting requires all 50 outputs × 5 repetitions. It saves `stability_scores.jsonl` and `table11_stability.csv`, reporting each metric's mean score and mean within-output sample standard deviation (`ddof=1`). The generated diagrams and judge settings are fixed across repetitions.

Newly sampled outputs are a rerun of the experimental procedure. They are not claimed to be the paper's historical 50 outputs, and no exact match to the historical Table 11 numbers is implied.
