# Commands

Run from the repo root on the lab machine.

## 1. Get the branch

```bash
git fetch && git checkout comm-experiments && git pull
```

## 2. Start all 18 arms in a detached screen, logging to run.log

```bash
screen -dmS pattn -L -Logfile experiments/04_patch_attention/run.log uv run python experiments/04_patch_attention/run.py
```

Pick a GPU by putting `CUDA_VISIBLE_DEVICES=1` in front of `screen`. Run only some arms by listing them at the end, e.g. `... run.py p2_mlp_add p2_mlp_replace`.

## 3. Watch it

```bash
tail -f experiments/04_patch_attention/run.log
```

Or attach with `screen -r pattn` (detach: `Ctrl-a` then `d`).

## If it gets interrupted

Run the step 2 command again. Finished arms and seeds are skipped.

## Aim

```bash
uv run aim up
```

Filter: `run.experiment.startswith("pa_")`
