# Commands

Run from the repo root.

## 1. Start tmux

```bash
tmux new -s mask
```

## 2. Run all 11 arms (about 5 to 8 h)

```bash
systemd-inhibit --what=sleep:idle --why="attention masking" bash -c 'for a in mlp_all mlp_others mlp_self film_all film_others film_self mlp_all_unfreeze mlp_others_unfreeze film_all_unfreeze film_others_unfreeze k0_unfreeze; do bash experiments/03_attention_masking/run.sh $a || exit 1; done'
```

Detach: `Ctrl-b` then `d`. Reattach:

```bash
tmux attach -t mask
```

## If it gets interrupted

Run the step 2 command again. Finished arms and seeds are skipped.

## Aim

```bash
uv run aim up
```

Filter: `run.experiment.startswith("mask_")`
