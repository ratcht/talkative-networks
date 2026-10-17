# Commands

Run from the repo root.

## 1. Start screen

```bash
screen -S pattn
```

## 2. Run all 18 arms

```bash
systemd-inhibit --what=sleep:idle --why="patch attention" bash -c 'for a in p0_linear_add p0_mlp_add p1_linear_add_pattn p1_linear_replace_pattn p1_mlp_add_pattn p1_mlp_replace_pattn p1_linear_add p1_linear_replace p1_mlp_add p1_mlp_replace p2_linear_add_pattn p2_linear_replace_pattn p2_mlp_add_pattn p2_mlp_replace_pattn p2_linear_add p2_linear_replace p2_mlp_add p2_mlp_replace; do bash experiments/04_patch_attention/run.sh $a || exit 1; done'
```

Detach: `Ctrl-a` then `d`. Reattach:

```bash
screen -r pattn
```

## If it gets interrupted

Run the step 2 command again. Finished arms and seeds are skipped.

## Aim

```bash
uv run aim up
```

Filter: `run.experiment.startswith("pa_")`
