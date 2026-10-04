# Commands

Run from the repo root, inside tmux (`tmux new -s reg`).

## All four arms

```bash
systemd-inhibit --what=sleep:idle --why="overfitting study" bash -c 'bash experiments/02_overfitting/run.sh freeze_fc && bash experiments/02_overfitting/run.sh early_stop && bash experiments/02_overfitting/run.sh member_loss && bash experiments/02_overfitting/run.sh all'
```

Rerunning resumes: finished arms and seeds are skipped.

## Aim

```bash
uv run aim up
```

Filter: `run.experiment == "reg_all"` (or `reg_freeze_fc`, `reg_early_stop`, `reg_member_loss`)
