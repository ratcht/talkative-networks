# Commands

Run from the repo root, inside tmux (`tmux new -s film`).

## film_p0 then film_p2

```bash
systemd-inhibit --what=sleep:idle --why="film runs" bash -c 'uv run python scripts/replicate_joint.py --experiment film_p0 --state-file experiments/01_multiplicative_injection/state/film_p0.json -- communicative --manifest selection/manifests/encoder_study_group.json --decoder mlp --inject film --epochs 70 --k-rounds 1 --output-dir experiments/01_multiplicative_injection/checkpoints && uv run python scripts/replicate_joint.py --experiment film_p2 --state-file experiments/01_multiplicative_injection/state/film_p2.json -- communicative --manifest selection/manifests/encoder_study_group.json --decoder mlp --inject film --patch 2 --epochs 70 --k-rounds 1 --output-dir experiments/01_multiplicative_injection/checkpoints'
```

Rerunning resumes: finished seeds are skipped.

## Aim

```bash
uv run aim up
```

Filter: `run.experiment == "film_p0"`
