# Experiments

One folder per experiment, numbered in the order they were run.

| # | Experiment | Status |
|---|---|---|
| 00 | [patch2_additive](00_patch2_additive/README.md) | done (3 of 4 seeds) |
| 01 | [multiplicative_injection](01_multiplicative_injection/README.md) | done (film_p2: 3 of 4 seeds) |
| 02 | [overfitting](02_overfitting/README.md) | done |
| 03 | [attention_masking](03_attention_masking/README.md) | not run |

## Layout

```
NN_name/
  README.md      question, setup, arms, results
  commands.md    exact commands to run or resume
  run.sh         arm launcher, if the experiment has one
  checkpoints/   <arm>/seed<seed>.pt
  state/         <arm>.json, replicate_joint.py resume state
```

## Conventions

- Each arm is one Aim experiment. The arm name is passed as `--experiment`.
- Pass `--state-file experiments/NN_name/state/<arm>.json` to `replicate_joint.py` and `--output-dir experiments/NN_name/checkpoints` to `ensemble/train.py`, so all files land in the experiment folder.
- Compare on test accuracy. Val is used for early stopping and for expert group selection.
- Untrained probability averaging on `encoder_study_group`: val 0.6686, test 0.6652.
