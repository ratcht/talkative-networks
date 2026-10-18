# Commands

Run from the repo root on the lab machine.

## 1. Get the branch

```bash
git fetch && git checkout comm-experiments && git pull
```

## 2. Start all 36 arms in a screen, logging to run.log

```bash
CUDA_VISIBLE_DEVICES=1 screen -S mexch -L -Logfile experiments/05_minimal_exchange/run.log uv run python experiments/05_minimal_exchange/run.py
```

This opens the screen attached. Change `CUDA_VISIBLE_DEVICES` to pick the GPU. Run only some arms by listing them at the end, e.g. `... run.py gate_p32_mlp_add gate_p32_mlp_replace`.

To split the sweep over both GPUs, start two screens with different names and arm lists:

```bash
CUDA_VISIBLE_DEVICES=0 screen -S mexch0 -L -Logfile experiments/05_minimal_exchange/run0.log uv run python experiments/05_minimal_exchange/run.py <arms...>
CUDA_VISIBLE_DEVICES=1 screen -S mexch1 -L -Logfile experiments/05_minimal_exchange/run1.log uv run python experiments/05_minimal_exchange/run.py <arms...>
```

## 3. Screen

| Action | Keys / command |
|---|---|
| Detach, leave it running | `Ctrl-a` then `d` |
| Reattach | `screen -r mexch` |
| List sessions | `screen -ls` |
| Stop the sweep | `Ctrl-c` while attached, or `screen -S mexch -X quit` |

`Ctrl-c` inside screen stops the sweep. To watch without attaching:

```bash
tail -f experiments/05_minimal_exchange/run.log
```

The screen closes by itself when the sweep finishes.

## If it gets interrupted

Run the step 2 command again. Finished arms and seeds are skipped. A seed that was running restarts from scratch.

## Aim

```bash
uv run aim up --port 43801
```

Port 43800 is taken by another user's Aim server. From your laptop: `ssh -N -L 43801:localhost:43801 lab`, then open http://localhost:43801.

Filter: `run.experiment.startswith("me_")`
