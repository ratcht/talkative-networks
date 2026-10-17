# Running 03 on the lab server

Run these on the lab server unless marked "on your laptop".

## 1. Check what the server has

```bash
nvidia-smi
```

```bash
which tmux screen uv git
```

Use tmux if it is installed, otherwise screen. Both work the same for this.

## 2. Install uv (only if `which uv` printed nothing, no sudo needed)

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

```bash
source $HOME/.local/bin/env
```

Check it worked:

```bash
uv --version
```

If `curl` is not installed, use `wget` instead (then run the `source` line above):

```bash
wget -qO- https://astral.sh/uv/install.sh | sh
```

If the install script cannot reach the internet, download the uv binary directly:

```bash
mkdir -p $HOME/.local/bin && wget -qO- https://github.com/astral-sh/uv/releases/latest/download/uv-x86_64-unknown-linux-gnu.tar.gz | tar xz --strip-components=1 -C $HOME/.local/bin && export PATH="$HOME/.local/bin:$PATH"
```

If conda is on the server:

```bash
conda install -c conda-forge uv
```

## 3. Get the code

If the repo is not on the server yet:

```bash
git clone -b bus-improvements https://github.com/ratcht/talkative-networks.git && cd talkative-networks
```

If it is already there (cd into it first):

```bash
git fetch origin && git switch bus-improvements && git pull
```

The expert checkpoints, split file and manifest are in git, so nothing else needs copying. CIFAR-100 downloads automatically on the first run.

## 4. Python environment

```bash
uv venv --python 3.11 && uv pip install -r requirements.txt
```

This needs no system pip. uv creates `.venv` in the repo and installs into it. If the server has no Python 3.11, uv downloads one.

Check the GPU is visible from torch:

```bash
CUDA_VISIBLE_DEVICES=1 uv run python -c "import torch; print(torch.__version__, torch.cuda.is_available(), torch.cuda.device_count(), torch.cuda.get_device_name(0))"
```

If this prints `False`, the server's driver does not support the CUDA 13 build of torch in `requirements.txt`. Compare the CUDA version shown by `nvidia-smi` and reinstall torch for that version from pytorch.org.

Create the Aim repo (only if there is no `.aim` folder in the repo root):

```bash
uv run aim init
```

## 5. Start a session

tmux:

```bash
tmux new -s mask
```

or screen:

```bash
screen -S mask
```

## 6. Pin to GPU 1 and smoke test (about 1 min)

Run this once inside the session. Everything after it in the same session only sees GPU 1 (torch calls it `cuda:0`):

```bash
export CUDA_VISIBLE_DEVICES=1
```

```bash
uv run python ensemble/train.py communicative --manifest selection/manifests/encoder_study_group.json --freeze-fc --decoder mlp --inject film --attn others --unfreeze layer1.2 layer1.3 layer1.4 --epochs 1 --no-track --experiment smoke && rm -r ensemble/checkpoints/smoke
```

Expect `trainable params=79,488`.

## 7. Run all 11 arms on GPU 1

```bash
export CUDA_VISIBLE_DEVICES=1 && for a in mlp_all mlp_others mlp_self film_all film_others film_self mlp_all_unfreeze mlp_others_unfreeze film_all_unfreeze film_others_unfreeze k0_unfreeze; do bash experiments/03_attention_masking/run.sh $a || break; done
```

## 8. Detach and reattach

tmux: detach with `Ctrl-b` then `d`. Reattach:

```bash
tmux attach -t mask
```

screen: detach with `Ctrl-a` then `d`. Reattach:

```bash
screen -r mask
```

You can log out of ssh after detaching. Training keeps running.

## 9. If it gets interrupted

Run the step 7 command again. Finished arms and seeds are skipped.

## 10. Watch in Aim from your laptop

On the server, in a second tmux window (`Ctrl-b` then `c`) or screen window (`Ctrl-a` then `c`):

```bash
uv run aim up --host 127.0.0.1 --port 43800
```

On your laptop, in a new terminal (replace with your ssh login):

```bash
ssh -N -L 43800:127.0.0.1:43800 USER@LAB_SERVER
```

Then open http://127.0.0.1:43800 on your laptop.

## 11. Bring results back (on your laptop, after it finishes)

Replace `USER@LAB_SERVER` and the repo path on the server.

```bash
rsync -av USER@LAB_SERVER:~/talkative-networks/experiments/03_attention_masking/ experiments/03_attention_masking/
```

```bash
rsync -av USER@LAB_SERVER:~/talkative-networks/.aim/ ~/lab_aim/
```

The second copies the server's Aim runs into `~/lab_aim`, separate from your local `.aim`, so nothing local is overwritten. Tell me when it is there and I will read the results from it.
