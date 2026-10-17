"""
Is the communication channel routing to the *correct* specialist, or just
routing sharply?

Attention entropy says how concentrated a receiver's attention is, not whether
it concentrated on the right sender. A sharp gate onto a wrong specialist is
worse than a soft gate onto a right one, and the two are indistinguishable by
entropy alone. This script asks, per validation example and per receiving
specialist: the specialist you attended to most — was it actually right?

Reported against two references:

  1. Random routing. Attending uniformly at random would hit a correct
     specialist at the mean of the specialists' own accuracies. Anything at
     that level is routing that carries no information about correctness.

  2. Split by what communication actually did (corrected / corrupted, same
     definitions as answer_shift_stats in ensemble/train.py). This is the
     sharper test: aggregate routing quality can look fine while being
     unrelated to the examples where communication helped or hurt. If routing
     is causally doing the work, the examples it corrected should be ones it
     routed well on, and the ones it corrupted should be ones it routed badly
     on.

Index alignment is the thing that would silently produce garbage here rather
than an error, so it is established by construction and then checked:
per-expert correctness comes from stage 3's prediction cache, and this script
builds its loader with selection/cache.py's own make_val_loader — the exact
function that produced those arrays — so row r of the cache is the r-th example
the loader yields. --verify-alignment additionally re-runs the frozen
backbones and asserts their predictions reproduce the cache.

Usage
-----
uv run python scripts/analyze_routing.py \\
    --checkpoints ensemble/checkpoints/communicative_*.pt
"""

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch

_REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_REPO_ROOT))

import selection.cache as C
from ensemble.train import resolve_specialist_specs, build_communicative, routing_counts


# ---------------------------------------------------------------------------
# cached per-expert ground truth (stage 3) — never recomputed here
# ---------------------------------------------------------------------------

def load_expert_cache(manifest: dict, cache_root: Path) -> tuple[np.ndarray, np.ndarray]:
  """(correct, labels) — correct is (n_experts, n_val) bool, in manifest order.

  Ordering is load-bearing: the cache is keyed by expert digest, and the rows
  within each file follow split["val_indices"]. Stacking in manifest order
  makes axis 0 line up with specialist position in the Orchestrator, which is
  also built in manifest order (build_communicative iterates specs in order).
  """
  pred_dir = cache_root / "preds" / manifest["split_digest"]
  labels = np.load(pred_dir / "labels.npy")

  correct = []
  for e in manifest["experts"]:
    path = pred_dir / f"{e['digest']}.npy"
    if not path.exists():
      raise SystemExit(
        f"missing cached predictions for expert {e['index']} ({e['digest']}) at {path}\n"
        f"regenerate with: uv run python selection/select.py compute ..."
      )
    probs = np.load(path)
    correct.append(probs.argmax(1) == labels)
  return np.stack(correct), labels


# ---------------------------------------------------------------------------
# model reconstruction — architecture comes from the checkpoint, never defaults
# ---------------------------------------------------------------------------

def load_arm(ckpt_path: Path, device: str):
  """Rebuild the trained Orchestrator described by a checkpoint."""
  ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
  meta = ckpt["metadata"]

  backbone_paths = [(_REPO_ROOT / p).resolve() for p in meta["backbones"]]
  # every architectural choice is read back from the checkpoint: an arm differs
  # from another only in these, so taking any of them from a CLI default would
  # silently rebuild the wrong architecture and fail to load (or worse, load)
  defaults = argparse.Namespace(
    early=meta["early"], late=meta["late"],
    encoder=meta["encoder"], decoder=meta["decoder"],
  )
  specs = resolve_specialist_specs(backbone_paths, defaults, None)
  model = build_communicative(
    specs, meta["key_dim"], meta["value_dim"],
    num_classes=ckpt["state_dict"]["specialists.0.driver.backbone.fc.weight"].shape[0],
    device=device,
  )
  model.load_state_dict(ckpt["state_dict"])
  model.eval()
  return model, meta


# ---------------------------------------------------------------------------
# the one pass: attention weights + pre/post opinions, per val example
# ---------------------------------------------------------------------------

@torch.no_grad()
def collect(model, loader, device: str, k_rounds: int):
  """Run val once, returning per-example attention and pre/post predictions.

  attn        (n_val, n, n)  post-softmax attention, receiver-major
  pre_ans     (n, n_val)     each specialist's pre-communication argmax
  post_ans    (n, n_val)     each specialist's post-communication argmax
  """
  captured: list[torch.Tensor] = []

  def hook(_module, _inp, output):
    # TarMACBus.forward returns (attn @ V, attn)
    captured.append(output[1].detach().cpu())

  handle = model.bus.register_forward_hook(hook)
  attn_batches, pre_batches, post_batches = [], [], []
  try:
    for x, _y in loader:
      captured.clear()
      model(x.to(device), k_rounds=k_rounds)

      # k_rounds=1 means exactly one bus call, whose output feeds the single
      # subsequent backbone pass that produces the final answer. Anything else
      # makes "the attention" ambiguous, so refuse rather than pick one.
      if len(captured) != 1:
        raise SystemExit(
          f"expected exactly 1 bus call per forward at k_rounds={k_rounds}, "
          f"got {len(captured)} — this analysis assumes k_rounds=1"
        )
      shift = model.last_shift
      if shift is None:
        raise SystemExit("Orchestrator.last_shift is None — no communication round ran")

      attn_batches.append(captured[0])
      pre_batches.append(shift["pre"].argmax(-1).cpu())
      post_batches.append(shift["post"].argmax(-1).cpu())
  finally:
    handle.remove()

  return (
    torch.cat(attn_batches).numpy(),
    torch.cat(pre_batches, dim=1).numpy(),
    torch.cat(post_batches, dim=1).numpy(),
  )


def entropy_of(attn: np.ndarray) -> float:
  """Mean per-row entropy of the attention distribution, in nats."""
  return float(-(attn * np.log(attn + 1e-12)).sum(-1).mean())


# ---------------------------------------------------------------------------
# the metric
# ---------------------------------------------------------------------------

def analyze(attn, pre_ans, post_ans, correct, labels, expert_val_acc):
  """Routing-correctness rates, overall and split by corrected/corrupted.

  correct is the *cached* per-expert correctness (the frozen experts as stage 3
  scored them). pre_ans is the same specialists' opinions inside the trained
  model, whose fc heads are left trainable — so the two can legitimately
  disagree, and both notions of "was the attended-to specialist right" are
  reported rather than silently picking one.
  """
  n_val, n, _ = attn.shape
  attended = attn.argmax(-1)                      # (n_val, n) receiver -> sender
  correct_T = correct.T                           # (n_val, n) cached
  pre_correct_T = (pre_ans == labels[None, :]).T  # (n_val, n) in-model, pre-comm

  # did the specialist this receiver attended to get this example right
  rows = np.arange(n_val)[:, None]
  routed_cached = correct_T[rows, attended]        # (n_val, n) bool
  routed_inmodel = pre_correct_T[rows, attended]

  # random routing hits a correct specialist at the mean specialist accuracy;
  # computed from the arrays rather than trusted from the manifest, then
  # cross-checked against it
  baseline_cached = float(correct.mean())
  baseline_inmodel = float(pre_correct_T.mean())

  # what communication did, per (example, receiver) — answer_shift_stats' defs
  right_pre = (pre_ans == labels[None, :]).T
  right_post = (post_ans == labels[None, :]).T
  corrected = ~right_pre & right_post
  corrupted = right_pre & ~right_post

  self_attended = (attended == np.arange(n)[None, :])
  other = ~self_attended

  # Self-attention makes the corrected/corrupted split structurally misleading:
  # a corrected receiver was wrong pre-communication by definition, so if it
  # attended to itself it scores "routed wrong" no matter how routing behaves.
  # The off-diagonal subset is the one that actually asks about routing, and its
  # chance level is the mean accuracy of the *other* specialists, not of all.
  off_diag_chance = np.zeros((n_val, n), dtype=float)
  for i in range(n):
    others = [j for j in range(n) if j != i]
    off_diag_chance[:, i] = pre_correct_T[:, others].mean(1)

  def subset(mask, arr):
    return (float(arr[mask].mean()), int(mask.sum())) if mask.any() else (float("nan"), 0)

  # How many distinct receiver->sender patterns occur at all. 1 means attention
  # ignores the input entirely — a constant wiring rather than routing, which
  # no accuracy-vs-chance comparison would reveal on its own.
  patterns = np.unique(attended, axis=0)

  return {
    "n_val": n_val,
    "n_specialists": n,
    "attn_entropy": entropy_of(attn),
    "max_entropy": float(np.log(n)),
    "self_attn_frac": float(self_attended.mean()),
    "distinct_patterns": int(len(patterns)),
    "modal_pattern": patterns[0].tolist() if len(patterns) == 1 else None,
    "peak_attn_median": float(np.median(attn.max(-1))),

    "routed_correct_cached": float(routed_cached.mean()),
    "routed_correct_inmodel": float(routed_inmodel.mean()),
    "baseline_cached": baseline_cached,
    "baseline_inmodel": baseline_inmodel,
    "baseline_manifest": float(np.mean(expert_val_acc)),

    "corrected_rate": subset(corrected, routed_inmodel),
    "corrupted_wrong_rate": subset(corrupted, ~routed_inmodel),
    "corrected_n": int(corrected.sum()),
    "corrupted_n": int(corrupted.sum()),

    # same three, restricted to receivers that attended to someone else
    "routed_correct_offdiag": subset(other, routed_inmodel),
    "offdiag_chance": float(off_diag_chance[other].mean()) if other.any() else float("nan"),
    "corrected_rate_offdiag": subset(corrected & other, routed_inmodel),
    "corrupted_wrong_rate_offdiag": subset(corrupted & other, ~routed_inmodel),
    "self_attn_in_corrected": subset(corrected, self_attended),
    "self_attn_in_corrupted": subset(corrupted, self_attended),

    # agreement between the cached frozen experts and the same specialists
    # inside the trained model; < 1 is expected, since fc stays trainable
    "cache_vs_model_agreement": float((pre_correct_T == correct_T).mean()),
  }


# ---------------------------------------------------------------------------

def main() -> None:
  ap = argparse.ArgumentParser(description=__doc__,
                               formatter_class=argparse.RawDescriptionHelpFormatter)
  ap.add_argument("--checkpoints", type=Path, nargs="+", required=True,
                  help="Trained communicative checkpoints, one or more arms/seeds")
  ap.add_argument("--manifest", type=Path,
                  default=_REPO_ROOT / "selection/manifests/encoder_study_group.json")
  ap.add_argument("--split-file", type=Path,
                  default=_REPO_ROOT / "splits/three_way_seed42.pt")
  ap.add_argument("--cache-root", type=Path, default=_REPO_ROOT / "selection/cache")
  ap.add_argument("--data-root", type=str, default=str(_REPO_ROOT / "data"))
  ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
  ap.add_argument("--json-out", type=Path, default=None)
  args = ap.parse_args()

  manifest = json.loads(args.manifest.read_text())

  # the cache is keyed by split digest; if the manifest was built against a
  # different partition than --split-file, every row would be misaligned
  val_indices, split_dig = C.read_split(args.split_file)
  if split_dig != manifest["split_digest"]:
    raise SystemExit(
      f"split digest mismatch: --split-file is {split_dig}, manifest was built "
      f"against {manifest['split_digest']} — the cached predictions do not "
      f"describe these validation examples"
    )

  correct, labels = load_expert_cache(manifest, args.cache_root)
  # same function that produced the cache, so row r of the cache is the r-th
  # example this loader yields — alignment by construction, not by convention
  loader = C.make_val_loader(val_indices, args.data_root)

  expert_val_acc = [e["val_acc"] for e in manifest["experts"]]
  print(f"[routing] {len(val_indices)} val examples, {len(manifest['experts'])} experts, "
        f"split {split_dig}")
  print(f"[routing] cached expert accuracies: "
        f"{', '.join(f'{a:.4f}' for a in correct.mean(1))} "
        f"(manifest: {', '.join(f'{a:.4f}' for a in expert_val_acc)})")

  rows = []
  for ckpt_path in args.checkpoints:
    model, meta = load_arm(ckpt_path, args.device)
    if meta["k_rounds"] != 1:
      print(f"[routing] skipping {ckpt_path.name}: k_rounds={meta['k_rounds']}, not 1")
      continue

    attn, pre_ans, post_ans = collect(model, loader, args.device, meta["k_rounds"])
    if attn.shape[0] != len(labels):
      raise SystemExit(
        f"got {attn.shape[0]} rows from the loader but the cache has {len(labels)} — "
        f"alignment is broken, refusing to report"
      )

    stats = analyze(attn, pre_ans, post_ans, correct, labels, expert_val_acc)
    stats["checkpoint"] = str(ckpt_path)
    stats["arm"] = f"{meta['encoder']}/k{meta['key_dim']}/v{meta['value_dim']}"
    stats["seed"] = meta.get("seed")
    rows.append(stats)
    del model

  if not rows:
    raise SystemExit("no usable checkpoints")

  _report(rows)
  if args.json_out:
    args.json_out.parent.mkdir(parents=True, exist_ok=True)
    args.json_out.write_text(json.dumps(rows, indent=2))
    print(f"\nwrote {args.json_out}")


def _report(rows: list[dict]) -> None:
  def pct(x):
    return "  n/a " if x != x else f"{100 * x:5.1f}%"

  print(f"\n{'=' * 100}\nRouting correctness — did the attended-to specialist get it right?\n{'=' * 100}")
  hdr = f"{'arm':<18} {'seed':>11} {'entropy':>16} {'routed':>7} {'chance':>7} {'delta':>7} {'self-attn':>9}"
  print(hdr)
  print("-" * len(hdr))
  for r in rows:
    ent = f"{r['attn_entropy']:.3f}/{r['max_entropy']:.3f}"
    delta = r["routed_correct_inmodel"] - r["baseline_inmodel"]
    print(f"{r['arm']:<18} {str(r['seed']):>11} {ent:>16} "
          f"{pct(r['routed_correct_inmodel'])} {pct(r['baseline_inmodel'])} "
          f"{delta * 100:+6.1f}pp {pct(r['self_attn_frac'])}")

  print(f"\n{'=' * 100}\nExcluding self-attention — only receivers that attended to another specialist\n{'=' * 100}")
  hdr2 = f"{'arm':<18} {'seed':>11} {'routed':>7} {'chance':>7} {'delta':>8} {'pairs':>8}"
  print(hdr2)
  print("-" * len(hdr2))
  for r in rows:
    rate, npairs = r["routed_correct_offdiag"]
    delta = rate - r["offdiag_chance"]
    print(f"{r['arm']:<18} {str(r['seed']):>11} {pct(rate)} {pct(r['offdiag_chance'])} "
          f"{delta * 100:+7.1f}pp {npairs:>8}")

  print(f"\n{'=' * 100}\nSplit by what communication did on that example\n{'=' * 100}")
  hdr3 = (f"{'arm':<18} {'seed':>11} {'corrected: routed ok':>22} {'corrupted: routed wrong':>25}")
  print(hdr3)
  print("-" * len(hdr3))
  for r in rows:
    crate, cn = r["corrected_rate_offdiag"]
    prate, pn = r["corrupted_wrong_rate_offdiag"]
    print(f"{r['arm']:<18} {str(r['seed']):>11} "
          f"{pct(crate)} (n={cn:>6}) {pct(prate)} (n={pn:>6})")
  print("  (self-attending pairs excluded — a corrected receiver was wrong pre-communication")
  print("   by definition, so counting its self-attention as 'routed wrong' is circular)")

  print(f"\n{'=' * 100}\nIs it routing at all? distinct receiver->sender patterns over the val set\n{'=' * 100}")
  hdr4 = f"{'arm':<18} {'seed':>11} {'patterns':>9} {'peak attn':>10}  {'verdict'}"
  print(hdr4)
  print("-" * (len(hdr4) + 30))
  for r in rows:
    verdict = ("CONSTANT — attention ignores the input"
               if r["distinct_patterns"] == 1 else "input-dependent")
    if r["modal_pattern"] is not None:
      verdict += f" (always {r['modal_pattern']})"
    print(f"{r['arm']:<18} {str(r['seed']):>11} {r['distinct_patterns']:>9} "
          f"{r['peak_attn_median']:>10.4f}  {verdict}")

  print("\nnotes")
  print("  chance = mean accuracy of the specialists a receiver could have routed to.")
  for r in rows:
    sc, _ = r["self_attn_in_corrected"]
    su, _ = r["self_attn_in_corrupted"]
    print(f"  {r['arm']} seed {r['seed']}: self-attention {pct(r['self_attn_frac']).strip()} overall, "
          f"{pct(sc).strip()} within corrected, {pct(su).strip()} within corrupted")
    print(f"  {r['arm']} seed {r['seed']}: cached-vs-model specialist agreement "
          f"{pct(r['cache_vs_model_agreement']).strip()} "
          f"(< 100% is expected — fc heads stay trainable during joint training)")


if __name__ == "__main__":
  main()
