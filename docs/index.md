# chatty-networks

Docs for communicating ensembles of frozen experts.

**Testbed (v4)**, the current pipeline in `testbed/`:

- [Experts](testbed/experts.md): the residual stream every channel reads and
  writes, the ResNet and ViT families, and their recipes.
- [Channel](testbed/channel.md): the communication channel's parts, its two
  controls, and their sizes.
- [Soft-gated MoE](testbed/moe.md): the baseline that weighs the experts'
  answers instead of letting them communicate.
- [Grafting](testbed/grafting.md): one expert, the anchor, alone sees the
  image and answers; the obligates take part only through the channel.
- [Metrics](testbed/metrics.md): what every logged metric means, how it is
  computed and how to read it.
- [Docket](docket.md): design decisions, and what is still open.

**Studies**, each with its own code in `studies/`:

- [Channel ladder](studies/channel-ladder.md): how much each step up in a
  channel's complexity adds, one GELU at a time.
- [Model grafting](studies/model-grafting.md): whether frozen experts help one
  another when communication is the only way they can.

The v2 pipeline (`backbone/`, `ensemble/`, `selection/`, `splits/`, `scripts/`)
and its Aim metric glossary are on `main`.

Serve locally with `uv run mkdocs serve`.
