# chatty-networks

Docs for communicating ensembles of frozen experts.

**Testbed (v3)**, the current pipeline in `testbed/`:

- [Channel](testbed/channel.md): the communication channel's parts, its two
  controls, and their sizes.
- [Soft-gated MoE](testbed/moe.md): the baseline that weighs the experts'
  answers instead of letting them communicate.
- [Metrics](testbed/metrics.md): what every logged metric means, how it is
  computed and how to read it.
- [Docket](docket.md): design decisions, and what is still open.

**Studies**, each with its own code in `studies/`:

- [Channel ladder](studies/channel-ladder.md): how much each step up in a
  channel's complexity adds, one GELU at a time.

The v2 pipeline (`backbone/`, `ensemble/`, `selection/`, `splits/`, `scripts/`)
and its Aim metric glossary are on `main`.

Serve locally with `uv run mkdocs serve`.
