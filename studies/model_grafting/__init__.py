"""
Model grafting (docs/studies/model-grafting.md): the anchor, the group's
first expert, alone sees the image and answers; the obligates take part only
through the graft's channel (testbed/grafting.py). This module holds the
study's arms, as flags for testbed.run:

  arm              mixer       the obligates' blocks
  graft_attention  attention   run between the two points
  graft_mean       mean        run between the two points
  bypass           attention   left out: the same channel, without them

The anchor alone is every run's step 0, so it has no arm. graft_attention
minus bypass is what the obligates' frozen blocks add; graft_attention minus
graft_mean is what choosing whom and where to listen adds.
"""

STUDY = "model-grafting"
EXPERIMENT = "model-grafting"

ARMS: dict[str, tuple[str, ...]] = {
  "graft_attention": ("--graft", "--graft-mixer", "attention"),
  "graft_mean": ("--graft", "--graft-mixer", "mean"),
  "bypass": ("--graft", "--graft-mixer", "attention", "--graft-bypass"),
}
# set by the arm, so a study run may not give them
ARM_FLAGS = ("--graft", "--graft-mixer", "--graft-bypass", "--arm")
