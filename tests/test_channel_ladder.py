"""
The channel ladder study (studies/channel_ladder): its architectural choices
are parts the testbed has, and its levels differ only where their GELUs are.

  uv run pytest tests/test_channel_ladder.py
"""

from studies.channel_ladder import GATE, LEVELS, MIXERS, WIDTH, at_level
from testbed.communication import DECODERS, ENCODERS, build_channel
from testbed.communication import MIXERS as TESTBED_MIXERS
from testbed.moe import GATES


def test_every_choice_is_a_part_the_testbed_has():
  for level in LEVELS.values():
    assert level.encoder in ENCODERS and level.decoder in DECODERS
  assert set(MIXERS) <= set(TESTBED_MIXERS) and MIXERS[0] == "attention"
  assert GATE in GATES


def test_every_level_has_the_same_weights_at_the_studys_width():
  """Levels differ only in where their GELUs are."""
  shapes = set()
  for name in LEVELS:
    config = at_level(name)
    assert config.width == WIDTH
    channel = build_channel(config, (32, 7, 7), (16, 14, 14), num_experts=2)
    shapes.add(tuple((k, v.shape) for k, v in channel.state_dict().items()))
  assert len(shapes) == 1
  assert at_level("1", width=32).width == 32  # the width can still be ablated
