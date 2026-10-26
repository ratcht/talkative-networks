"""
One package per study, holding only what is specific to it: its arms, the
labels on its runs, its defaults and its runner. The reusable parts (data,
experts, channel parts, training, metrics, the runner of one arm) live in
testbed/, which knows no study.
"""
