"""
Untrained backbones: does training the channel lift an ensemble of experts
that were never trained? The channel ladder's levels 1 and 3, with every
mixer, on trained experts and on the same experts at initialisation
(--expert-epochs 0), paired by trial seed. The runner is run.py.
"""
