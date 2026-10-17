import torch.nn as nn


class Driver(nn.Module):  
  def __init__(self, backbone: nn.Module, early: str, late: str):
    super().__init__()
    self.backbone = backbone
    self.early = backbone.get_submodule(early)
    self.late = backbone.get_submodule(late)
    # the whole specialist is frozen, classification head included: the encoder
    # and decoder are then the only parameters the loss can reach, so any
    # reduction in it is attributable to communication rather than to a readout
    # refitting itself to whatever distribution the injection produced
    for p in self.backbone.parameters():
      p.requires_grad_(False)
