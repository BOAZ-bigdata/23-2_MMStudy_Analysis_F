import torch.nn as nn

def set_parameter_requires_grad(model: nn.Module, feature_extracting: bool = True):
    if feature_extracting:
        for param in model.parameters():
            param.requires_grad = False
