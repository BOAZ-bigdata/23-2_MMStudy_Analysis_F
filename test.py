import torch
import torch.nn as nn
import torch.optim as optim

def test_model(model,input):
    predict = model(input).data.max(1, keepdim=True)[1]
    return predict[0][0]
