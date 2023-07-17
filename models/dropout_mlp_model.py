import torch
import torch.nn.functional as F

class DropoutModel(torch.nn.Module):
    def __init__(self):
        super(DropoutModel,self).__init__()
        self.layer1 = torch.nn.Linear(784,1200)
        self.dropout1 = torch.nn.Dropout(0.5)
        self.layer2 = torch.nn.Linear(1200,1200)
        self.dropout2 = torch.nn.Dropout(0.5)
        self.layer3 = torch.nn.Linear(1200,10)
    def forward(self,x):
        x = F.relu(self.layer1(x))
        x = self.dropout1(x)
        x = F.relu(self.layer2(x))
        x = self.dropout2(x)
        return self.layer3(x)