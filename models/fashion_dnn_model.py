import torch
import torch.nn.functional as F


class FashionDNN(torch.nn.Module):
    def __init__(self):
        super(FashionDNN,self).__init__()
        self.fc1 = torch.nn.Linear(in_features=784,out_features=256)
        self.drop = torch.nn.Dropout(0.25)
        self.fc2 = torch.nn.Linear(in_features=256,out_features=128)
        self.fc3 = torch.nn.Linear(in_features=128,out_features=10)

    def forward(self,input_data):
        out = input_data.view(-1,784)
        out = F.relu(self.fc1(out))
        out = self.drop(out)
        out = F.relu(self.fc2(out))
        out = self.fc3(out)
        return out
