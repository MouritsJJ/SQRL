import torch

class Max(torch.nn.Module):
    def __init__(self, agg):
        super(Max, self).__init__()
    
    def forward(self, x):
        return torch.amax(x, axis=3, keepdim=True)
