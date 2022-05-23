import torch

'''
    A class for aggregating over each row using max-aggregation
    The forward method assumes a pytorch structure of the batch (BatchSize, Channels, Height, Width)
'''
class Max(torch.nn.Module):
    def __init__(self):
        super(Max, self).__init__()
    
    def forward(self, x):
        return torch.amax(x, axis=3, keepdim=True)
