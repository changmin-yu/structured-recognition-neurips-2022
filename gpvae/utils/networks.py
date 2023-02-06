import torch.nn as nn
import torch.nn.functional as F

class MLP(nn.Module):
    def __init__(self, in_dim, out_dim, hidden_dims=(64, 64), nonlinearity=nn.ReLU):
        super().__init__()
        
        self.nonlinearity = nonlinearity
        
        in_dims = [in_dim] + hidden_dims[:-1]
        out_dims = hidden_dims[1:] + [out_dim]
        
        self.network = nn.ModuleList()
        i = 0
        for (dim1, dim2) in zip(in_dims, out_dims):
            if i < (len(in_dims)-1):
                self.network.extend([nn.Linear(dim1, dim2), nonlinearity()])
            else:
                self.network.append(nn.Linear(dim1, dim2))
        
        self.init_params()
    
    def init_params(self):
        for layer in self.network:
            if isinstance(layer, nn.Linear):
                layer.weight.data.normal_(std=0.01)
                if layer.bias.data is not None:
                    layer.bias.data.fill_(0.0)
    
    def forward(self, x):
        return self.network(x)