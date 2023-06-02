import torch.nn as nn

class LayerNorm(nn.Module):
    def __init__(self, config):
        super(LayerNorm, self).__init__()
        self.gamma = nn.Parameter(torch.ones(config["embedding_size"]))
        self.beta = nn.Parameter(torch.zeros(config["embedding_size"]))
        self.eps = config["eps"]

    def forward(self, x):
        mean = x.mean(-1, keepdim=True)
        var = x.var(-1, unbiased=False, keepdim=True)
        # '-1' means last dimension. 

        out = (x - mean) / torch.sqrt(var + self.eps)
        out = self.gamma * out + self.beta
        return out
