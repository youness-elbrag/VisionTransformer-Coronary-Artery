import math 
import torch.nn as nn 


class ScaleDotProductAttention(nn.Module):

"""
compute scale dot product attention

Query : given sentence that we focused on (decoder)
Key : every sentence to check relationship with Qeury(encoder)
Value : every sentence same with Key (encoder)
"""

def __init__(self,config):
    super(ScaleDotProductAttention, self).__init__()
    self.softmax = nn.Softmax(dim=-1)
    self.attention_dropout = nn.Dropout(config["attention_droput"])

def forward(self, q, k, v):
    # input is 4 dimension tensor
    # [batch_size, head, length, d_tensor]
    batch_size, head, length, d_tensor = k.size()

    # 1. dot product Query with Key^T to compute similarity
    k_t = k.transpose(2, 3)  # transpose
    score = (q @ k_t) / math.sqrt(d_tensor)  # scaled dot product

    # 3. pass them softmax to make [0, 1] range
    score = self.softmax(score)
    score =  self.attention_dropout(score)

    # 4. multiply with Value
    v = score @ v

    return v, score
