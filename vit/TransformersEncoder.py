import torch.nn as nn 
import torch
import numpy as np 
import Config 

"""
Doc : this file contain building blocks of ViT model 
----------------------------------------------------

Patch_Embedding : split image into sub-Patch image 16x16

Positional Emebedding : add Positional Encoding to Embedded input 

Attention Module : is refer to Attention all you need Paper 

Multi-Heads-Attention : make model learn Paralle 

MLP : Mulit-Layer preceptence for Classfication 

NormLayer : is used before MLP to normalize and stablize the model Transformer

"""

class Patch_Embedding(nn.Module):
    def __init__(self , config):
        super(Patch_Embedding,self).__init__()
        self.image_size = config["image_size"]
        self.patch_size = config["patch_size"]
        self.num_channel = config["num_Channle"]
        self.number_patch = ( self.image_size // self.patch_size) ** 2
        self.hidden_size = config["embedding_size"]
        self.Projection = nn.Conv2d(self.num_channel , self.hidden_size , 
                                    kernel_size=self.patch_size , 
                                    stride=self.patch_size)
    def forward(self, x ):
        # (batch_size, num_channels, image_size, image_size) -> (batch_size, num_patches, hidden_size)
        x= self.Projection(x)
        x = x.flatten(2).transpose(1,2)
        return x


class Embeddings(nn.Module):
    """
    Combine the patch embeddings with the class token and position embeddings.
    """
        
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.patch_embeddings = Patch_Embedding(config)
        self.cls_token = nn.Parameter(torch.randn(1, 1, config["embedding_size"]))
        self.position_embeddings = \
            nn.Parameter(torch.randn(1, self.patch_embeddings.number_patch + 1, config["embedding_size"]))
        self.dropout = nn.Dropout(config["hidden_dropout_prob"])
    def forward(self, x):
        x = self.patch_embeddings(x)
        batch_size, _, _ = x.size()
        
        # Expand the [CLS] token to the batch size
        # (1, 1, hidden_size) -> (batch_size, 1, hidden_size)
        cls_tokens = self.cls_token.expand(batch_size, -1, -1)
        # Concatenate the [CLS] token to the beginning of the input sequence
        # This results in a sequence length of (num_patches + 1)
        x = torch.cat((cls_tokens, x), dim=1)
        x = x + self.position_embeddings
        x = self.dropout(x)
        return x

class TransfomerEncoder(nn.Module):
    """
    The transformer encoder module.
    """

    def __init__(self, config):
        super().__init__()
        # Create a list of transformer blocks
        self.blocks = nn.ModuleList([])
        for _ in range(config["num_hidden_layers"]):
            block = Block(config)
            self.blocks.append(block)

    def forward(self, x, output_attentions=False):
        # Calculate the transformer block's output for each block
        all_attentions = []
        for block in self.blocks:
            x, attention_probs = block(x, output_attentions=output_attentions)
            if output_attentions:
                all_attentions.append(attention_probs)
        # Return the encoder's output and the attention probabilities (optional)
        if not output_attentions:
            return (x, None)
        else:
            return (x, all_attentions)


