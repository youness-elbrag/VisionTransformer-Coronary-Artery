import torch.nn as nn
import torch 
from vit.TransformersEncoder import TransformersEncoder 

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


class VisionTransformer(nn.Module):
    """
    The ViT model for classification.
    """

    def __init__(self, config):
        super().__init__()
        self.config = config
        self.image_size = config["image_size"]
        self.hidden_size = config["embedding_size"]
        self.num_classes = config["num_classes"]
        # Create the embedding module
        self.embedding = Embeddings(config)
        # Create the transformer encoder module
        self.encoder = TransformersEncoder(config)
        # Create a linear layer to project the encoder's output to the number of classes
        self.classifier = nn.Linear(self.hidden_size, self.num_classes)
        # Initialize the weights
        self.apply(self._init_weights)

    def forward(self, x, output_attentions=False):
        # Calculate the embedding output
        embedding_output = self.embedding(x)
        # Calculate the encoder's output
        encoder_output, all_attentions = self.encoder(embedding_output, output_attentions=output_attentions)
        # Calculate the logits, take the [CLS] token's output as features for classification
        logits = self.classifier(encoder_output[:, 0])
        # Return the logits and the attention probabilities (optional)
        if not output_attentions:
            return (logits, None)
        else:
            return (logits, all_attentions)
    
    def _init_weights(self, module):
        if isinstance(module, (nn.Linear, nn.Conv2d)):
            torch.nn.init.normal_(module.weight, mean=0.0, std=self.config["initializer_range"])
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)
        elif isinstance(module, Embeddings):
            module.position_embeddings.data = nn.init.trunc_normal_(
                module.position_embeddings.data.to(torch.float32),
                mean=0.0,
                std=self.config["initializer_range"],
            ).to(module.position_embeddings.dtype)

            module.cls_token.data = nn.init.trunc_normal_(
                module.cls_token.data.to(torch.float32),
                mean=0.0,
                std=self.config["initializer_range"],
            ).to(module.cls_token.dtype)
