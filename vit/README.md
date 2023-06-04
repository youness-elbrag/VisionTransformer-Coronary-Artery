## 1. Implementations

This is a simplified PyTorch implementation of the paper An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale. The goal of this project is to provide a simple and easy-to-understand implementation. The code is not optimized for speed and is not intended to be used for production.

Check out this post for step-by-step guide on implementing ViT in detail.

### 1.1 Patch Embedding

<div align="center">
    <img src="/assets/__results___28_0.png" width="500" height="550"/>
</div>

<br><br>

```python
class Patch_Embedding(nn.Module):
    def __init__(self , Config):
        super(Patch_Embedding,self).__init__()
        self.image_size = Config["image_size"]
        self.patch_size = Config["patch_size"]
        self.num_channel = Config["num_Channle"]
        self.number_patch = ( self.image_size // self.patch_size) ** 2
        self.hidden_size = Config["embedding_size"]
        self.Projection = nn.Conv2d(self.num_channel , self.hidden_size , 
                                    kernel_size=self.patch_size , 
                                    stride=self.patch_size)
    def forward(self, x ):
        # (batch_size, num_channels, image_size, image_size) -> (batch_size, num_patches, hidden_size)
        x= self.Projection(x)
        x = x.flatten(2).transpose(1,2)
        return x 
```
<br><br>


### 1.1 Positional Embedding


<div align="center">
    <img src="/assets/__results___46_0.png" />
</div>
<br><br>

```python
class Embeddings(nn.Module):
    """
    Combine the patch embeddings with the class token and position embeddings.
    """
        
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.patch_embeddings = Patch_Embedding(Config)
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
```
<br><br>

### 1.2 Multi-Head Attention

<div align="center">
    <img src="/assets/heads.gif" />
</div>

<br><br>

```python
class MultiHeadAttention(nn.Module):

    def __init__(self,config):
        super(MultiHeadAttention, self).__init__()
        self.n_head = Config["num_heads"]
        self.attention = ScaleDotProductAttention(config)
        self.w_q = nn.Linear(config["embedding_size"], config["embedding_size"],bias = config["qkv_bias"])
        self.w_k = nn.Linear(config["embedding_size"], config["embedding_size"],bias = config["qkv_bias"])
        self.w_v = nn.Linear(config["embedding_size"], config["embedding_size"],bias = config["qkv_bias"])
        self.w_concat = nn.Linear(config["embedding_size"], config["embedding_size"])

    def forward(self, q, k, v,output_attentions=False):
        # 1. dot product with weight matrices
        q, k, v = self.w_q(q), self.w_k(k), self.w_v(v)

        q, k, v = self.split(q), self.split(k), self.split(v)

        # 3. do scale dot product to compute similarity
        out, attention = self.attention(q, k, v,output_attentions=output_attentions)

        # 4. concat and pass to linear layer
        out = self.concat(out)
        out = self.w_concat(out)

        # 5. visualize attention map
        # TODO : we should implement visualization
        if not output_attentions:
            return (out, None)

        return ( out , attention )

    def split(self, tensor):
        """
        split tensor by number of head

        :param tensor: [batch_size, length, d_model]
        :return: [batch_size, head, length, d_tensor]
        """
        batch_size, length, d_model = tensor.size()

        d_tensor = d_model // self.n_head
        tensor = tensor.view(batch_size, length, self.n_head, d_tensor).transpose(1, 2)
        # it is similar with group convolution (split by number of heads)

        return tensor

    def concat(self, tensor):
        """
        inverse function of self.split(tensor : torch.Tensor)

        :param tensor: [batch_size, head, length, d_tensor]
        :return: [batch_size, length, d_model]
        """
        batch_size, head, length, d_tensor = tensor.size()
        d_model = head * d_tensor

        tensor = tensor.transpose(1, 2).contiguous().view(batch_size, length, d_model)
        return tensor
```
<br><br>

### 1.3 Scale Dot Product Attention

<div align="center">
    <img src="/assets/mha_img_original.png"  width="500" height="650"/>
</div>

<br><br>

```python
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

    def forward(self, q, k, v,output_attentions=False):
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
        if not output_attentions:
            return (v, None)

        return (v, score)
```
<br><br>

### 1.4 Layer Norm

<div align="center">
    <img src="/assets/layer_norm.jpg"/>
</div>   

<br><br>

```python
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

```
<br><br>

### 1.5 MLP

    
```python

class MLP(nn.Module):
    """
    A multi-layer perceptron module.
    """

    def __init__(self, config):
        super().__init__()
        self.dense_1 = nn.Linear(config["embedding_size"],config["mlp_ratio"] * config["embedding_size"])
        self.activation = nn.GELU()
        self.dense_2 = nn.Linear(config["embedding_size"] * config["mlp_ratio"], config["embedding_size"])
        self.dropout = nn.Dropout(config["hidden_dropout_prob"])

    def forward(self, x):
        x = self.dense_1(x)
        x = self.activation(x)
        x = self.dense_2(x)
        x = self.dropout(x)
        return x

```
<br><br>

### 1.5 Block Multi-Heads 
```python
class Block(nn.Module):
    """
    A single transformer block.
    """

    def __init__(self, config):
        super().__init__()
        self.attention = MultiHeadAttention(config)
        self.layernorm_1 = LayerNorm(config)
        self.mlp = MLP(config)
        self.layernorm_2 = LayerNorm(config)

    def forward(self, x, output_attentions=False):
        # Self-attention
        attention_output, attention_probs = \
            self.attention(q=x, k=x, v=x, output_attentions=output_attentions)
        attention_output = self.layernorm_1(attention_output)
        # Skip connection
        x = x + attention_output
        # Feed-forward network
        mlp_output = self.mlp(self.layernorm_2(x))
        # Skip connection
        x = x + mlp_output
        # Return the transformer block's output and the attention probabilities (optional)
        if not output_attentions:
            return (x, None)
        else:
            return (x, attention_probs)
```
<br><br>

### 1.6 TransformerEncoder
    
```python
class TrasnfomerEncoder(nn.Module):
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
```

<br><br>

### Vision Transfomer model 

<div align="center">
    <img src="/assets/vit.gif"/ width="600" height="350">
</div>

<br><br>

```python 

class VisionTransfomer(nn.Module):
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
        self.encoder = TrasnfomerEncoder(config)
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


```