from typing_extensions import TypedDict

class Paramaters(TypedDict):
    image_size : int
    patch_size : int
    embedding_size : int
    num_Channle : int
    hidden_dropout_prob : float
    attention_droput: float
    qkv_bias : bool
    eps: float
    mlp_ratio : int
    initializer_range: float    
    num_classes : int
    weight_decay : float
    lr: float
    batch_size: int
    Epoch: int
    Data_Process: str
    num_workers: int
        
def initializer() -> Paramaters:
    config: Paramaters = {
          "image_size":256 ,
          "patch_size":16,
          "embedding_size":256,
          "num_Channle":1,
          "hidden_dropout_prob": 0.0,
          "qkv_bias": True , 
          "num_heads" : 8 ,
          "attention_droput":0.0,
          "eps": 1e-12,
          "mlp_ratio":4,
          "num_hidden_layers":4,
          "num_classes": 2,
          "initializer_range":0.02,
          "weight_decay": 1e-2,
          "lr":1e-2,
          "batch_size": 1,
          "Epoch":400,
          "Data_Process": "./ProcessedFolder/",
          "num_workers":2


}
    return config
