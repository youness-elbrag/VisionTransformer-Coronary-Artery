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
        
def initializer() -> Paramaters:
    Config: Paramaters = {
          "image_size":256 ,
          "patch_size":16,
          "embedding_size":256,
          "num_Channle":1,
          "hidden_dropout_prob": 0.0,
          "qkv_bias": True,
          "attention_droput":0.0,
          "eps": 1e-12 , 
          "mlp_ratio": 4,
          "initializer_range":0.0,
          "num_classes":0.03
}
    return Config
            
