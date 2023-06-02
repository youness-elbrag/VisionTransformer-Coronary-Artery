from typing import TypeDict

class Paramaters(TypeDict):
    image_size : int
    patch_size : int
    embedding_size : int
    num_Channle : int
    hidden_dropout_prob : float
    attention_droput: float
    qkv_bias : bool
    eps: float



def Intialize() -> Paramaters:
    Config: Paramaters = { "image_size":256 ,
          "patch_size":16,
          "embedding_size":256,
          "num_Channle":1,
          "hidden_dropout_prob": 0.0,
          "qkv_bias": True,
          "attention_droput":0.0
        "eps": 1e-12
}
    return Config


            
