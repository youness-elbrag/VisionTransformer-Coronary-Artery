import torch
import torchvision
from torchvision import transforms
from torchvision.datasets import DatasetFolder
from torch.utils.data import DataLoader
import math
import torch.nn as nn 
import torchmetrics as Metric
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import TensorBoardLogger
from ViT import  VisionTransformer
from utils import read_config_from_yaml , write_config_to_yaml


## Loading the Dataset 
Transform_aug_Train = transforms.Compose([
         transforms.ToTensor(),
         transforms.RandomHorizontalFlip(p=0.5),
         transforms.Normalize(0.075,0.17)

     ])

 
Transform_aug_Val = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(0.075,0.17)
    ])

class VisionTrans(pl.LightningModule):
    def __init__(self,config, *args, **kwargs):
        
        super(VisionTrans,self).__init__()
        self.save_hyperparameters()
        self.Config = self.hparams.config
        self.model = VisionTransformer(self.Config)
        self.Loss = nn.CrossEntropyLoss()
        self.Optimizer = torch.optim.Adam(self.model.parameters(), lr=self.Config["lr"], weight_decay=self.Config["weight_decay"])
        self.Train_acc = Metric.Accuracy(task="binary", num_classes=self.Config["num_classes"])    
        self.Val_acc = Metric.Accuracy(task="binary", num_classes=self.Config["num_classes"])  
        self.Val_prec = Metric.Precision(task="binary", num_classes=self.Config["num_classes"])
        self.Val_recall = Metric.Recall(task="binary", num_classes=self.Config["num_classes"])
        
        
    def forward(self ,x ,output_attentions=False):
        return self.model(x ,output_attentions=output_attentions)

    
    def training_step(self , batch , batch_idx):
        image , label = batch
        pred = self(image)[0]
        label = label.long()
        logits = torch.argmax(pred).view(-1)
        loss = self.Loss(pred,label)
        self.log("Loss Training",loss ,sync_dist=True, on_epoch=True)
        if self.current_epoch % 5 == 0:
            self.Train_acc(logits , label)
            self.log("Acc Train:",self.Train_acc.compute() ,sync_dist=True, on_step=True , prog_bar=True)
        return {"loss": loss , "pred":logits ,"label": label}

    
    def training_step_end(self,outs):
        self.Train_acc(outs["pred"],outs["label"])
        self.log("Train Laat Epoch Step",self.Train_acc.compute(),sync_dist=True)
        self.Train_acc.reset()
    
    def validation_step(self , batch , batch_idx):
        image , label = batch
        pred = self(image)[0]
        label = label.long()
        logits = torch.argmax(pred).view(-1)
        loss = self.Loss(pred,label)
        self.log("Loss Validation",loss ,sync_dist=True, on_step=True)
        
        if self.current_epoch % 5 == 0:
            self.Val_acc(logits , label)
            self.Val_prec(logits , label)
            self.Val_recall(logits , label)
            self.log_dict({"Acc Val":self.Val_acc.compute(), 
                           "Precision Val ":self.Val_prec.compute() , 
                           "recall Val":self.Val_recall.compute()},
                      sync_dist=True, 
                      on_epoch=True,
                      prog_bar=True)
        return {"loss": loss , "pred":logits ,"label": label}
    
    def validation_step_end(self,outs):
        self.Val_acc(outs["pred"],outs["label"])
        self.Val_prec(outs["pred"],outs["label"])
        self.Val_recall(outs["pred"],outs["label"])
        self.log_dict({"Acc Val": self.Val_acc.compute(), "Precision Val ":self.Val_prec.compute() , "recall Val":self.Val_recall.compute()},
                      sync_dist=True, 
                      on_step=True,
                      prog_bar=True)

        self.Val_acc.reset()
        self.Val_prec.reset()
        self.Val_recall.reset()
        
    def configure_optimizers(self):
        return [self.Optimizer]  


def parse_args():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--Config", type=str, required=True)
    parser.add_argument("--device", type=str)

    args = parser.parse_args()
    if args.device is None:
        args.device = "gpu" if torch.cuda.is_available() else "cpu"
    return args

if __name__ == "__main__":
    args = parse_args()
    # Training parameter
    config=read_config_from_yaml(args.Config)

    # Hypyer-Parameters Training 
    device = args.device
    data_train = config["Data_Process"]
    num_workers = config["num_workers"]
    batch_size = config["batch_size"]
    Epoch = config["Epoch"]

    Loading_Train =DatasetFolder(root=data_train +"train",
                                                              loader=Loader_Data , 
                                                              extensions="npy",
                                                              transform=Transform_aug_Train)
    Loading_Val= DatasetFolder(root=data_train +"val",
                                                           loader=Loader_Data , 
                                                           extensions="npy",
                                                           transform=Transform_aug_Val)
    Training_Loader = DataLoader(Loading_Train,batch_size=batch_size,num_workers = 2, shuffle=True)
    Validation_Loader = DataLoader(Loading_Val,batch_size=batch_size,num_workers = 2, shuffle=False)
    Check_Point_Callbacks = ModelCheckpoint(
    monitor="Acc Val", 
    save_top_k=1,
    verbose=True,
    mode="max")
    ViT = VisionTrans(config)
    Trainer = pl.Trainer(accelerator=device,devices="auto",
                     logger=TensorBoardLogger(save_dir= "./processed/logs"), log_every_n_steps=1,
                     callbacks=Check_Point_Callbacks,                    
                     max_epochs=Epoch,fast_dev_run=False)
                     # Trainer = pl.Trainer(resume_from_checkpoint='./processed/logs/lightning_logs/version_0/checkpoints/epoch=60-step=6222.ckpt')
    Trainer.fit(ViT,Training_Loader,Validation_Loader)
    Trainer.save_checkpoint("./ViT.ckpt")

    
