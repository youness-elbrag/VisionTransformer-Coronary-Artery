### VisionTransformer-Coronary-Artery
This repo provide Full Implmentation of VisionTransformer following Series of Article About Vision-Langauge model , in project i implmented from Scratch using **Pytorch**
we need to kepp mind that there's no big different only few modifications which include 

<div align="left">
    <img src="assets/vit.gif" width="400" height="250"/>
</div>

1. [introduction](#introduction)
2. [environment project](#environment-project)
3. [run project](#run-project)
5. [Predict](#Predict)

### introduction

### Major Different Bwtween Transformer in Language and Vision 

Vision Transformer included few modification in Architicture main are :

1. **Linear Projection:** that used Convolution Network but not in matter of extract features instead of it used to split the image of size 256x256 into sub-Patches to make the model Transformer able to learn and process the Image because of most used in NLP Seq2Seq modeling , here Linear Projection is make each Patch as **Token in Vector**

2. **MLP multi- Layer Perceptence:** to make the model do the task classification used MLP because is widely implemented in Classification only we add **CLS**


**Notation** **in Vision-Transformer only we take Encoder blocks instead of all the model Transfotmer for me infotmation read the Article**

### environment project

first Creat an ENV to run the poject in Dir

**Packages**:
1. numpy
2. torchmetrics
3. matplotlib
4. torch
5. torchvision
6. pytorch-lightning
7. opencv-python

* create the enviromenet here you will need to run 

```sh
conda create --name Segemnetation python=3.6
```

* make sure the requirements.txt exist to the repo 
  the packges if you want fisrt neeed to run 

```sh
pip install -r requirements.txt
```
### run project 
 in The transformer model there's many og Hyper-Parameters to tune baed on the exprement
 and data Size , to make easy to Tune the model there's Script Called **CONFIG.py**
 contain all the Parameters setup based on your purpose it will automatically Generate YAML config.yml FILE 

**Notation** : in this project i used **Pytorch-Lighting Framework** because is easy to creat Loop Traning and use Mulit-GPU 

to Run the model Traninig Folowwing Commmand :
after finishing the Traning a**uto-Checkpoint Save model** Called **ViT.ckpt** will save in current Folder project 

```python
     python train.py --Config config.yml --device  "gpu"
``` 

### Predict

After the Training is done  Run Predict.py to check the prediction using Save **CHECKPOINT** following Command:
Path_Checkpoint

**INSERT_Val_DATA: this one should be Validation data or TestDATA already processed**

```python
     python predict.py --Path INSERT_Val_DATA  -- Path_Checkpoint INSERT_CHECKPOINT_MODEL --OUTPUT INSERT_OUTPUT_STR.PNG
```   

<div align="center">
    <img src="assets/attention.png" width="400" height="500" />
</div>

