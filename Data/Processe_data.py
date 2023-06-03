import os 
from pathlib import Path 
import os 
import cv2
from tqdm.notebook import tqdm
import glob 
import numpy as np
import argparse


# Path_save = Path("./ProcessedFolder/")
# Path_Image = "./coronary-artery-diseaes-dataset-normal-abnormal/Coronary_Artery/Dataset"

def class_into_idx(path_train):
    list_folders = os.listdir(path_train)
    for i in range(len(list_folders)):
        full_path_case = os.path.join(path_train,list_folders[i])
        if full_path_case.split("/")[-1] in "normal":
            img_normal = glob.glob(full_path_case+"/*.jpg")
        else:
            img_abnormal = glob.glob(full_path_case+"/*.jpg")

    return img_normal , img_abnormal

def GrayScale_Resize(img_path):
    y , x , w , h = 20 , 20 , 296 , 296
    image_to_array  = cv2.imread(img_path,cv2.IMREAD_GRAYSCALE)
    image_to_array = cv2.resize(image_to_array, (296,296)).astype(np.float16) 
    crop_image = image_to_array[y:h-y,x:w-x]
    Gray = np.expand_dims(crop_image,-1)
    return Gray


def class_to_idx(class_name:str):
    sem_classes = ['normal','abnormal']
    sem_class_to_idx = {cls: idx for (idx, cls) in enumerate(sem_classes)}
    label = sem_class_to_idx[class_name]
    return label

def Processed_Image(Path_Class,PathFolder):
    sums , sum_square = 0 , 0
    for index_img , img_idx in enumerate(tqdm(Path_Class)):
        image_resize = GrayScale_Resize(img_idx) / 255
        train_or_val = 'train' if index_img < ( len(Path_Class) // 2 ) else "val"
        label = img_idx.split("/")[-2]
        label_idx = class_to_idx(label)
        current_save_path = PathFolder/train_or_val/str(label_idx)
        Is_Exist = os.path.exists(current_save_path)
        if Is_Exist == False:
            current_save_path.mkdir(parents=True , exist_ok=True)
        path_save_image = os.path.join(current_save_path , label+"_"+ str(index_img))
        np.save(path_save_image,image_resize)
        normalizer = 256*256
        if train_or_val == 'train':
            sums += np.sum(image_resize) / normalizer 
            sum_square += (image_resize ** 2).sum() / normalizer
            
    mean = sums / 205
    std = np.sqrt(sum_square / 205 - mean**2)
    print(f"mean Image {mean} \n std Image:{std}")
    return f"Done Process stage {label}"

def parse_args():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--Path", type=str, required=True)
    parser.add_argument("--output", type=str,required=True)

    args = parser.parse_args()
    return args
    
if __name__ == "__main__":
    args = parse_args()
    Path = args.Path
    output = args.output
    Path_save = Path(output)
    print(f"Normal Process Image:{Processed_Image(img_normal,Path_save)} \n Abnormal Processed Image:{Processed_Image(img_abnormal,Path_save)}")