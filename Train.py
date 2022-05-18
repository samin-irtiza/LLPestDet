import torchvision.transforms as transforms
from InsectPestDataset import InsectPestDataset as IPDataset
import os
from torchvision.utils import save_image
import random


my_transforms=transforms.Compose([transforms.ToPILImage(),
transforms.ColorJitter(brightness=[0.1,0.25]),
transforms.ToTensor()])
root_dir="./Dataset"
dataset= IPDataset(root_dir,transforms=my_transforms)

save_dir=os.path.join(root_dir,"TransformedImages")
if not os.path.isdir(save_dir):
    os.mkdir(save_dir)
    pass

img_num=0
for img,label in dataset:
    save_image(img,os.path.join(save_dir,f"IP{img_num}.jpg"))
    img_num+=1
