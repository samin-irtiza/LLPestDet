import os
import numpy as np
import torch
import torch.utils.data
import pandas as pd
import xml.etree.ElementTree as ET
import torchvision.io as io



# data = []
# cols = []
# xml_data= ET.parse("./Annotations/IP000000077.xml")
# root=xml_data.getroot()
# objects=[obj for obj in root if obj.tag=="object"]
# className = [(int(obj[0].text)+1) for obj in objects]
# labels=torch.tensor(className,dtype=torch.int64)
# print(labels)
class InsectPestDataset(torch.utils.data.Dataset):
    def __init__(self, root, transforms=None):
        self.root = root
        self.transforms = transforms
        # load all image files, sorting them to
        # ensure that they are aligned
        self.imgs = list(sorted(os.listdir(os.path.join(root, "JPEGImages"))))
        self.Annotations = list(sorted(os.listdir(os.path.join(root, "Annotations"))))

    def __getitem__(self, idx):
        img_path = os.path.join(self.root, "JPEGImages", self.imgs[idx])
        anno_path = os.path.join(self.root, "Annotations", self.Annotations[idx])
        img=io.read_image(img_path,io.ImageReadMode.RGB)
        xml_data= ET.parse(anno_path)
        root=xml_data.getroot()
        image_id=torch.tensor([idx])
        objects=[obj for obj in root if obj.tag=="object"]
        iscrowd = torch.zeros((len(objects),), dtype=torch.int64)
        className = [(int(obj[0].text)+1) for obj in objects]
        labels=torch.tensor(className,dtype=torch.int64)
        boxes = []
        for i,val in enumerate(objects):
            xmin = int(val[4][0].text)
            ymin = int(val[4][1].text)
            xmax = int(val[4][2].text)
            ymax = int(val[4][3].text)
            boxes.append([xmin, ymin, xmax, ymax])
        boxes=np.asarray(boxes)
        area = (boxes[:,3] - boxes[:, 1]) * (boxes[:, 2] - boxes[:, 0])

        target = {}
        target["boxes"] = boxes
        target["labels"] = labels
        target["image_id"] = image_id
        target["area"] = area
        target["iscrowd"] = iscrowd
        if self.transforms is not None:
            img = self.transforms(img)

        return img, target


def __len__(self):
    return len(self.imgs)
