import os
import torch
from torch.utils.data import Dataset
from torchvision import transforms
from PIL import Image
import joblib
from os import listdir
from torchvision.transforms import Normalize, ToTensor
import numpy as np
import cv2

# class_list = ['BACKGROUND_Google', 'Faces_easy', 'Leopards', 'Motorbikes', 'accordion', 'airplanes', 'anchor', 'ant', 'barrel', 
# 'bass', 'beaver', 'binocular', 'bonsai', 'brain', 'brontosaurus', 'buddha', 'butterfly', 'camera', 'cannon', 'car_side', 'ceiling_fan', 
# 'cellphone', 'chair', 'chandelier', 'cougar_body', 'cougar_face', 'crab', 'crayfish', 'crocodile', 'crocodile_head', 'cup', 'dalmatian', 
# 'dollar_bill', 'dolphin', 'dragonfly', 'electric_guitar', 'elephant', 'emu', 'euphonium', 'ewer', 'ferry', 'flamingo', 'flamingo_head', 
# 'garfield', 'gerenuk', 'gramophone', 'grand_piano', 'hawksbill', 'headphone', 'hedgehog', 'helicopter', 'ibis', 'inline_skate', 'joshua_tree', 
# 'kangaroo', 'ketch', 'lamp', 'laptop', 'llama', 'lobster', 'lotus', 'mandolin', 'mayfly', 'menorah', 'metronome', 'minaret', 'nautilus', 
# 'octopus', 'okapi', 'pagoda', 'panda', 'pigeon', 'pizza', 'platypus', 'pyramid', 'revolver', 'rhino', 'rooster', 'saxophone', 'schooner', 
# 'scissors', 'scorpion', 'sea_horse', 'snoopy', 'soccer_ball', 'stapler', 'starfish', 'stegosaurus', 'stop_sign', 'strawberry', 'sunflower',
#  'tick', 'trilobite', 'umbrella', 'watch', 'water_lilly', 'wheelchair', 'wild_cat', 'windsor_chair', 'wrench', 'yin_yang']

def pil_loader(path):
    # open path as file to avoid ResourceWarning (https://github.com/python-pillow/Pillow/issues/835)
    with open(path, 'rb') as f:
        img = Image.open(f)
        return img.convert('RGB')

class_list = ['Faces_easy', 'Leopards', 'Motorbikes', 'accordion', 'airplanes', 'anchor', 'ant', 'barrel', 'bass', 'beaver', 
'binocular', 'bonsai', 'brain', 'brontosaurus', 'buddha', 'butterfly', 'camera', 'cannon', 'car_side', 'ceiling_fan', 'cellphone', 
'chair', 'chandelier', 'cougar_body', 'cougar_face', 'crab', 'crayfish', 'crocodile', 'crocodile_head', 'cup', 'dalmatian', 'dollar_bill',
 'dolphin', 'dragonfly', 'electric_guitar', 'elephant', 'emu', 'euphonium', 'ewer', 'ferry', 'flamingo', 'flamingo_head', 'garfield', 'gerenuk', 
 'gramophone', 'grand_piano', 'hawksbill', 'headphone', 'hedgehog', 'helicopter', 'ibis', 'inline_skate', 'joshua_tree', 'kangaroo', 'ketch',
  'lamp', 'laptop', 'llama', 'lobster', 'lotus', 'mandolin', 'mayfly', 'menorah', 'metronome', 'minaret', 'nautilus', 'octopus', 'okapi',
   'pagoda', 'panda', 'pigeon', 'pizza', 'platypus', 'pyramid', 'revolver', 'rhino', 'rooster', 'saxophone', 'schooner', 'scissors', 'scorpion', 
   'sea_horse', 'snoopy', 'soccer_ball', 'stapler', 'starfish', 'stegosaurus', 'stop_sign', 'strawberry', 'sunflower', 'tick', 'trilobite', 
   'umbrella', 'watch', 'water_lilly', 'wheelchair', 'wild_cat', 'windsor_chair', 'wrench', 'yin_yang']

split_dict = {
        'train': './data/caltech_train.txt',
        'test': './data/caltech_test.txt'
        }

def read_text_lines(filepath):
    with open(filepath, 'r') as f:
        lines = f.readlines()
    lines = [l.rstrip() for l in lines]
    return lines



class Caltech101Data(Dataset):
    def __init__(self, data_dir = './data/caltech-101/caltech-101/101_ObjectCategories', split = 'test'):
        super().__init__()
        self.split = split # This defines the split you are going to use
                           # (split files are called 'train.txt' and 'test.txt')
        
        self.data = []
        train = open("./data/caltech_train.txt", "r")
        test = open("./data/caltech_test.txt", "r")
        root = data_dir
        label_set = set()
        for line in train:
            line = line.strip()
            label, _ = line.split("/")
            if label in ["BACKGROUND_Google", "Faces"]:
                continue
            label_set.add(label)
        for line in test:
            line = line.strip()
            label, _ = line.split("/")
            if label in ["BACKGROUND_Google", "Faces"]:
                continue
            label_set.add(label)

        labels = dict()
        label_set = sorted(label_set)

        for i, label in enumerate(label_set):
            labels.update({label : i})

        if(self.split == "train"):
            train.seek(0, 0)
            for line in train:
                line = line.strip()
                label, image = line.split("/")
                if label in ["BACKGROUND_Google", "Faces"]:
                    continue
                # image = Image.open(root + "/" + line)
                image = root + "/" + line
                l = labels[label]
                self.data.append((image, l))
        else:
            test.seek(0, 0)
            for line in test:
                line = line.strip()
                label, image = line.split("/")
                if label in ["BACKGROUND_Google", "Faces"]:
                    continue
                # import pdb; pdb.set_trace()
                # print(root + "/" + line)
                # image = Image.open(root + "/" + line)
                image = root + "/" + line
                l = labels[label]
                self.data.append((image, l))
        self.norm = Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711))
        self.totensor = ToTensor()
        self.transform = torch.nn.Sequential(
                    transforms.Resize((224, 224))
        )

    def __getitem__(self, index):
        label = self.data[index][1]
        image = pil_loader(self.data[index][0])
        image_np = np.array(image)
        gray = cv2.cvtColor(image_np, cv2.COLOR_BGR2GRAY)
        gaussian = cv2.GaussianBlur(gray, (3, 3), 0)
        edge = cv2.Laplacian(gaussian, cv2.CV_8U, ksize=3)
        # import pdb; pdb.set_trace()
        # edge = self.transform(self.norm(self.totensor(edge).unsqueeze(0).repeat(1,3,1,1))).squeeze(0)
        # image = self.transform(self.norm(self.totensor(gray).unsqueeze(0).repeat(1,3,1,1))).squeeze(0)
        # image = self.transform(self.norm(self.totensor(gaussian).unsqueeze(0).repeat(1,3,1,1))).squeeze(0)
        
        # image = self.norm(self.totensor(image))
        # import pdb; pdb.set_trace()
        
        image = self.totensor(image)
        edge = self.totensor(edge)
        
        # image = self.transform(self.norm(self.totensor(image).unsqueeze(0))).squeeze(0)
        image = torch.nn.functional.upsample(image.unsqueeze(0).repeat(1,1,1,1), size=(224, 224), mode='bilinear', align_corners=True).squeeze(0)
        edge = torch.nn.functional.upsample(edge.unsqueeze(0).repeat(1,3,1,1), size=(224, 224), mode='bilinear', align_corners=True).squeeze(0)
        # pdb.set_trace()
        return image, edge, label
    
    def __len__(self):
        return len(self.data)

