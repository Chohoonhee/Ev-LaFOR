import os
import torch
from torch.utils.data import Dataset
from torchvision import transforms
from PIL import Image
from PIL import ImageFile
import joblib
from os import listdir
import numpy as np
from torchvision.transforms import Normalize, ToTensor
import cv2
import torch.multiprocessing as mp
import json
from pathlib import Path
import glob
import random
# class_list = ['BACKGROUND_Google', 'Faces_easy', 'Leopards', 'Motorbikes', 'accordion', 'airplanes', 'anchor', 'ant', 'barrel', 
# 'bass', 'beaver', 'binocular', 'bonsai', 'brain', 'brontosaurus', 'buddha', 'butterfly', 'camera', 'cannon', 'car_side', 'ceiling_fan', 
# 'cellphone', 'chair', 'chandelier', 'cougar_body', 'cougar_face', 'crab', 'crayfish', 'crocodile', 'crocodile_head', 'cup', 'dalmatian', 
# 'dollar_bill', 'dolphin', 'dragonfly', 'electric_guitar', 'elephant', 'emu', 'euphonium', 'ewer', 'ferry', 'flamingo', 'flamingo_head', 
# 'garfield', 'gerenuk', 'gramophone', 'grand_piano', 'hawksbill', 'headphone', 'hedgehog', 'helicopter', 'ibis', 'inline_skate', 'joshua_tree', 
# 'kangaroo', 'ketch', 'lamp', 'laptop', 'llama', 'lobster', 'lotus', 'mandolin', 'mayfly', 'menorah', 'metronome', 'minaret', 'nautilus', 
# 'octopus', 'okapi', 'pagoda', 'panda', 'pigeon', 'pizza', 'platypus', 'pyramid', 'revolver', 'rhino', 'rooster', 'saxophone', 'schooner', 
# 'scissors', 'scorpion', 'sea_horse', 'snoopy', 'soccer_ball', 'stapler', 'starfish', 'stegosaurus', 'stop_sign', 'strawberry', 'sunflower',
#  'tick', 'trilobite', 'umbrella', 'watch', 'water_lilly', 'wheelchair', 'wild_cat', 'windsor_chair', 'wrench', 'yin_yang']

ImageFile.LOAD_TRUNCATED_IMAGES = True

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


def random_shift_events(events, max_shift=20, resolution=(180, 240)):
    H, W = resolution
    x_shift, y_shift = np.random.randint(-max_shift, max_shift+1, size=(2,))
    events[:,0] += x_shift
    events[:,1] += y_shift

    valid_events = (events[:,0] >= 0) & (events[:,0] < W) & (events[:,1] >= 0) & (events[:,1] < H)
    events = events[valid_events]

    return events

def random_flip_events_along_x(events, resolution=(180, 240), p=0.5):
    H, W = resolution
    if np.random.random() < p:
        events[:,0] = W - 1 - events[:,0]
    return events

def read_dataset(event_path):
    """Reads in the TD events contained in the N-MNIST/N-CALTECH101 dataset file specified by 'filename'"""
    event = np.load(event_path)
    event = event['event_data']
    event = np.array([event['x'], event['y'], event['t'], event['p'].astype(np.uint8)]).T
    event = event.astype(np.float32)
    
    

    return torch.tensor(event)
    


class N_Imagenet_Tiny(Dataset):
    def __init__(self, prompts, code, image_dir = '/mnt4/ILSVRC2012', event_dir = '/mnt4/N_ImageNet/N_Imagenet',
            split = 'test', label_type = 'soft_labels'):
        super().__init__()
        self.split = split # This defines the split you are going to use
                           # (split files are called 'train.txt' and 'test.txt')
        
        self.data = []
        train = open("/mnt4/N_ImageNet/N_Imagenet/train_list.txt", "r")
        test = open("/mnt4/N_ImageNet/N_Imagenet/val_list.txt", "r")
        image_root = image_dir
        event_root = event_dir
        
        
        
        self.code_prompts = {code[i]: prompts[i] for i in range(len(code))}
        
        
        self.code_labels = dict()

        for i, label in enumerate(code):
            self.code_labels.update({label : i})

        image_dict = dict()
        for i in range(len(prompts)):
            image_dict.update({i : []})
            
        if(self.split == "train"):
            image_path = Path(image_root + '/extracted_train')
        else:
            image_path = Path(image_root + '/extracted_val')
        
        for child in image_path.iterdir():
            img_code = str(child).split('/')[-1]
        
            if img_code in self.code_labels.keys():
                img_l = self.code_labels[img_code]
                for path in glob.glob(str(child) + '/*.JPEG'):        
                    image_dict[img_l].append(path)

        if(self.split == "train"):
            # train.seek(0, 0)
            for line in train:
                event = line.strip()
                code = event.split('/')[2]
                if code in self.code_labels.keys():
                    
                    img = image_dir + event.replace('.npz', '.JPEG')
                    event = event_dir + event
                    l = self.code_labels[code]
                    self.data.append((l, event, img))
                    # self.data.append((l, event))         
        else:
            # test.seek(0, 0)
            for line in test:
                event = line.strip()
                code = event.split('/')[2]
                img = image_dir + '/extracted_val/' + event.replace('.npz', '.JPEG').split('/')[-1]
                if not os.path.isfile(img):
                    continue
                if code in self.code_labels.keys():
                    event = event_dir + event
                    l = self.code_labels[code]
                    self.data.append((l, event, img))
        self.image_path_dict = image_dict       
        # self.norm = Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711))
        self.norm = Normalize((0.5, ), (0.5, ))
        self.totensor = ToTensor()
        self.transform = torch.nn.Sequential(
                    transforms.Resize((224, 224))
        )

    def __getitem__(self, index):
        label = self.data[index][0]
        events = read_dataset(self.data[index][1])
        if (self.split == "train"):
            # image = pil_loader(random.choice(self.image_path_dict[int(label)]))
            image = pil_loader(self.data[index][2])
        else:
            image = pil_loader(self.data[index][2])
        image_np = np.array(image)
        gray = cv2.cvtColor(image_np, cv2.COLOR_BGR2GRAY)
        image = self.totensor(gray)
        image = self.norm(image)
        image = torch.nn.functional.upsample(image.unsqueeze(0).repeat(1,3,1,1), size=(224, 224), mode='bilinear', align_corners=True).squeeze(0)
        

        # return image, events, label, edge
        return events, label, index, image
    
    def __len__(self):
        return len(self.data)

