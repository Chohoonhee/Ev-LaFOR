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
import pdb
import torch.nn as nn
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
    
    event_stack = render(event['x'], event['y'], event['p'], 480, 640)
    # kernel = np.ones((2,2), np.uint8)
    # event_stack = cv2.morphologyEx(event_stack, cv2.MORPH_OPEN, kernel)

    inverse_event = np.array([event['x'][::-1], event['y'][::-1], event['t'].max() - event['t'], event['p'][::-1].astype(np.uint8)]).T
    event = np.array([event['x'], event['y'], event['t'], event['p'].astype(np.uint8)]).T
    
    
    event_stack = event_stack.astype(np.float32)
    event = event.astype(np.float32)
    inverse_event = inverse_event.astype(np.float32)
    return torch.tensor(event), torch.tensor(event_stack), torch.tensor(inverse_event)

def render(x: np.ndarray, y: np.ndarray, pol: np.ndarray, H: int, W: int) -> np.ndarray:
    assert x.size == y.size == pol.size
    assert H > 0
    assert W > 0
    img = np.full((H,W,1), fill_value=255,dtype='uint8')
    mask = np.zeros((H,W),dtype='int32')
    pol = pol.astype('int')
    pol[pol==0]=-1
    mask1 = (x>=0)&(y>=0)&(W>x)&(H>y)
    mask[y[mask1],x[mask1]]=pol[mask1]
    img[mask==0]=255
    # img[mask==-1]=[255,0,0]
    img[mask==1]=0
    return img
    

class N_Imagenet100(Dataset):
    def __init__(self, prompts, code, preprocess=None, im_size=(224,224), color=False, sub=False, image_dir = '/mnt4/media_from_jm/N_ImageNet/data/ImageNet', event_dir = '/mnt4/media_from_jm//N_ImageNet/data/N_ImageNet',
            split = 'test',  pair= False, test_type = 'zero', clsw =  False, zero_code = None):
        super().__init__()
        self.split = split # This defines the split you are going to use
                           # (split files are called 'train.txt' and 'test.txt')
        self.color = color
        self.data = []
        train = open("./data/imagenet100_train_split_half.txt", "r")
        test = open("./data/val_100_list.txt", "r")
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
            image_path = Path(image_root + '/extracted_100_train')
            event_path = Path(event_root + '/extracted_100_train')
        else:
            image_path = Path(image_root + '/extracted_100_val')
            event_path = Path(event_root + '/extracted_100_val')
        
        label_set = set()
        img_dict = dict()
        if(self.split == "train"):
            train.seek(0, 0)
            for line in train:
                line = line.strip()
                event_list = line.split("&")[0].split(' ')
                image_list = line.split("&")[1].split(' ')
                code = event_list[0].split('/')[0]
                img_dict.update({label : image_list})
                for event in event_list:
                    if code in self.code_labels.keys():
                        if pair:
                            img = image_path / event.replace('.npz', '.JPEG')
                        else:
                            img = image_path / image_list[random.choice(range(len(image_list)))].replace('.npz', '.JPEG')
                        event = event_path / event
   
                        l = self.code_labels[code]
                        self.data.append((l, event, img)) 
                label_set.add(label)       

        inv_labels = dict()
        for i, label in enumerate(label_set):
            inv_labels.update({i : label})
        
        for line in test:
            line = line.strip()
            label = line.split("/")[0]
            label_set.add(label)

        labels = dict()
        label_set = sorted(label_set)
        
        for i, label in enumerate(label_set):
            labels.update({label : i})
        
        if self.split != "train":
            test.seek(0, 0)
            for line in test:
                event = line.strip()
                # event = event.replace('extracted_val', 'extracted_100_val')
                code = event.split('/')[0]
                if zero_code != None:
                    if code not in zero_code:
                        continue
                img = image_dir + '/extracted_100_val/' + event.replace('.npz', '.JPEG').split('/')[-1]
                if not os.path.isfile(img):
                    pdb.set_trace()
                    continue
                if code in self.code_labels.keys():
                    event = event_path / event
                    l = self.code_labels[code]
                    self.data.append((l, event, img))
        self.image_path_dict = image_dict
        
        if self.split == 'train':
            if sub:
                self.data = np.array(self.data)[sorted(np.random.choice(len(self.data), 12000, replace=False))]
                self.data = [(int(d[0]), d[1], d[2]) for d in self.data.tolist()]
            else:
                self.data = np.array(self.data)
                self.data = [(int(d[0]), d[1], d[2]) for d in self.data.tolist()]
        self.labels = labels
        self.inv_labels = inv_labels
        self.img_dict = img_dict
                    
        if preprocess == None:
            # self.norm = Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711))
            if color:
                self.norm = Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711))
            else:
                self.norm = Normalize((0.5, ), (0.5, ))
            self.totensor = ToTensor()
            self.transform = torch.nn.Sequential(
                        transforms.Resize((224, 224))
            )
        else:
            if color:
                self.norm = preprocess.transforms[4]    
            else:
                mean = np.array(preprocess.transforms[4].mean)
                std = np.array(preprocess.transforms[4].std)
                # https://docs.opencv.org/2.4/modules/imgproc/doc/miscellaneous_transformations.html#void%20cvtColor%28InputArray%20src,%20OutputArray%20dst,%20int%20code,%20int%20dstCn%29
                weight = np.array((0.299, 0.587, 0.114))
                gray_mean = sum(mean * weight)
                gray_std = np.sqrt(np.sum(np.power(weight,2) * np.power(std,2)))
                self.norm = Normalize((gray_mean, ), (gray_std, ))
            self.totensor = preprocess.transforms[3]
            self.transform = preprocess.transforms[0]
        
        self.im_size = im_size

    def __getitem__(self, index):
        label = self.data[index][0]
        events, event_stack, inverse_event = read_dataset(self.data[index][1])
        
        if (self.split == "train"):
            # image = pil_loader(random.choice(self.image_path_dict[int(label)]))
            image = pil_loader(self.data[index][2])
        else:
            image = pil_loader(self.data[index][2])
        image_np = np.array(image)
        if self.color:
            image = self.totensor(image_np)
            image = self.norm(image)
            image = torch.nn.functional.upsample(image.unsqueeze(0), size=self.im_size, mode='bilinear', align_corners=True).squeeze(0)
        else:
            gray = cv2.cvtColor(image_np, cv2.COLOR_BGR2GRAY)
            image = self.totensor(gray)
            image = self.norm(image)
            image = torch.nn.functional.upsample(image.unsqueeze(0).repeat(1,3,1,1), size=self.im_size, mode='bilinear', align_corners=True).squeeze(0)
        # image = torch.nn.functional.upsample(image.unsqueeze(0).repeat(1,3,1,1), size=(224, 224), mode='bicubic', align_corners=True).squeeze(0)
        # import pdb; pdb.set_trace()
        m = nn.Upsample(size=self.im_size, mode='nearest')
        # event_stack = torch.nn.functional.upsample(event_stack.unsqueeze(0).repeat(1,3,1,1), size=(224, 224), mode='bicubic', align_corners=True).squeeze(0)
        event_stack = m(event_stack.reshape(1,480,640).unsqueeze(0).repeat(1,3,1,1)).squeeze(0)

        # return image, events, label, edge
        return events, label, index, image, event_stack, inverse_event
    
    def __len__(self):
        return len(self.data)

