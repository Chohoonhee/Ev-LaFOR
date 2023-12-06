import os
import torch
from torch.utils.data import Dataset
from torchvision import transforms
from PIL import Image
import joblib
from os import listdir
import numpy as np
from torchvision.transforms import Normalize, ToTensor
import cv2
import random
import torch.nn as nn

import torch.nn.functional as F


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

def read_dataset(filename):
    """Reads in the TD events contained in the N-MNIST/N-CALTECH101 dataset file specified by 'filename'"""
    f = open(filename, 'rb')
    raw_data = np.fromfile(f, dtype=np.uint8)
    f.close()
    raw_data = np.int32(raw_data)

    all_y = raw_data[1::5]
    all_x = raw_data[0::5]
    all_p = (raw_data[2::5] & 128) >> 7 #bit 7
    all_ts = ((raw_data[2::5] & 127) << 16) | (raw_data[3::5] << 8) | (raw_data[4::5])


    #Process time stamp overflow events
    time_increment = 2 ** 13
    overflow_indices = np.where(all_y == 240)[0]
    for overflow_index in overflow_indices:
        all_ts[overflow_index:] += time_increment

    # event_stack = render_3ch(all_x, all_y, all_p, 180, 240)
    event_stack = render_3ch(all_x[::-1], all_y[::-1], all_p[::-1], 180, 240)
    # event_stack = event_stack.astype(np.float32)
    # return all_p, all_x, all_y, all_ts
    # return all_x, all_y, all_ts, all_p
    reverse_ts = all_ts[::-1]
    
    return torch.tensor(np.array([all_x, all_y, all_ts, all_p]).T, dtype=torch.float32), torch.tensor(event_stack), \
        torch.tensor(np.array([all_x[::-1], all_y[::-1], reverse_ts.max()-reverse_ts, all_p[::-1]]).T, dtype=torch.float32)

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

def render_3ch(x: np.ndarray, y: np.ndarray, pol: np.ndarray, H: int, W: int) -> np.ndarray:
    assert x.size == y.size == pol.size
    assert H > 0
    assert W > 0
    img = np.full((H,W,3), fill_value=255,dtype='uint8')
    mask = np.zeros((H,W),dtype='int32')
    pol = pol.astype('int')
    pol[pol==0]=-1
    mask1 = (x>=0)&(y>=0)&(W>x)&(H>y)
    mask[y[mask1],x[mask1]]=pol[mask1]
    img[mask==0]=[255,255,255]
    img[mask==-1]=[255,0,0]
    img[mask==1]=[0,0,255]
    
    return img


# image_dir = '/media/ssd2tb2/data/ImageNet', event_dir = '/media/ssd2tb2/data/N_ImageNet',

class Caltech101DataEventOursUnpairNoise(Dataset):
    def __init__(self, data_dir = '/mnt4/media_from_jm/caltech-101/caltech-101/101_ObjectCategories', event_dir = '/mnt4/media_from_jm/Caltech101',
            split = 'test', train_type = 'zero', pair= False, test_type = 'zero', clsw =  False, preprocess=None, color=None):
        super().__init__()
        self.split = split # This defines the split you are going to use
                           # (split files are called 'train.txt' and 'test.txt')
        self.cls = clsw
        self.pair = pair
        self.data = []
        # train = open("./data/caltech_train_ours_zeroshot.txt", "r")
        # test = open("./data/caltech_test_ours.txt", "r")
        # test = open("./data/caltech_test_ours_zeroshot.txt", "r")
        if train_type == 'zero':
            train = open("./data/caltech_train_ours_zeroshot20.txt", "r")
        else:
            train = open("./data/caltech_train_split_half.txt", "r")
            
        if test_type == 'zero':
            test = open("./data/caltech_test_ours_zeroshot20.txt", "r")
        else:
            test = open("./data/caltech_test_ours.txt", "r")
        # test = open("./data/caltech_test.txt", "r")

        self.root = data_dir
        event_root = event_dir
        label_set = set()
        img_dict = dict()

        for line in train:
            line = line.strip()
            event_list = line.split("&")[0].split(' ')
            image_list = line.split("&")[1].split(' ')
            label = event_list[0].split('/')[0]
            if label in ["BACKGROUND_Google", "Faces"]:
                continue
            img_dict.update({label : image_list})
            for event in event_list:
                if pair:
                    image  = self.root + "/" +  event
                else:
                    image  = self.root + "/" + image_list[random.choice(range(len(image_list)))]
                event = event_root + "/" + event.replace('jpg', 'bin')
                if self.split == "train":
                    self.data.append((image, label, event))
            label_set.add(label)
        
        inv_labels = dict()
        for i, label in enumerate(label_set):
            inv_labels.update({i : label})
        
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
            

        # if(self.split == "train"):
        #     train.seek(0, 0)
        #     for line in train:
        #         line = line.strip()
        #         label, image = line.split("/")
        #         if label in ["BACKGROUND_Google", "Faces"]:
        #             continue
        #         # image = Image.open(root + "/" + line)
        #         image = root + "/" + line
        #         event = event_root + "/" + line.replace('jpg', 'bin')
        #         l = labels[label]
        #         self.data.append((image, l, event))
        # else:
        if self.split != "train":
            
            test.seek(0, 0)
            for line in test:
                line = line.strip()
                label, image = line.split("/")
                if label in ["BACKGROUND_Google", "Faces"]:
                    continue
                # import pdb; pdb.set_trace()
                # print(root + "/" + line)
                # image = Image.open(root + "/" + line)
                image = self.root + "/" + line
                event = event_root + "/" + line.replace('jpg', 'bin')
                l = labels[label]
                self.data.append((image, l, event))
        # self.norm = Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711))
        self.totensor = ToTensor()
        self.transform = torch.nn.Sequential(
                    transforms.Resize((224, 224))
        )
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

    def __getitem__(self, index):
        if self.split == "train":
            if self.cls or self.pair:
                image = pil_loader(self.data[index][0])
            else:
                im_lab = self.inv_labels[random.choice(range(len(self.img_dict)))]
                image_path = self.root + "/" + random.choice(self.img_dict[im_lab])
                image = pil_loader(image_path)
            label = self.labels[self.data[index][1]]
        else:         
            image_path = self.data[index][0]
            image = pil_loader(image_path)
            label = self.data[index][1]
        
        if self.split == 'train':
            if np.random.random() < 0.5:
                events, event_stack, inv_event = read_dataset(self.data[index][2])
            else:
                inv_event, event_stack, events = read_dataset(self.data[index][2])
        else:
            events, event_stack, inv_event = read_dataset(self.data[index][2])
        
        
        image_np = np.array(image)
        gray = cv2.cvtColor(image_np, cv2.COLOR_BGR2GRAY)
       
        image = self.totensor(image)
        # image = self.totensor(gray)
        if self.split == "train":
            image = self.norm(image)
            
        # image = torch.nn.functional.upsample(image.unsqueeze(0).repeat(1,3,1,1), size=(224, 224), mode='bilinear', align_corners=True).squeeze(0)
        image = torch.nn.functional.upsample(image.unsqueeze(0), size=(224, 224), mode='bilinear', align_corners=True).squeeze(0)
        return events, label, index, image, event_stack, inv_event
    
    def __len__(self):
        return len(self.data)