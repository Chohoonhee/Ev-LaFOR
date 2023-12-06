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

    # return all_p, all_x, all_y, all_ts
    # return all_x, all_y, all_ts, all_p
    return torch.tensor(np.array([all_x, all_y, all_ts, all_p]).T, dtype=torch.float32)
    


class Caltech101DataEvent(Dataset):
    def __init__(self, data_dir = './data/caltech-101/caltech-101/101_ObjectCategories', event_dir = './data/Caltech101',
            split = 'test'):
        super().__init__()
        self.split = split # This defines the split you are going to use
                           # (split files are called 'train.txt' and 'test.txt')
        
        self.data = []
        train = open("./data/caltech_train.txt", "r")
        test = open("./data/caltech_test.txt", "r")
        root = data_dir
        event_root = event_dir
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
                event = event_root + "/" + line.replace('jpg', 'bin')
                l = labels[label]
                self.data.append((image, l, event))
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
                event = event_root + "/" + line.replace('jpg', 'bin')
                l = labels[label]
                self.data.append((image, l, event))
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

        # image = self.totensor(image)
        image = self.totensor(gray)
        edge = self.totensor(edge)
        edge = torch.nn.functional.upsample(edge.unsqueeze(0).repeat(1,3,1,1), size=(224, 224), mode='bilinear', align_corners=True).squeeze(0)
        # image = torch.nn.functional.upsample(image.unsqueeze(0).repeat(1,1,1,1), size=(224, 224), mode='bilinear', align_corners=True).squeeze(0)
        image = torch.nn.functional.upsample(image.unsqueeze(0).repeat(1,3,1,1), size=(224, 224), mode='bilinear', align_corners=True).squeeze(0)
        events = read_dataset(self.data[index][2])
        events = events[torch.where(events[:, 2]<10000)]
        # return image, events, label, edge
        return events, label, index, image
    
    def __len__(self):
        return len(self.data)

