import os
import torch
from torch.utils.data import Dataset
from torchvision import transforms
from PIL import Image
import joblib
from os import listdir
import numpy as np
from torchvision.transforms import Normalize, ToTensor
from .event.representations import VoxelGrid, ReconVoxelGrid
from .event.eventslicer import EventSlicer
from joblib import Parallel, delayed
import random
import cv2

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
    
    event_stack = render(all_x, all_y, all_p, 180, 240)
    # height = 180
    # width = 240
    # img = render(all_x, all_y, all_p, height, width)
    # return img
    
    td_indices = np.where(all_y != 240)[0]

    all_x = all_x[td_indices]
    width = all_x.max() + 1
    all_y = all_y[td_indices]
    height = all_y.max() + 1
    all_ts = all_ts[td_indices]
    all_p = all_p[td_indices]
    
    
    return all_p, all_ts, all_x, all_y, width, height, torch.tensor(event_stack)

    # return all_x, all_y, all_ts, all_p
    # return torch.tensor(np.array([all_x, all_y, all_ts, all_p]).T, dtype=torch.float32)
    


class Caltech101DataEventVoxelUnpair(Dataset):
    def __init__(self, data_dir = './data/caltech-101/caltech-101/101_ObjectCategories', event_dir = './data/Caltech101',
            split = 'test', train_type = 'zero', pair= False, test_type = 'zero', clsw =  False):
        super().__init__()
        self.split = split # This defines the split you are going to use
                           # (split files are called 'train.txt' and 'test.txt')
        self.cls = clsw
        self.pair = pair
        self.nr_events_data = 5
        self.recon_num_bins = 5
        self.nr_events_per_data = 1000
        self.separate_pol = False
        self.normalize_event = False
        self.remove_time_window = 250

        self.height = 180
        self.width = 240
        
        # self.recon_voxel_grid240180 = ReconVoxelGrid(self.recon_num_bins, self.height, self.width, normalize=False)
        # self.recon_voxel_grid180240 = ReconVoxelGrid(self.recon_num_bins, self.width, self.height, normalize=False)



        self.data = []
        # train = open("./data/caltech_train.txt", "r")
        if train_type == 'zero':
            train = open("./data/caltech_train_ours_zeroshot20.txt", "r")
        else:
            train = open("./data/caltech_train_split_half.txt", "r")
        # test = open("./data/caltech_test.txt", "r")
        if test_type == 'zero':
            test = open("./data/caltech_test_ours_zeroshot20.txt", "r")
        else:
            test = open("./data/caltech_test_ours.txt", "r")
        root = data_dir
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
                    image  = root + "/" +  event
                else:
                    image  = root + "/" + image_list[random.choice(range(len(image_list)))]
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
                image = root + "/" + line
                event = event_root + "/" + line.replace('jpg', 'bin')
                l = labels[label]
                self.data.append((image, l, event))
        self.norm = Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711))
        self.totensor = ToTensor()
        self.transform = torch.nn.Sequential(
                    transforms.Resize((224, 224))
        )
        self.inv_labels = inv_labels
        self.img_dict = img_dict
        self.root = root

    def recon_events_to_voxel_grid(self,x, y, p, t, voxel_grid):
        t = (t - t[0]).astype('float32')
        t = (t/t[-1])
        x = x.astype('float32')
        y = y.astype('float32')
        pol = p.astype('float32')
        
        # return self.recon_voxel_grid.convert(
        #         torch.from_numpy(x),
        #         torch.from_numpy(y),
        #         torch.from_numpy(pol),
        #         torch.from_numpy(t))
        # return self.recon_voxel_grid240180.convert(
        #         torch.from_numpy(x),
        #         torch.from_numpy(y),
        #         torch.from_numpy(pol),
        #         torch.from_numpy(t))
        return voxel_grid.convert(
                torch.from_numpy(x),
                torch.from_numpy(y),
                torch.from_numpy(pol),
                torch.from_numpy(t))

    def generate_event_tensor(self, job_id, events, event_tensor, nr_events_per_data, voxel_grid):
        id_start = job_id * nr_events_per_data
        id_end = (job_id + 1) * nr_events_per_data
        events_temp = events[id_start:id_end]
        event_representation = self.recon_events_to_voxel_grid(events_temp[:, 0], events_temp[:, 1], events_temp[:, 3],
                                                         events_temp[:, 2], voxel_grid)
    
        event_tensor[(job_id * self.recon_num_bins):((job_id+1) * self.recon_num_bins), :, :] = event_representation

        return event_tensor

    def recon_events_to_voxel_grid2(self,x, y, p, t):
        t = (t - t[0]).astype('float32')
        t = (t/t[-1])
        x = x.astype('float32')
        y = y.astype('float32')
        pol = p.astype('float32')
        
        return self.recon_voxel_grid180240.convert(
                torch.from_numpy(x),
                torch.from_numpy(y),
                torch.from_numpy(pol),
                torch.from_numpy(t))

    def generate_event_tensor2(self, job_id, events, event_tensor, nr_events_per_data):
        id_start = job_id * nr_events_per_data
        id_end = (job_id + 1) * nr_events_per_data
        events_temp = events[id_start:id_end]
        event_representation = self.recon_events_to_voxel_grid2(events_temp[:, 0], events_temp[:, 1], events_temp[:, 3],
                                                         events_temp[:, 2])
    
        event_tensor[(job_id * self.recon_num_bins):((job_id+1) * self.recon_num_bins), :, :] = event_representation

        return event_tensor

    def __getitem__(self, index):
        label = self.data[index][1]
        
        if self.split == 'train':
            if self.cls or self.pair:
                image = pil_loader(self.data[index][0])
            else:
                im_lab = self.inv_labels[random.choice(range(len(self.img_dict)))]
                image_path = self.root + "/" + random.choice(self.img_dict[im_lab])
                image = pil_loader(image_path)
        else:
            image = pil_loader(self.data[index][0])
            # image = self.transform(self.totensor(image).unsqueeze(0)).squeeze(0)
        image_np = np.array(image)
        gray = cv2.cvtColor(image_np, cv2.COLOR_BGR2GRAY)
        image = self.transform(self.totensor(gray).unsqueeze(0).repeat(1,3,1,1)).squeeze(0)
        p, t, x, y, width, height, event_stack = read_dataset(self.data[index][2])
        events = np.stack([x, y, t, p], axis=-1)
        nr_events_loaded = t.size
        num_bins_total = self.nr_events_data * self.recon_num_bins
        self.nr_events = self.nr_events_data * self.nr_events_per_data
        nr_events_temp = nr_events_loaded // self.nr_events_data


        # if x.max() > y.max():
        event_tensor = torch.zeros((num_bins_total, height, width))
        recon_voxel_grid = ReconVoxelGrid(self.recon_num_bins, height, width, normalize=False)
        Parallel(n_jobs=8, backend="threading")(
            delayed(self.generate_event_tensor)(i, events, event_tensor, nr_events_temp, recon_voxel_grid) for i in range(self.nr_events_data))
        # else:
        #     event_tensor = torch.zeros((num_bins_total, self.width, self.height))
        #     Parallel(n_jobs=8, backend="threading")(
        #         delayed(self.generate_event_tensor2)(i, events, event_tensor, nr_events_temp) for i in range(self.nr_events_data))

        event_tensor = torch.nn.functional.upsample(event_tensor.unsqueeze(0), size=(256, 256), mode='bilinear').squeeze(0)
        # return events, label
        return event_tensor, label, image, event_stack

    def __len__(self):
        return len(self.data)

