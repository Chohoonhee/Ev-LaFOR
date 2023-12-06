# import torch
# import numpy as np

# from torch.utils.data.dataloader import default_collate


# class Loader:
#     def __init__(self, dataset, batch_size, num_workers, pin_memory, device, shuffle=True):
#         self.device = device
#         split_indices = list(range(len(dataset)))
#         if shuffle:
#             sampler = torch.utils.data.sampler.SubsetRandomSampler(split_indices)
#             self.loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, sampler=sampler,
#                                                       num_workers=num_workers, pin_memory=pin_memory,
#                                                       collate_fn=collate_events)
#         else:
#             self.loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size,
#                                                       num_workers=num_workers, pin_memory=pin_memory,
#                                                       collate_fn=collate_events)
#     def __iter__(self):
#         for data in self.loader:
#             data = [d.to(self.device) for d in data]
#             yield data

#     def __len__(self):
#         return len(self.loader)


# def collate_events(data):
#     labels = []
#     events = []
#     histograms = []
#     for i, d in enumerate(data):
#         labels.append(d[1])
#         histograms.append(d[2])
#         ev = np.concatenate([d[0], i*np.ones((len(d[0]), 1), dtype=np.float32)], 1)
#         events.append(ev)
#     events = torch.from_numpy(np.concatenate(events, 0))
#     labels = default_collate(labels)

#     histograms = default_collate(histograms)

#     return events, labels, histograms


import torch
import numpy as np

from torch.utils.data.dataloader import default_collate


# train_dataloader = DataLoader(, batch_size=args.batch_size, shuffle=True, num_workers=4, drop_last=True, collate_fn=collate_events)
    # val_loader = DataLoader(Caltech101Data(split='train'), batch_size=args.test_batch_size, shuffle=True, num_workers=4)
    # test_loader = DataLoader(, batch_size=args.test_batch_size, num_workers=4, collate_fn=collate_events)

class Loader:
    def __init__(self, dataset, args, device, split, proto_type=False):
        self.device = device
        split_indices = list(range(len(dataset)))
        sampler = torch.utils.data.sampler.SubsetRandomSampler(split_indices)
        # if split == 'train':
        #     shuffle = True
        # else:
        #     shuffle = False
        # shuffle = False
        if proto_type:
            self.loader = torch.utils.data.DataLoader(dataset, batch_size=args.batch_size, sampler=sampler,
                                             num_workers=4,
                                             collate_fn=collate_events_proto)
                                            #  shuffle = shuffle)
        else:
            self.loader = torch.utils.data.DataLoader(dataset, batch_size=args.batch_size, sampler=sampler,
                                             num_workers=4,
                                             collate_fn=collate_events)
                                            #  shuffle = shuffle)
        
    def __iter__(self):
        for data in self.loader:
            data = [d.to(self.device) if not isinstance(d, list) else d for d in data]
            yield data

    def __len__(self):
        return len(self.loader)

# def collate_events(data):
#     labels = []
#     events = []
#     images = []
#     for i, d in enumerate(data):
#         labels.append(d[2])
#         images.append(d[0])
#         ev = np.concatenate([d[1], i*np.ones((len(d[1]),1), dtype=np.float32)],1)
#         # print(ev.shape)
#         # events.append(ev.T)
#         events.append(ev)
#     events = torch.from_numpy(np.concatenate(events,0))
#     labels = default_collate(labels)
#     images = default_collate(images)
#     return images, events, labels

# def collate_events(data):
#     labels = []
#     events = []
#     images = []
#     edges = []
#     for i, d in enumerate(data):
#         labels.append(d[2])
#         images.append(d[0])
#         edges.append(d[3])
#         ev = np.concatenate([d[1], i*np.ones((len(d[1]),1), dtype=np.float32)],1)
#         # print(ev.shape)
#         # events.append(ev.T)
#         events.append(ev)
#     events = torch.from_numpy(np.concatenate(events,0))
#     labels = default_collate(labels)
#     images = default_collate(images)
#     edges = default_collate(edges)
#     return images, events, labels, edges

# def collate_events(data):
#     labels = []
#     events = []
#     indexs = []
#     images = []
#     for i, d in enumerate(data):
#         labels.append(d[1])
#         indexs.append(d[2])
#         images.append(d[3])
#         ev = np.concatenate([d[0], i*np.ones((len(d[0]),1), dtype=np.float32)],1)
#         # print(ev.shape)
#         # events.append(ev.T)
#         events.append(ev)
#     events = torch.from_numpy(np.concatenate(events,0))
#     labels = default_collate(labels)
#     indexs = default_collate(indexs)
#     images = default_collate(images)
#     return events, labels, indexs, images

def collate_events(data):
    labels = []
    events = []
    indexs = []
    images = []
    event_stack = []
    inv_events = []
    for i, d in enumerate(data):
        labels.append(d[1])
        indexs.append(d[2])
        images.append(d[3])
        event_stack.append(d[4])
        ev = np.concatenate([d[0], i*np.ones((len(d[0]),1), dtype=np.float32)],1)
        inv_ev = np.concatenate([d[5], i*np.ones((len(d[5]),1), dtype=np.float32)],1)
        # print(ev.shape)
        # events.append(ev.T)
        events.append(ev)
        inv_events.append(inv_ev)    
    events = torch.from_numpy(np.concatenate(events,0))
    inv_events = torch.from_numpy(np.concatenate(inv_events,0))
    labels = default_collate(labels)
    indexs = default_collate(indexs)
    images = default_collate(images)
    event_stack = default_collate(event_stack)
    return events, labels, indexs, images, event_stack, inv_events

def collate_events_proto(data):
    labels = []
    events = []
    indexs = []
    images = []
    event_stack = []
    image_path = []
    for i, d in enumerate(data):
        labels.append(d[1])
        indexs.append(d[2])
        images.append(d[3])
        event_stack.append(d[4])
        image_path.append(d[5])
        ev = np.concatenate([d[0], i*np.ones((len(d[0]),1), dtype=np.float32)],1)
        # print(ev.shape)
        # events.append(ev.T)
        events.append(ev)
    events = torch.from_numpy(np.concatenate(events,0))
    labels = default_collate(labels)
    indexs = default_collate(indexs)
    images = default_collate(images)
    event_stack = default_collate(event_stack)
    return events, labels, indexs, images, event_stack, image_path