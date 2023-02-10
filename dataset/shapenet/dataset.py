from torch.utils.data import Dataset, DataLoader, TensorDataset
import os
import torch
import numpy as np
import sys
from tqdm import tqdm 
import json
from plyfile import PlyData, PlyElement
import pytorch_lightning as pl 

def get_segmentation_classes(root):
    catfile = os.path.join(root, 'synsetoffset2category.txt')
    cat = {}
    meta = {}

    with open(catfile, 'r') as f:
        for line in f:
            ls = line.strip().split()
            cat[ls[0]] = ls[1]

    for item in cat:
        dir_seg = os.path.join(root, cat[item], 'points_label')
        dir_point = os.path.join(root, cat[item], 'points')
        fns = sorted(os.listdir(dir_point))
        meta[item] = []
        for fn in fns:
            token = (os.path.splitext(os.path.basename(fn))[0])
            meta[item].append((os.path.join(dir_point, token + '.pts'), os.path.join(dir_seg, token + '.seg')))
    
    with open(os.path.join(os.path.dirname(os.path.realpath(__file__)), '../misc/num_seg_classes.txt'), 'w') as f:
        for item in cat:
            datapath = []
            num_seg_classes = 0
            for fn in meta[item]:
                datapath.append((item, fn[0], fn[1]))

            for i in tqdm(range(len(datapath))):
                l = len(np.unique(np.loadtxt(datapath[i][-1]).astype(np.uint8)))
                if l > num_seg_classes:
                    num_seg_classes = l

            print("category {} num segmentation classes {}".format(item, num_seg_classes))
            f.write("{}\t{}\n".format(item, num_seg_classes))

class DataModule(pl.LightningDataModule):
    def __init__(self, task, data_args):
        super().__init__()
        self.data_dir = data_args.data_dir
        self.task = task
        self.batch_size = data_args.batch_size
        self.num_workers = data_args.num_workers
        self.num_points = data_args.num_points
        self.class_choice = data_args.class_choice
        self.data_augmentation = data_args.data_augmentation
        
        self.train_dataset = ShapenetDataset(data_dir = self.data_dir, 
                                             num_points = self.num_points, 
                                             task = self.task,
                                             class_choice = self.class_choice, 
                                             split = 'train', 
                                             data_augmentation = self.data_augmentation)
        self.val_dataset = ShapenetDataset(data_dir = self.data_dir, 
                                           num_points =  self.num_points, 
                                           task = self.task,
                                           class_choice = self.class_choice, 
                                           split = 'val', 
                                           data_augmentation = self.data_augmentation)
        self.test_dataset = ShapenetDataset(data_dir = self.data_dir, 
                                            num_points = self.num_points, 
                                            task = self.task,
                                            class_choice = self.class_choice, 
                                            split = 'test', 
                                            data_augmentation = self.data_augmentation)
    
    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size = self.batch_size, num_workers=self.num_workers, shuffle=True) # , collate_fn = lambda x:x

    def val_dataloader(self):
        return DataLoader(self.val_dataset, batch_size = self.batch_size, num_workers=self.num_workers, shuffle=False)
    
    def test_dataloader(self):
        return DataLoader(self.test_dataset, batch_size = self.batch_size, num_workers=self.num_workers, shuffle=False, drop_last=False)


class ShapenetDataset(Dataset):
    def __init__(self, data_dir, num_points=2500, task='cls', class_choice:list[str]=[], split='train', data_augmentation=True):
        self.num_points = num_points
        self.task = task
        self.data_dir = data_dir
        self.catfile = os.path.join(self.data_dir, 'synsetoffset2category.txt')
        self.cat = {}
        self.data_augmentation = data_augmentation
        self.seg_classes = {}
        
        with open(self.catfile, 'r') as f:
            for line in f:
                ls = line.strip().split()
                self.cat[ls[0]] = ls[1]

        if len(class_choice) != 0:
            self.cat = {k: v for k, v in self.cat.items() if k in class_choice}

        self.id2cat = {v: k for k, v in self.cat.items()}

        self.meta = {}
        splitfile = os.path.join(self.data_dir, 'train_test_split', 'shuffled_{}_file_list.json'.format(split))
        #from IPython import embed; embed()
        filelist = json.load(open(splitfile, 'r'))
        for item in self.cat:
            self.meta[item] = []

        for file in filelist:
            _, category, uuid = file.split('/')
            if category in self.cat.values():
                self.meta[self.id2cat[category]].append((os.path.join(self.data_dir, category, 'points', uuid+'.pts'),
                                        os.path.join(self.data_dir, category, 'points_label', uuid+'.seg')))

        self.datapath = []
        for item in self.cat:
            for fn in self.meta[item]:
                self.datapath.append((item, fn[0], fn[1]))

        self.classes = dict(zip(sorted(self.cat), range(len(self.cat))))
        # print(self.classes)
        
        with open('./misc/num_seg_classes.txt', 'r') as f:
            for line in f:
                ls = line.strip().split()
                self.seg_classes[ls[0]] = int(ls[1])
        # self.num_seg_classes = self.seg_classes[list(self.cat.keys())[0]]
        # print(self.seg_classes) # self.num_seg_classes)

    def __getitem__(self, index):
        fn = self.datapath[index]
        cls = self.classes[self.datapath[index][0]]
        point_set = np.loadtxt(fn[1]).astype(np.float32)
        seg = np.loadtxt(fn[2]).astype(np.int64)
        #print(point_set.shape, seg.shape)

        choice = np.random.choice(len(seg), self.num_points, replace=True)
        
        #resample
        point_set = point_set[choice, :]

        point_set = point_set - np.expand_dims(np.mean(point_set, axis = 0), 0) # center
        dist = np.max(np.sqrt(np.sum(point_set ** 2, axis = 1)),0)
        point_set = point_set / dist #scale

        if self.data_augmentation:
            theta = np.random.uniform(0,np.pi*2)
            rotation_matrix = np.array([[np.cos(theta), -np.sin(theta)],[np.sin(theta), np.cos(theta)]])
            point_set[:,[0,2]] = point_set[:,[0,2]].dot(rotation_matrix) # random rotation
            point_set += np.random.normal(0, 0.02, size=point_set.shape) # random jitter

        seg = seg[choice]
        point_set = torch.from_numpy(point_set)
        seg = torch.from_numpy(seg)
        cls = torch.from_numpy(np.array([cls]).astype(np.int64))

        if self.task == 'cls':
            return point_set, cls
        elif self.task == 'seg':
            return point_set, seg

    def __len__(self):
        return len(self.datapath)

if __name__ == '__main__':
    datapath = sys.argv[1]

    d = Dataset(root = datapath, class_choice = ['Chair'])
    print(len(d))
    ps, seg = d[0]
    print(ps.size(), ps.type(), seg.size(),seg.type())

    d = Dataset(root = datapath, classification = True)
    print(len(d))
    ps, cls = d[0]
    print(ps.size(), ps.type(), cls.size(),cls.type())
    # get_segmentation_classes(datapath)