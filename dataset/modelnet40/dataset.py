from torch.utils.data import Dataset, DataLoader, TensorDataset
import os
import os.path
import torch
import numpy as np
import sys
from tqdm import tqdm 
import json
from plyfile import PlyData, PlyElement
import pytorch_lightning as pl 

def gen_modelnet_id(root):
    classes = []
    with open(os.path.join(root, 'train.txt'), 'r') as f:
        for line in f:
            classes.append(line.strip().split('/')[0])
    classes = np.unique(classes)
    with open(os.path.join(os.path.dirname(os.path.realpath(__file__)), '../misc/modelnet_id.txt'), 'w') as f:
        for i in range(len(classes)):
            f.write('{}\t{}\n'.format(classes[i], i))

class DataModule(pl.LightningDataModule):
    def __init__(self, data_args):
        super().__init__()
        self.data_dir = data_args.data_dir
        self.batch_size = data_args.batch_size
        self.num_workers = data_args.num_workers
        self.num_points = data_args.num_points
        self.class_choice = data_args.class_choice
        self.data_augmentation = data_args.data_augmentation
        
        self.train_dataset = Modelnet40Dataset(data_dir = self.data_dir, 
                                             num_points = self.num_points, 
                                             class_choice = self.class_choice, 
                                             split = 'train', 
                                             data_augmentation = self.data_augmentation)
        self.val_dataset = Modelnet40Dataset(data_dir = self.data_dir, 
                                           num_points =  self.num_points, 
                                           class_choice = self.class_choice, 
                                           split = 'val', 
                                           data_augmentation = self.data_augmentation)
        self.test_dataset = Modelnet40Dataset(data_dir = self.data_dir, 
                                            num_points = self.num_points, 
                                            class_choice = self.class_choice, 
                                            split = 'test', 
                                            data_augmentation = self.data_augmentation)
    
    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size = self.batch_size, num_workers=self.num_workers, shuffle=True) # , collate_fn = lambda x:x

    def val_dataloader(self):
        return DataLoader(self.val_dataset, batch_size = self.batch_size, num_workers=self.num_workers, shuffle=False)
    
    def test_dataloader(self):
        return DataLoader(self.test_dataset, batch_size = self.batch_size, num_workers=self.num_workers, shuffle=False, drop_last=False)




class Modelnet40Dataset(Dataset):
    def __init__(self, data_dir, num_points=2500, split='train', data_augmentation=True):
        self.num_points = num_points
        self.data_dir = data_dir
        self.split = split
        self.data_augmentation = data_augmentation
        self.fns = []
        with open(os.path.join(data_dir, '{}.txt'.format(self.split)), 'r') as f:
            for line in f:
                self.fns.append(line.strip())

        self.cat = {}
        with open(os.path.join(os.path.dirname(os.path.realpath(__file__)), '../misc/modelnet_id.txt'), 'r') as f:
            for line in f:
                ls = line.strip().split()
                self.cat[ls[0]] = int(ls[1])

        print(self.cat)
        self.classes = list(self.cat.keys())

    def __getitem__(self, index):
        fn = self.fns[index]
        cls = self.cat[fn.split('/')[0]]
        with open(os.path.join(self.data_dir, fn), 'rb') as f:
            plydata = PlyData.read(f)
        pts = np.vstack([plydata['vertex']['x'], plydata['vertex']['y'], plydata['vertex']['z']]).T
        choice = np.random.choice(len(pts), self.num_points, replace=True)
        point_set = pts[choice, :]

        point_set = point_set - np.expand_dims(np.mean(point_set, axis=0), 0)  # center
        dist = np.max(np.sqrt(np.sum(point_set ** 2, axis=1)), 0)
        point_set = point_set / dist  # scale

        if self.data_augmentation:
            theta = np.random.uniform(0, np.pi * 2)
            rotation_matrix = np.array([[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]])
            point_set[:, [0, 2]] = point_set[:, [0, 2]].dot(rotation_matrix)  # random rotation
            point_set += np.random.normal(0, 0.02, size=point_set.shape)  # random jitter

        point_set = torch.from_numpy(point_set.astype(np.float32))
        cls = torch.from_numpy(np.array([cls]).astype(np.int64))
        return point_set, cls

    def __len__(self):
        return len(self.fns)
    
    
if __name__ == '__main__':
    datapath = sys.argv[1]

    gen_modelnet_id(datapath)
    d = Dataset(root=datapath)
    print(len(d))
    print(d[0])