# PointNet.pytorch_lightning
This repo is implementation for PointNet(https://arxiv.org/abs/1612.00593) in pytorch. The model is in `model/pointnet/architecture.py`.

This repository is a conversion of fxia22's pytorch code(https://github.com/fxia22/pointnet.pytorch) to pytorch lightning.

It is tested with pytorch-lightning: 1.7.0.

# Download data

```
git clone https://github.com/cyun9601/pointnet.pytorch_lightning
cd pointnet.pytorch_lightning
pip install -e .
```

Download shapenet and Build visualization tool
```
cd script
bash build.sh # build C++ code for visualization
bash download.sh # download dataset
```

This model has also been implemented for the modelnet40 data for classification task.

To use this data, download modelnet40 and put it in the `data/modelnet40` folder, making class folders inside.

ex) 
train: `data/modelnet40/airplane/train/airplane_0001.off`

test: `data/modelnet40/airplane/test/airplane_0627.off`

# Training

The implementation has been done by putting configuration values in the yaml file. 

The argparse is planned to be added later.

```
python train.py
```

# Testing

The test code has not yet been implemented with Pytorch lightning. This is planned to be done in the future.

# Performance

## Classification performance

On ModelNet40:

|  | Overall Acc | 
| :---: | :---: | 
| Original implementation | 89.2 | 
| this implementation(w/o feature transform) | 86.4 | 
| this implementation(w/ feature transform) | 87.0 | 

On [A subset of shapenet](http://web.stanford.edu/~ericyi/project_page/part_annotation/index.html)

|  | Overall Acc | 
| :---: | :---: | 
| Original implementation | N/A | 
| this implementation(w/o feature transform) | 98.1 | 
| this implementation(w/ feature transform) | 97.7 | 

## Segmentation performance

Segmentation on  [A subset of shapenet](http://web.stanford.edu/~ericyi/project_page/part_annotation/index.html).

| Class(mIOU) | Airplane | Bag| Cap|Car|Chair|Earphone|Guitar|Knife|Lamp|Laptop|Motorbike|Mug|Pistol|Rocket|Skateboard|Table
| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | 
| Original implementation |  83.4 | 78.7 | 82.5| 74.9 |89.6| 73.0| 91.5| 85.9| 80.8| 95.3| 65.2| 93.0| 81.2| 57.9| 72.8| 80.6| 
| this implementation(w/o feature transform) | 73.5 | 71.3 | 64.3 | 61.1 | 87.2 | 69.5 | 86.1|81.6| 77.4|92.7|41.3|86.5|78.2|41.2|61.0|81.1|
| this implementation(w/ feature transform) |  |  |  |  | 87.6 |  | | | | | | | | | |81.0|

Note that this implementation trains each class separately, so classes with fewer data will have slightly lower performance than reference implementation.

Sample segmentation result:

![seg](https://raw.githubusercontent.com/fxia22/pointnet.pytorch/master/misc/show3d.png?token=AE638Oy51TL2HDCaeCF273X_-Bsy6-E2ks5Y_BUzwA%3D%3D)

# Links

- [Project Page](http://stanford.edu/~rqi/pointnet/)
- [Tensorflow implementation](https://github.com/charlesq34/pointnet)
- [Pytorch implementation](https://github.com/fxia22/pointnet.pytorch)

