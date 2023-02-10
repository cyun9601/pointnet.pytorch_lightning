import os
import sys 

prj_dir = os.path.dirname(__file__)
os.chdir(prj_dir)
sys.path.append(prj_dir)

import importlib
from datetime import datetime, timezone, timedelta

from shutil import copyfile
import wandb
import pytorch_lightning as pl 
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping
from pytorch_lightning.loggers import WandbLogger

from modules.utils import get_logger, load_config

# parser = argparse.ArgumentParser()
# parser.add_argument('--batch_size', type=int, default=32, help='input batch size')
# parser.add_argument('--num_points', type=int, default=2500, help='number of sample points')
# parser.add_argument('--workers', type=int, help='number of data loading workers', default=4)
# parser.add_argument('--max_epoch', type=int, default=250, help='number of epochs to train for')
# parser.add_argument('--save_dir', type=str, default='./result', help='output folder')
# parser.add_argument('--pretrain_path', type=str, default='', help='pretrained path')
# parser.add_argument('--model', type=str, default='pointnet', help='model name')
# parser.add_argument('--data_dir', type=str, required=True, help="dataset path")
# parser.add_argument('--dataset_type', type=str, default='shapenet', help="dataset type shapenet|modelnet40")
# parser.add_argument('--feature_transform', action='store_true', help="use feature transform")
# args = parser.parse_args()
# print(args)

PRJ_DIR = os.path.dirname(os.path.realpath(__file__))

# config 
CONFIG_PATH = os.path.join(PRJ_DIR, 'config/train.yaml')
args = load_config(CONFIG_PATH)

# debug mode
if args.mode == 'debug':
    model_name = 'debug'
    args['train']['min_epochs'] = 1
    args['train']['max_epochs'] = 2
    
elif args.mode == 'train':
    # Train serial
    kst = timezone(timedelta(hours=9))
    serial_number = datetime.now(tz=kst).strftime("%Y%m%d_%H%M%S")
    model_name = f'{args.model}-' + serial_number

print(f'Model name: {model_name}')
RESULT_DIR = os.path.join(PRJ_DIR, f'results/train/{args.model}/{model_name}/')
os.makedirs(RESULT_DIR, exist_ok=True)

#-- save the config 
copyfile(CONFIG_PATH, RESULT_DIR + '/train.yaml')
pl.seed_everything(args.seed)

#-- Logger
# Set logger
offline_logger = get_logger(name='train', dir_='./', stream=False)
offline_logger.info(f"Set Logger {RESULT_DIR}")

#-- train
print(f'Train Start')

# wandb logger
if 'wandb_logger' in globals():
    wandb.finish()
wandb_logger = WandbLogger(project = 'PointNet_test', entity = 'cychoi', name = model_name)
wandb_logger.experiment.config.update(dict(args))

# blue = lambda x: '\033[94m' + x + '\033[0m'

## checkpoint callback
checkpoint_callback = ModelCheckpoint(
    dirpath = os.path.join(RESULT_DIR, 'checkpoints/'), # 체크포인트 저장 위치와 이름 형식 지정
    filename = 'best_param', # '{epoch:d}'  
    verbose = True, # 체크포인트 저장 결과 출력
    save_last = True, # 마지막 체크포인트 저장
    # save_top_k = SAVE_TOP_K, # 최대 몇 개의 체크포인트를 저장할 지 지정. save_last에 의해 저장되는 체크포인트는 제외 
    monitor = 'val loss', # 어떤 metric을 기준으로 체크포인트를 저장할 지 결정
    mode = 'min' # 지정한 metric의 어떤 기준으로 체크포인트를 저장할 지 지정
)

## early stopping callback 
early_stopping_callback = EarlyStopping(
    monitor = 'val loss', # 모니터링할 Metric을 지정
    patience = args.callbacks.patience,
    verbose = True, # 진행 결과 출력 
    mode = 'min' # metric을 어떤 기준으로 성능을 측정할 지 결정         
)

if args.data.dataset_type in ['shapenet', 'modelnet40']:
    data_module_class = getattr(importlib.import_module(f'dataset.{args.data.dataset_type}.dataset'), 'DataModule')
    data_module = data_module_class(args.task, args.data)
else:
    exit('wrong dataset type')

print(len(data_module.train_dataset), len(data_module.val_dataset), len(data_module.test_dataset))
num_classes = len(data_module.train_dataset.classes)
print('classes', num_classes)

try:
    os.makedirs(RESULT_DIR, exist_ok=True)
except OSError:
    pass

model_module_class = getattr(importlib.import_module(f'model.{args.model}.module'), 'LightningModel')
model_module = model_module_class(task = args.task, lr = args.train.lr, pretrain_path = args.train.pretrain_path, num_classes=num_classes, feature_transform=args.train.feature_transform)

trainer_args = {
    'devices': args.train.devices,
    'accelerator': args.train.accelerator,
    'min_epochs': args.train.min_epochs,
    'max_epochs': args.train.max_epochs,
    'callbacks': [checkpoint_callback, early_stopping_callback],
    'logger': wandb_logger
}

trainer = pl.Trainer(**trainer_args)
trainer.fit(model_module, data_module)