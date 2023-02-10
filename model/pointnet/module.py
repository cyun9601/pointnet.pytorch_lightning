import torch
import torch.nn.parallel
import torch.utils.data
import torch.nn.functional as F
import pytorch_lightning as pl 
from torch.optim.lr_scheduler import StepLR
from .architecture import PointNet

def feature_transform_regularizer(trans):
    d = trans.size()[1]
    batch_size = trans.size()[0]
    I = torch.eye(d)[None, :, :]
    if trans.is_cuda:
        I = I.cuda()
    loss = torch.mean(torch.norm(torch.bmm(trans, trans.transpose(2,1)) - I, dim=(1,2)))
    return loss

class LightningModel(pl.LightningModule):
    def __init__(self, task, lr, pretrain_path = '', num_classes = 2, feature_transform=False):
        super().__init__()
        
        self.task = task
        self.lr = lr
        self.num_classes = num_classes
        self.feature_transform = feature_transform
        
        self.pointnet = PointNet(self.task, self.num_classes, feature_transform=self.feature_transform)
        if pretrain_path != '':
            self.pointnet.load_state_dict(torch.load(pretrain_path)) 
        
    def forward(self, points):
        points = points.transpose(2, 1)
        pred, trans, trans_feat = self.pointnet(points)
        return pred, trans, trans_feat
        
    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.lr, betas=(0.9, 0.999))
        scheduler = {
            "scheduler": StepLR(optimizer, step_size=20, gamma = 0.5),
            "interval": "epoch",
        }
        return [optimizer], [scheduler]

    def training_step(self, batch, batch_idx):
        points, target = batch
        pred, trans, trans_feat = self(points)
        
        if self.task == 'cls':
            target = target[:, 0]
        elif self.task == 'seg':
            pred = pred.view(-1, self.num_classes)
            target = target.view(-1, 1)[:, 0] - 1
        
        loss = F.nll_loss(pred, target)
        if self.feature_transform:
            loss += feature_transform_regularizer(trans_feat) * 0.001
            
        pred_choice = pred.data.max(1)[1]
        acc = pred_choice.eq(target.data).cpu().sum()
        return {"loss": loss, "acc": acc} #, pred
    
    def training_epoch_end(self, training_step_outputs) -> None:
        loss = torch.hstack([output['loss'] for output in training_step_outputs]).float().mean()
        acc = torch.hstack([output['acc'] for output in training_step_outputs]).float().mean()
        self.log('train loss', loss)
        self.log('train acc', acc)

    def validation_step(self, batch, batch_idx):
        points, target = batch
        # target = target[:, 0]
        pred, trans, trans_feat = self(points)
        
        if self.task == 'cls':
            target = target[:, 0]
        elif self.task == 'seg':
            pred = pred.view(-1, self.num_classes)
            target = target.view(-1, 1)[:, 0] - 1
        
        loss = F.nll_loss(pred, target)
        if self.feature_transform:
            loss += feature_transform_regularizer(trans_feat) * 0.001
            
        pred_choice = pred.data.max(1)[1]
        acc = pred_choice.eq(target.data).cpu().sum()
            
        return {'loss': loss, 'acc': acc} #, pred
    
    def validation_epoch_end(self, validation_step_outputs) -> None:
        loss = torch.hstack([output['loss'] for output in validation_step_outputs]).float().mean()
        acc = torch.hstack([output['acc'] for output in validation_step_outputs]).float().mean()
        self.log('val loss', loss)
        self.log('val acc', acc)
        
    def test_step(self, batch, batch_idx):
        points, target = batch
        # target = target[:, 0]
        pred, trans, trans_feat = self(points)
        
        if self.task == 'cls':
            target = target[:, 0]
        elif self.task == 'seg':
            pred = pred.view(-1, self.num_classes)
            target = target.view(-1, 1)[:, 0] - 1
        
        loss = F.nll_loss(pred, target)
        if self.feature_transform:
            loss += feature_transform_regularizer(trans_feat) * 0.001
            
        pred_choice = pred.data.max(1)[1]
        acc = pred_choice.eq(target.data).cpu().sum()
            
        return {'loss': loss, 'acc': acc} #, pred
    
    def test_epoch_end(self, test_step_outputs) -> None:
        loss = torch.hstack([output['loss'] for output in test_step_outputs]).float().mean()
        acc = torch.hstack([output['acc'] for output in test_step_outputs]).float().mean()
        self.log('test loss', loss)
        self.log('test acc', acc)
