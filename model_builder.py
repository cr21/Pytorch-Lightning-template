import pytorch_lightning as pl
import torch
from torch import nn
import torch.optim as optim

from torch.utils.data import Dataset


from torchmetrics import Accuracy, F1Score
from torchmetrics import Metric
from metrics import MyAccuracy
from config import LR_ADAM



class TinyVGG(pl.LightningModule):
    def __init__(self, input_shape,hidden_units, output_shape):
        super().__init__()
        self.training_step_outputs = []
        self.conv1=nn.Sequential(
            nn.Conv2d(in_channels=input_shape, out_channels=hidden_units, kernel_size=3, stride=1, padding=0),
            nn.ReLU(),
            nn.Conv2d(in_channels=hidden_units, out_channels=hidden_units, kernel_size=3, stride=1, padding=0),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )

        self.conv2=nn.Sequential(
            nn.Conv2d(in_channels=hidden_units, out_channels=hidden_units, kernel_size=3, padding=0),
            nn.ReLU(),
            nn.Conv2d(in_channels=hidden_units, out_channels=hidden_units, kernel_size=3, padding=0),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )

        self.classifier=nn.Sequential(
            nn.Flatten(),
            nn.Linear(in_features=hidden_units*53*53, out_features=output_shape)
        )

        self.loss_fn=nn.CrossEntropyLoss()
        self.accuracy = Accuracy(task='multiclass', threshold=0.5, num_classes=output_shape)
        self.f1_score = F1Score(task='multiclass', threshold=0.5, num_classes=output_shape)
        self.my_acc = MyAccuracy()
        
        
    def forward(self, x):
        x=self.conv1(x)
        x=self.conv2(x)
        x=self.classifier(x)
        return x
    
    def __common_step(self,batch, batch_idx):
        x,y=batch
        out=self(x)
        loss=self.loss_fn(out, y)
        return loss, out,y
    
    def training_step(self,batch, batch_idx):
        loss, out,y = self.__common_step(batch, batch_idx)
        #accuracy = self.accuracy(out,y)
        #f1score = self.f1_score(out, y)
        #my_acc = self.my_acc(out, y)
        self.log_dict({
                        'train_loss':loss
                    },
                    on_epoch=True, 
                    on_step=False,
                    prog_bar=True
        )
        self.training_step_outputs.append({'loss':loss,'outputs':out,'y':y})
        return {'loss':loss,'outputs':out,'y':y}
        # loss, out,y = self.__common_step(batch, batch_idx)
        # accuracy = self.accuracy(out,y)
        # f1score = self.f1_score(out, y)
        # my_acc = self.my_acc(out, y)
        # self.log_dict({
        #                 'train_loss':loss,
        #                 'train_accuracy':accuracy,
        #                 'train_f1_score':f1score,
        #                 'my_accuracy':my_acc
        #             },
        #             on_epoch=True, 
        #             on_step=False,
        #             prog_bar=True
        # )
        # return loss
    
    def on_train_epoch_end(self):
        outs = torch.cat([x['outputs'] for x in self.training_step_outputs])
        y=torch.cat([x['y'] for x in self.training_step_outputs])

        self.log_dict({
            'train_acc':self.my_acc(outs,y),
            'train_f1':self.f1_score(outs,y),
            },
            on_epoch=True, 
            on_step=False,
            prog_bar=True
        )
    def validation_step(self,batch, batch_idx):
        loss, out,y = self.__common_step(batch, batch_idx)
        self.log('val_loss', loss)
        return loss
    
    def test_step(self,batch, batch_idx):
        loss, out,y = self.__common_step(batch, batch_idx)
        self.log('test_loss', loss)
        return loss
    
    def __common_step(self,batch, batch_idx):
        x,y=batch
        out=self(x)
        loss=self.loss_fn(out, y)
        return loss, out,y
    
    def __common_step(self,batch, batch_idx):
        x,y=batch
        out=self(x)
        loss=self.loss_fn(out, y)
        return loss, out,y

    def configure_optimizers(self):
        return optim.Adam(self.parameters(), lr=LR_ADAM)
    
    
    def predict_step(self, batch, batch_idx):
        x,y=batch
        out=self(x)
        preds=torch.argmax(out,dim=1)
        return preds
        
    

    
   