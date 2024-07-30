 

from torch.utils.data import DataLoader
from torchvision import datasets
import os
import pytorch_lightning as pl
from pathlib import Path
class FashionDataModule(pl.LightningDataModule):
    def __init__(self,data_dir, batch_size, num_workers, transforms, test_transforms ):
        super().__init__()
        self.data_dir=Path(data_dir)
        self.batch_size= batch_size
        self.num_workers= num_workers
        self.transforms=transforms
        self.test_transforms=test_transforms
    def prepare_data(self):
        # download, IO, etc. Useful with shared filesystems
        # only called on 1 GPU/TPU in distributed
        ...

    def setup(self, stage):
        print()
        self.train_dataset=datasets.ImageFolder(root=self.data_dir / 'train',
                                   transform=self.transforms)

        self.valid_dataset=datasets.ImageFolder(root=self.data_dir / 'val',
                                        transform=self.transforms)

        self.test_dataset=datasets.ImageFolder(root=self.data_dir / 'test',
                                        transform=self.test_transforms)


    def train_dataloader(self):
        return DataLoader(self.train_dataset, 
                            batch_size=self.batch_size,
                            pin_memory=True,
                            num_workers=self.num_workers,
                            shuffle=True
                        )

    def val_dataloader(self):
        return DataLoader(self.valid_dataset, 
                            batch_size=self.batch_size,
                            pin_memory=True,
                            num_workers=self.num_workers,
                            shuffle=False
                        )

    def test_dataloader(self):
        return DataLoader(self.test_dataset, 
                            batch_size=self.batch_size,
                            pin_memory=True,
                            num_workers=self.num_workers,
                            shuffle=False
                        )