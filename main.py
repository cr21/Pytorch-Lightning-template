import os
from pt_lightning.model_builder import TinyVGG
import pytorch_lightning as pl
from pt_lightning.dataset import FashionDataModule
from torchvision import transforms
from pt_lightning.config import CUDA_VISIBLE_DEVICES, INPUT_CHANNEL, NUM_CLASSES,\
                HIDDEN_UNITS, IMG_W, IMG_H, BATCH_SIZE, NUM_WORKERS,DATA_DIR,\
                DEVICES, MIN_EPOCH, MAX_EPOCH, ACCELERATOR

from pt_lightning.callbacks import CustomPrintCallBack
from pytorch_lightning.callbacks import EarlyStopping

from pytorch_lightning.loggers import TensorBoardLogger

if __name__=='__main__':
    logger=TensorBoardLogger(save_dir='tb_logs',name='fashion_dataset_tiny_vgg_epoch5_earlystopping_bs_32')
    model=TinyVGG(input_shape=INPUT_CHANNEL,
                   hidden_units=HIDDEN_UNITS,
                    output_shape=NUM_CLASSES)
    manual_transform = transforms.Compose([
            transforms.Resize((IMG_H,IMG_W)),
            transforms.ToTensor()
        ])
    dm=FashionDataModule(data_dir=DATA_DIR,
                     batch_size=BATCH_SIZE,
                     num_workers=NUM_WORKERS,
                     transforms=manual_transform,
                     test_transforms =manual_transform)
    os.environ["CUDA_VISIBLE_DEVICES"]=str(CUDA_VISIBLE_DEVICES)
    trainer=pl.Trainer(accelerator=ACCELERATOR,
                    logger=logger,
                    devices=DEVICES,
                    min_epochs=MIN_EPOCH,
                    max_epochs=MAX_EPOCH,
                    enable_progress_bar=True,
                    callbacks=[
                                CustomPrintCallBack(),
                                EarlyStopping(monitor='val_loss')
                            ]
                    )
    trainer.fit(model,dm )
    trainer.validate(model,dm )
    trainer.test(model,dm )