from pytorch_lightning import LightningModule, Trainer
from pytorch_lightning.callbacks  import Callback

class PrintCallBack(Callback):
    def __init__(self):
        super().__init__()
    
    def on_train_epoch_start(self, trainer: Trainer, pl_module: LightningModule) -> None:
        print(f"Starting Training !!! {trainer.current_epoch}")
        print("^"*50)

    def on_train_epoch_end(self, trainer: Trainer, pl_module: LightningModule) -> None:
        print(f"Finished Training !!! {trainer.current_epoch}")
        print("^"*50)

class CustomPrintCallBack(Callback):
    def __init__(self):
        super().__init__()
    
    def on_train_epoch_start(self, trainer: Trainer, pl_module: LightningModule) -> None:
        print(f"Starting Training !!! {trainer.current_epoch}")
        print("^"*50)

    def on_train_epoch_end(self, trainer: Trainer, pl_module: LightningModule) -> None:
        print(f"Finished Training !!! {trainer.current_epoch}")
        print("^"*50)