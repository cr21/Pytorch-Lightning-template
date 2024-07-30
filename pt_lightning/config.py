import os
CUDA_VISIBLE_DEVICES=0
INPUT_CHANNEL=3
HIDDEN_UNITS=10
NUM_CLASSES=5
IMG_H=224
IMG_W=224
DATA_DIR='/kaggle/input/product-data/Product3'
DATA_DIR='/Users/chiragtagadiya/Downloads/DeepLearning Projects/Buy Me That Look/Product3'
BATCH_SIZE=32
NUM_WORKERS=os.cpu_count()
ACCELERATOR='cpu'
DEVICES=1
MIN_EPOCH=1
MAX_EPOCH=5
LR_ADAM=0.001
