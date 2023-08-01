import torch
import albumentations as A
from albumentations.pytorch import ToTensorV2

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
TRAIN_DIR = "data/trainA"
VAL_DIR = "data/testA"
BATCH_SIZE = 1
LEARNING_RATE = 2e-4
LAMBDA_IDENTITY = 2.5
LAMBDA_CYCLE = 10
NUM_WORKERS = 6
NUM_EPOCHS = 200
LOAD_MODEL = True
SAVE_MODEL = True
CHECKPOINT_GEN_H = "memory/genh.tar"
CHECKPOINT_GEN_Z = "memory/genz.tar"
CHECKPOINT_CRITIC_H = "memory/critich.tar"
CHECKPOINT_CRITIC_Z = "memory/criticz.tar"

transforms = A.Compose(
    [
        A.Resize(width=256, height=256),
        A.HorizontalFlip(p=0.5),
        A.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5], max_pixel_value=255),
        ToTensorV2(),
     ],
    additional_targets={"image0": "image"},
)