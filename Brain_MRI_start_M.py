# Brain_MRI_start.py

import random
import time
import albumentations as A
import numpy as np
import torch
import h5py
import torchvision

from albumentations.pytorch import ToTensorV2

from segmentation_models_pytorch import UnetPlusPlus as UnetPlusPlus_smp
from segmentation_models_pytorch import Unet as Unet_smp
from segmentation_models_pytorch import Linknet
from torch.optim.lr_scheduler import (
    ReduceLROnPlateau,
    StepLR,
)
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
# Импортируем классы метрик и SummaryWriter
from torchmetrics.classification import JaccardIndex
from torchmetrics.segmentation import DiceScore

from tqdm.auto import tqdm
from segmentation_models_pytorch.losses import (
    DiceLoss,
    JaccardLoss,
    LovaszLoss,
    FocalLoss,
    TverskyLoss
)
from torchvision import models
from torch import nn, optim
from typing import Callable
from MRIDataset import MRIDataset, MRIDataset2
import os
from segmentation_models_pytorch import UnetPlusPlus as Unetpp_smp


# Путь к вашему файлу с данными
DATASET = "./data/train_dataset.h5"
VAL_DATASET = "./data/val_dataset.h5"

BATCH_SIZE = 8
torch.autograd.set_detect_anomaly(True)

random.seed(42)
np.random.seed(42)
torch.manual_seed(42)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(42)



train_additional_transforms = None

# Определение дополнительных трансформаций, если необходимо
additional_transforms = A.Compose([
    A.Resize(256, 256),
    # A.Normalize(normalization = "min_max_per_channel"),
    ToTensorV2(),
])


# Определение устройства (CPU или GPU)
device = torch.device("cuda:0")
# torch.backends.cudnn.benchmark = True

# Вывод информации об устройстве
print(f"Using CUDA device: {torch.cuda.current_device()} - {torch.cuda.get_device_name(device)}")


dataset = MRIDataset2(path = DATASET, augment = True)
val_dataset = MRIDataset2(path = VAL_DATASET, augment = False, transform = additional_transforms)

train_loader = DataLoader(dataset, batch_size=BATCH_SIZE, drop_last=True, shuffle = True)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, drop_last=False, shuffle = True)

net = Unet_smp(encoder_name = 'efficientnet-b2', in_channels = 2)
net.to(device)

# Определение функции потерь и оптимизатора
criterion = JaccardLoss(mode = 'binary')
optimizer = torch.optim.Adam(  # NadamW
    net.parameters(),
    lr=0.0025, weight_decay=2e-4,
)

# scheduler = StepLR(optimizer, step_size=5, gamma=0.8)

n_epochs = 300

print(f'Number of parameters: {sum(p.numel() for p in net.parameters()):,}')

start_time = time.monotonic()

# Инициализация метрик
train_jaccard = JaccardIndex(task='binary', average = 'macro').to(device)
val_jaccard = JaccardIndex(task='binary', average = 'macro').to(device)

train_dice = DiceScore(num_classes = 2, average = 'macro').to(device)
val_dice = DiceScore(num_classes = 2, average = 'macro').to(device)

# Инициализация SummaryWriter для TensorBoard
writer = SummaryWriter()

# Обучение
best_loss = float('inf')
ebar = tqdm(total=n_epochs, desc="Epochs", leave=False)

for epoch in range(n_epochs):
    net.train()
    train_loss = 0
    ibar = tqdm(total=len(train_loader), desc="Training", leave=False)
    for x, y_true in train_loader:
        x, y_true = x.to(device), y_true.to(device)

        optimizer.zero_grad()
        y_pred = net(x)
        loss = criterion(y_pred, y_true)
        loss.backward()
        optimizer.step()
        train_loss += loss.item()

        with torch.no_grad():
            y_pred_probs = torch.sigmoid(y_pred)
            y_pred_labels = (y_pred_probs > 0.5).int()
            y_true_labels = y_true.int()

            train_jaccard.update(y_pred_labels, y_true_labels)
            train_dice.update(y_pred_labels, y_true_labels)
        ibar.update(1)
    ibar.close()
    
    train_jaccard_score = train_jaccard.compute().item()
    train_dice_score = train_dice.compute().item()


    train_jaccard.reset()
    train_dice.reset()

    # val dataset inference
    net.eval()
    val_loss = 0
    ibar = tqdm(total=len(val_loader), desc="Validation", leave=False)
    with torch.no_grad():
        for x, y_true in val_loader:
            x, y_true = x.to(device), y_true.to(device)
            y_pred = net(x)
            loss = criterion(y_pred, y_true)
            val_loss += loss.item()

            y_pred_probs = torch.sigmoid(y_pred)
            y_pred_labels = (y_pred_probs > 0.5).int()
            y_true_labels = y_true.int()

            # Обновление метрик сегментации
            val_jaccard.update(y_pred_labels, y_true_labels)
            val_dice.update(y_pred_labels, y_true_labels)

            ibar.update(1)
    ibar.close()
    # Вычисляем метрики
    val_jaccard_score = val_jaccard.compute().item()
    val_dice_score = val_dice.compute().item()

    val_jaccard.reset()
    val_dice.reset()


    writer.add_scalar('Loss/Train', train_loss / len(train_loader), epoch)
    writer.add_scalar('Loss/Validation', val_loss / len(val_loader), epoch)
    writer.add_scalar('IoU/Train', train_jaccard_score, epoch)
    writer.add_scalar('IoU/Validation', val_jaccard_score, epoch)

    # Обновление прогресс-бара
    ebar.set_description(
        f"Epoch {epoch + 1}/{n_epochs} - Train Loss: {train_loss / len(train_loader):.03f} - Val Loss: {val_loss / len(val_loader):.03f} - Val IoU: {val_jaccard_score:.3f} - Train IoU: {train_jaccard_score:.3f} - Train Dice: {train_dice_score:.3f} - Val Dice: {val_dice_score:.3f}"
    )
    ebar.update(1)

    if val_loss / len(val_loader) <= best_loss:
        best_loss = val_loss / len(val_loader)
        torch.save({
            'epoch': epoch,
            'state_dict': net.state_dict(),
            'optimizer_state_dict': optimizer.state_dict()
            },
            'unet_model.pt'
        )
    # Сохранение модели каждые 20 эпох
    if (epoch + 1) % 20 == 0:
        torch.save({
            'epoch': epoch,
            'state_dict': net.state_dict(),
            'optimizer_state_dict': optimizer.state_dict()
            },
            f'unet_model_{epoch + 1}.pt'
        )