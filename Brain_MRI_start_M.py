# Brain_MRI_start.py
import datetime
import random
import time
import albumentations as A
import numpy as np
import torch
import h5py
import torchvision
import optuna
import os

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
from segmentation_models_pytorch import UnetPlusPlus as Unetpp_smp
from optuna.trial import TrialState


# Путь к вашему файлу с данными
DATASET = "./data/train_dataset.h5"
VAL_DATASET = "./data/val_dataset_.h5"
BATCH_SIZE = 12

torch.autograd.set_detect_anomaly(False)

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
device = torch.device("cuda")
# torch.backends.cudnn.benchmark = True

# Вывод информации об устройстве
print(f"Using CUDA device: {torch.cuda.current_device()} - {torch.cuda.get_device_name(device)}")


dataset = MRIDataset2(path = DATASET, augment = True)
val_dataset = MRIDataset2(path = VAL_DATASET, augment = False, transform = additional_transforms)

train_loader = DataLoader(dataset, batch_size=BATCH_SIZE, drop_last=True, shuffle = True)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, drop_last=False, shuffle = True)

def objective(trial):
    encoder_name = trial.suggest_categorical("encoder_name", [
        "efficientnet-b3",
        "efficientnet-b4",
        "se_resnet50",
        "mit_b0",
        "mit_b1",
    ])
    net = Unet_smp(encoder_name = encoder_name, in_channels = 2)
    net.to(device)
    # Определение функции потерь и оптимизатора
    criterion = JaccardLoss(mode = 'binary')
    optimizer_name = trial.suggest_categorical("optimizer", ["Adam", "SGD", "NAdam"])
    lr = trial.suggest_float("lr", 1e-5, 3e-2, log=True)
    weight_decay = trial.suggest_float("weight_decay", 5e-5, 2e-2, log=True)
    optimizer = getattr(optim, optimizer_name)(net.parameters(), lr=lr, weight_decay=weight_decay) 
    # scheduler = StepLR(optimizer, step_size=5, gamma=0.8)

    n_epochs = 30

    print(f'Number of parameters: {sum(p.numel() for p in net.parameters()):,}')

    start_time = time.time()
    print(start_time)
    os.mkdir(f'./models_weights/{start_time}')
    # Инициализация метрик
    train_jaccard = JaccardIndex(task='binary', average = 'macro').to(device)
    val_jaccard = JaccardIndex(task='binary', average = 'macro').to(device)

    # train_dice = DiceScore(num_classes = 2, average = 'macro').to(device)
    # val_dice = DiceScore(num_classes = 2, average = 'macro').to(device)

    # Инициализация SummaryWriter для TensorBoard
    writer = SummaryWriter()
    # Обучение
    best_loss = float('inf')
    ebar = tqdm(total=n_epochs, desc="Epochs", leave=False)
    scaler = torch.amp.GradScaler("cuda", enabled = True)
    for epoch in range(n_epochs):
        net.train()
        train_loss = 0
        ibar = tqdm(total=len(train_loader), desc="Training", leave=False)
        for x, y_true in train_loader:
            x, y_true = x.to(device), y_true.to(device)
            optimizer.zero_grad(set_to_none = True)
            # optimizer.zero_grad()
            with torch.autocast(device_type = "cuda", dtype = torch.float16, enabled = True):
                y_pred = net(x)
                loss = criterion(y_pred, y_true)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            train_loss += loss.item()

            with torch.no_grad():
                y_pred_probs = torch.sigmoid(y_pred)
                y_pred_labels = (y_pred_probs > 0.5).int()
                y_true_labels = y_true.int()

                train_jaccard.update(y_pred_labels, y_true_labels)
                # train_dice.update(y_pred_labels, y_true_labels)
            ibar.update(1)
        ibar.close()

        train_jaccard_score = train_jaccard.compute().item()
        # train_dice_score = train_dice.compute().item()


        train_jaccard.reset()
        # train_dice.reset()

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
                # val_dice.update(y_pred_labels, y_true_labels)

                ibar.update(1)
        ibar.close()
        # Вычисляем метрики
        val_jaccard_score = val_jaccard.compute().item()
        val_jaccard.reset()
        # val_dice_score = val_dice.compute().item()
        if trial.should_prune():
            raise optuna.exceptions.TrialPruned()
        # val_dice.reset()

        trial.report(val_loss / len(val_loader), epoch)

        writer.add_scalar('Loss/Train', train_loss / len(train_loader), epoch)
        writer.add_scalar('Loss/Validation', val_loss / len(val_loader), epoch)
        writer.add_scalar('IoU/Train', train_jaccard_score, epoch)
        writer.add_scalar('IoU/Validation', val_jaccard_score, epoch)

        # Обновление прогресс-бара
        ebar.set_description(
            f"Epoch {epoch + 1}/{n_epochs} - Train Loss: {train_loss / len(train_loader):.03f} - Val Loss: {val_loss / len(val_loader):.03f} - Val IoU: {val_jaccard_score:.3f} - Train IoU: {train_jaccard_score:.3f}"
        )
        ebar.update(1)

        if val_loss / len(val_loader) <= best_loss:
            best_loss = val_loss / len(val_loader)
            torch.save({
                'epoch': epoch,
                'state_dict': net.state_dict(),
                'optimizer_state_dict': optimizer.state_dict()
                },
                f'./models_weights/{start_time}/best_{encoder_name}.pt'
            )
        # Сохранение модели каждые 20 эпох
        if (epoch + 1) % 5 == 0:
            torch.save({
                'epoch': epoch,
                'state_dict': net.state_dict(),
                'optimizer_state_dict': optimizer.state_dict()
                },
                f'./models_weights/{start_time}/{encoder_name}_{epoch + 1}.pt'
            )
    return val_loss / len(val_loader)

if __name__ == "__main__":
    study = optuna.create_study(direction="minimize")
    study.optimize(objective, n_trials=30)

    pruned_trials = study.get_trials(deepcopy=False, states=[TrialState.PRUNED])
    complete_trials = study.get_trials(deepcopy=False, states=[TrialState.COMPLETE])

    print("Study statistics: ")
    print("  Number of finished trials: ", len(study.trials))
    print("  Number of pruned trials: ", len(pruned_trials))
    print("  Number of complete trials: ", len(complete_trials))

    print("Best trial:")
    trial = study.best_trial

    print("  Value: ", trial.value)

    print("  Params: ")
    for key, value in trial.params.items():
        print("    {}: {}".format(key, value))
