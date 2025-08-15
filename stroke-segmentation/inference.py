import torch
import albumentations as A
import matplotlib.pyplot as plt
import numpy as np
import configparser
from torch import nn
from MRIDataset import MRIDataset2
from torchmetrics.classification import JaccardIndex, BinaryAUROC
from torchmetrics.segmentation import DiceScore
from segmentation_models_pytorch import Unet as Unet_smp
from segmentation_models_pytorch import UnetPlusPlus as Unetpp_smp
from torch.utils.data import DataLoader
from albumentations.pytorch import ToTensorV2
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from sklearn.metrics import recall_score, f1_score


def main():
    config = configparser.ConfigParser()
    config.read("../config.ini")
    random_seed = 42
    torch.manual_seed(random_seed)
    np.random.seed(random_seed)
    torch.cuda.manual_seed_all(random_seed)
    TEST_DATASET_PATH = config['DATASET']['test_dataset']
    BATCH_SIZE = int(config['INFERENCE']['batch_size'])
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


    additional_transforms = A.Compose([
        A.Resize(256, 256),
        # A.Normalize(normalization = "min_max"),
        ToTensorV2(),
    ])
    # Создание валидационного датасета и загрузчика данных
    test_dataset = MRIDataset2(path = TEST_DATASET_PATH, augment = False, transform = additional_transforms)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, drop_last=False, shuffle = True)

    model = Unet_smp(encoder_name = config['INFERENCE']['encoder_name'], in_channels = 2)

    model.to(DEVICE)

    # Загрузка сохранённых весов
    checkpoint = torch.load(config['INFERENCE']['model_weights'], map_location=DEVICE)
    model.load_state_dict(checkpoint['state_dict'])
    model.eval()

    print("Модель успешно загружена и готова к использованию.")

    # Инициализация метрик
    jaccard_metric = JaccardIndex(task='binary', average = 'macro').to(DEVICE)
    val_dice = DiceScore(num_classes = 2, average = 'macro').to(DEVICE)

    # Итерация по валидационному датасету
    for idx, (x, y_true) in enumerate(test_loader):
        x = x.to(DEVICE)
        y_true = y_true.to(DEVICE)

        with torch.no_grad():
            with torch.autocast(device_type='cuda', dtype=torch.float16):
                y_pred = model(x)

            y_pred_probs = torch.sigmoid(y_pred)
            y_pred_labels = (y_pred_probs > 0.5).int()
            y_true = y_true.int()

        jaccard_metric.update(y_pred_labels, y_true)
        val_dice.update(y_pred_labels, y_true)

    print(f"{jaccard_metric.compute().item()=}")
    print(f"{val_dice.compute().item()=}")


if __name__ == "__main__":
    main()
