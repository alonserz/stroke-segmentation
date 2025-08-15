# visualize_predictions.py

import torch
import matplotlib.pyplot as plt
import numpy as np
import albumentations as A
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

# Настройки


# Функция для наложения масок на изображение
def overlay_masks(image, true_mask, pred_mask = None, alpha=0.5):
    """
    Наложение истинной и предсказанной масок на изображение.

    Parameters:
        image (torch.Tensor): Исходное изображение (C, H, W).
        true_mask (torch.Tensor): Истинная маска (H, W).
        pred_mask (torch.Tensor): Предсказанная маска (H, W).
        alpha (float): Прозрачность масок.

    Returns:
        overlay (numpy.ndarray): Изображение с наложенными масками.
    """
    # Если изображение имеет 2 канала, усредняем их для визуализации
    if image.shape[0] == 2:
        image_np = image.cpu().numpy()
        image_np = np.mean(image_np, axis=0)  # Среднее по каналам
    else:
        # Если другое количество каналов, адаптируйте под свои данные
        image_np = image.cpu().numpy().transpose(1, 2, 0)

    # Нормализация изображения до [0, 1]
    image_np = (image_np - image_np.min()) / (image_np.max() - image_np.min())
    # Преобразование масок в numpy
    true_mask_np = true_mask.cpu().numpy()
    pred_mask_np = pred_mask.cpu().numpy()

    # Создание цветных масок
    true_mask_color = np.zeros((*true_mask_np.shape, 3))
    pred_mask_color = np.zeros((*pred_mask_np.shape, 3))

    true_mask_color[..., 0] = true_mask_np  # Красный канал для истинной маски
    pred_mask_color[..., 2] = pred_mask_np  # Синий канал для предсказанной маски

    # Накладываем маски на изображение
    overlay = np.stack([image_np, image_np, image_np], axis=-1)  # Конвертируем в RGB

    overlay = np.clip(overlay + alpha * true_mask_color + alpha * pred_mask_color, 0, 1)

    return overlay

def main():
    # Установка случайных семян для воспроизводимости
    random_seed = 42
    torch.manual_seed(random_seed)
    np.random.seed(random_seed)
    torch.cuda.manual_seed_all(random_seed)

    config = configparser.ConfigParser()
    config.read("../config.ini")
    # Создание валидационного датасета и загрузчика данных
    TEST_DATASET_PATH = config['DATASET']['test_dataset']
    BATCH_SIZE = 1
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    additional_transforms = A.Compose([
        A.Resize(256, 256),
        # A.Normalize(normalization = "min_max"),
        ToTensorV2(),
    ])
    # Создание валидационного датасета и загрузчика данных
    test_dataset = MRIDataset2(path = TEST_DATASET_PATH, augment = False, transform = additional_transforms)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, drop_last=False, shuffle = True)

    model = Unet_smp(encoder_name = dataset['INFERENCE']['encoder_name'], in_channels = 2)

    model.to(DEVICE)

    # Загрузка сохранённых весов
    checkpoint = torch.load(dataset['INFERENCE']['model_weights'], map_location=DEVICE)
    model.load_state_dict(checkpoint['state_dict'])
    model.eval()

    print("Модель успешно загружена и готова к использованию.")

    # Инициализация метрик
    jaccard_metric = JaccardIndex(task='binary', average = 'macro').to(DEVICE)
    val_dice = DiceScore(num_classes = 2, average = 'macro').to(DEVICE)

    # Итерация по валидационному датасету
    try:
        for idx, (x, y_true) in enumerate(test_loader):
            x = x.to(DEVICE)
            y_true = y_true.to(DEVICE)

            with torch.no_grad():
                y_pred = model(x)
                y_pred_probs = torch.sigmoid(y_pred)
                y_pred_labels = (y_pred_probs > 0.5).int()

            # Вычисление метрик для текущего примера
            dice_score = val_dice(y_pred_labels, y_true).item()
            jaccard_score = jaccard_metric(y_pred_labels, y_true).item()

            # Визуализация
            image = x[0]  # Поскольку batch_size=1
            true_mask = y_true[0]
            pred_mask = y_pred_labels[0]

            overlay = overlay_masks(image, true_mask.squeeze(), pred_mask.squeeze(), alpha=0.5)

            plt.figure(figsize=(15, 5))

            # Исходное изображение
            plt.subplot(1, 3, 1)
            if image.shape[0] == 2:
                # Если 2 канала, отображаем усреднённое изображение
                img_display = image.cpu().numpy()
                img_display = np.mean(img_display, axis=0)
                plt.imshow(img_display, cmap='gray')
                plt.title('Исходное изображение')
            else:
                # Если другое количество каналов, адаптируйте под свои данные
                plt.imshow(image.cpu().numpy().transpose(1, 2, 0))
                plt.title('Исходное изображение')
            plt.axis('off')

            # Истинная маска с метриками
            plt.subplot(1, 3, 2)
            plt.imshow(true_mask.squeeze().cpu().numpy(), cmap='gray')
            plt.title(f'Истинная маска\nDice: {dice_score:.2f}, IoU: {jaccard_score:.2f}')
            plt.axis('off')

            # Предсказанная маска
            plt.subplot(1, 3, 3)
            plt.imshow(pred_mask.squeeze().cpu().numpy(), cmap='gray')
            plt.title('Предсказанная маска')
            plt.axis('off')

            plt.tight_layout()
            plt.show()

            plt.figure(figsize=(7, 7))
            plt.imshow(overlay)
            plt.title('Наложенные маски (Красный: Истинная, Синий: Предсказанная)')
            plt.axis('off')
            plt.tight_layout()
            plt.show()

            # Пауза до следующего изображения или выход по прерыванию
            input("Нажмите Enter для следующего изображения или Ctrl+C для выхода...")

    except KeyboardInterrupt:
        print("\nВизуализация прервана пользователем.")

if __name__ == "__main__":
    main()

