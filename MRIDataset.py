# MRIDataset.py

import torch
import albumentations as A
from albumentations.pytorch import ToTensorV2
import numpy as np
from brainMRI import Dataset
from albumentations.core.transforms_interface import DualTransform
from torchvision import transforms
import h5py

# Вставляем здесь реализацию SymmetricElasticTransform (см. выше)
class SymmetricElasticTransform(DualTransform):
    def __init__(self, alpha=1, sigma=50, alpha_affine=50, interpolation=1, border_mode=4,
                 value=None, mask_value=None, always_apply=False, approximate=False, same_dxdy=False, p=0.5):
        super(SymmetricElasticTransform, self).__init__(always_apply, p)
        self.elastic = A.ElasticTransform(
            alpha=alpha,
            sigma=sigma,
            alpha_affine=alpha_affine,
            interpolation=interpolation,
            border_mode=border_mode,
            value=value,
            mask_value=mask_value,
            approximate=approximate,
            same_dxdy=same_dxdy,
            always_apply=True,  # Всегда применяем внутри этого класса
            p=1.0  # Вероятность 1.0, так как контролируется внешним параметром
        )

    def apply(self, img, **params):
        return self._symmetric_transform(img)

    def apply_to_mask(self, mask, **params):
        return self._symmetric_transform(mask)

    def _symmetric_transform(self, img):
        """
        Реализует симметричное преобразование изображения или маски.
        Поддерживает изображения с любым количеством каналов.

        Параметры:
        - img (np.array): Изображение или маска в формате (H, W, C) или (H, W).

        Возвращает:
        - img_transformed (np.array): Преобразованное изображение или маска.
        """
        # Убедимся, что img имеет размерность (H, W, C)
        if img.ndim == 2:
            img = img[:, :, np.newaxis]

        H, W, C = img.shape
        mid = W // 2

        # Разделение на левую и правую половины
        left = img[:, :mid, :]
        right = img[:, mid:, :]

        # Отражение левой половины по горизонтали
        left_flipped = left[:, ::-1, :]

        # Объединение правой половины и отраженной левой половины
        combined = np.concatenate([right, left_flipped], axis=1)  # (H, W, C)

        # Применение ElasticTransform к объединенному изображению
        augmented = self.elastic(image=combined)['image']

        # Разделение обратно на правую и левую половины
        right_aug = augmented[:, :mid, :]
        left_aug_flipped = augmented[:, mid:, :]

        # Отражение левой половины обратно
        left_aug = left_aug_flipped[:, ::-1, :]

        # Объединение левой и правой половин
        img_transformed = np.concatenate([left_aug, right_aug], axis=1)

        # Если исходное изображение было 2D, возвращаем в исходную форму
        if img_transformed.shape[2] == 1:
            img_transformed = img_transformed[:, :, 0]

        return img_transformed


class MRIDataset(torch.utils.data.Dataset):
    """Создает MRI датасет, совместимый с PyTorch DataLoader."""

    def __init__(self, 
                 path, 
                 dataset = None,
                 ds_type: str = 'Train', 
                 volume: str = None,
                 transform=None, 
                 augment=True) -> None:
        """
        Инициализирует MRI датасет.

        Параметры:
        - transform (albumentations.Compose): Трансформации для изображений и масок.
        - augment (bool): Флаг для применения аугментаций.
        """
        super().__init__()
        self.dataset = Dataset(path=path, dataset=dataset, ds_type=ds_type, volume=volume)
        self.augment = augment
        self.labels = []

        # Определяем трансформации
        if self.augment and transform is None:
            self.transform = A.Compose([
                A.Resize(224, 224),
                A.Rotate((-15, 15), limit = 2, p = 0.2),
                A.D4(p = 0.15),
                #A.CoarseDropout(num_holes_range = [1, 3], p = 0.15),
                # A.RandomSizedBBoxSafeCrop(p = 0.1),
                # A.RandomBrightnessContrast(p=0.1),
                # A.Normalize(normalization = "min_max"),
                A.ElasticTransform(p = 0.2),
                ToTensorV2(),
            ])
        else:
            self.transform = transform

        # Проверка корректности инициализации Dataset
        if not hasattr(self.dataset, '__len__') or len(self.dataset) == 0:
            raise ValueError("Dataset не загружен корректно или пуст")

    def __getitem__(self, index: int) -> (torch.Tensor, torch.Tensor, str):
        """Возвращает образец по указанному индексу."""
        image, mask, ident = self.dataset[index]

        # Преобразуем в numpy массивы
        image = np.array(image)  # (C, H, W)
        mask = np.array(mask)    # (H, W) или (1, H, W)

        # Транспонируем изображение в формат (H, W, C)
        image = image.transpose(1, 2, 0)  # (H, W, C)

        # Убедимся, что маска имеет формат (H, W)
        if mask.ndim == 3 and mask.shape[0] == 1:
            mask = mask.squeeze(0)

        # Применяем аугментации
        if self.transform:
            augmented = self.transform(image=image, mask=mask)
            image = augmented['image']  # torch.Tensor, (C, H, W)
            mask = augmented['mask']    # torch.Tensor, (H, W)
        # Преобразуем в тензоры без аугментаций
        else:
            image = torch.from_numpy(image).permute(2, 0, 1).float()  # (C, H, W)
            mask = torch.from_numpy(mask).long()                      # (H, W)

        # Если необходимо, добавляем размерность канала к маске
        if mask.dim() == 2:
            mask = mask.unsqueeze(0)  # (1, H, W)

        return image[0].unsqueeze(0), mask

    def __len__(self) -> int:
        """Возвращает длину датасета."""
        return len(self.dataset)

    def get_labels(self):
        for ds in range(len(self.dataset)):
            image, mask, ident = self.dataset[ds]
            self.labels.append(1 if torch.argmax(torch.Tensor(mask)).item() > 0 else 0)
        return self.labels



class MRIDataset2(torch.utils.data.Dataset):
    """Создает MRI датасет, совместимый с PyTorch DataLoader."""

    def __init__(self, 
                 path, 
                 augment=True,
                 transform = None) -> None:
        """
        Инициализирует MRI датасет.

        Параметры:
        - transform (albumentations.Compose): Трансформации для изображений и масок.
        - augment (bool): Флаг для применения аугментаций.
        """

        self.augment = augment
        dataset = h5py.File(path, 'r')
        self.images = dataset['images'][:]
        self.masks = dataset['masks'][:]
        self.labels = []
        dataset.close()

        # Определяем трансформации
        if self.augment:
            self.transform = A.Compose([
                A.Resize(256, 256),
                A.D4(p = 0.3),
                A.Rotate((-60, 60), limit = 4, p = 0.3),
                # A.ShiftScaleRotate(p = 0.2),
                #A.CoarseDropout(num_holes_range = [1, 3], p = 0.15),
                # A.RandomCrop(224, 224, p = 1.),
                A.PixelDropout(p = 0.2),
                # A.Normalize(normalization = "min_max_per_channel"),
                A.RandomBrightnessContrast(p = 0.1),
                # A.ChannelDropout(p = 0.2),
                A.ElasticTransform(p = 0.2),
                ToTensorV2(),
            ])
        else:
            self.transform = transform

    def __getitem__(self, index: int) -> (torch.Tensor, torch.Tensor, str):
        """Возвращает образец по указанному индексу."""
        image, mask = self.images[index], self.masks[index]

        # Преобразуем в numpy массивы
        image = np.array(image)  # (C, H, W)
        mask = np.array(mask)    # (H, W) или (1, H, W)


        
        image = image.transpose(1, 2, 0)
#
        # Убедимся, что маска имеет формат (H, W)
        if mask.ndim == 3 and mask.shape[0] == 1:
            mask = mask.squeeze(0)


        # Применяем аугментации
        if self.transform:
            augmented = self.transform(image=image, mask=mask)
            image = augmented['image']  # torch.Tensor, (C, H, W)
            mask = augmented['mask']    # torch.Tensor, (H, W)

        # Если необходимо, добавляем размерность канала к маске
        if mask.dim() == 2:
            mask = mask.unsqueeze(0)  # (1, H, W)


        mask = torch.clamp(mask, min = 0, max = 1)


        return image, mask

    def __len__(self) -> int:
        """Возвращает длину датасета."""
        assert len(self.images) == len(self.masks)
        return len(self.images)