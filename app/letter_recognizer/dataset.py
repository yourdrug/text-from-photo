import random

from PIL import ImageFilter
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

# Импортируем параметры из конфигурации проекта
# IMAGE_SIZE — размер изображения (например 64x64)
# BATCH_SIZE — размер батча для обучения
# DATASET_PATH — путь к датасету
from letter_recognizer.config import IMAGE_SIZE, BATCH_SIZE, DATASET_PATH


# Функция возвращает трансформации для обучающего датасета
# Здесь используются аугментации (искусственное изменение изображений),
# чтобы увеличить разнообразие данных и уменьшить переобучение модели
def get_train_transforms():
    return transforms.Compose(
        [
            # Переводим изображение в grayscale (1 канал)
            # модель ожидает изображения формата (1, H, W)
            transforms.Grayscale(num_output_channels=1),
            # Изменяем размер изображения до фиксированного
            transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
            # Случайно меняем толщину линии буквы
            # используется кастомная функция random_thickness
            transforms.Lambda(lambda img: random_thickness(img)),
            # Случайный поворот изображения
            # имитирует наклон рукописных букв
            transforms.RandomRotation(degrees=15),
            # Случайное геометрическое преобразование
            transforms.RandomAffine(
                degrees=5,  # небольшой поворот
                translate=(0.2, 0.2),  # случайный сдвиг (20%)
                scale=(0.8, 1.2),  # случайное масштабирование
            ),
            # Изменение яркости и контраста
            # имитирует разное освещение и качество сканирования
            transforms.ColorJitter(brightness=0.5, contrast=0.5),
            # Перевод изображения в тензор PyTorch
            # формат становится (C, H, W)
            transforms.ToTensor(),
            # Нормализация значений пикселей
            # значения переводятся примерно в диапазон [-1, 1]
            transforms.Normalize((0.5,), (0.5,)),
        ]
    )


# Трансформации для validation / теста
# Здесь НЕТ аугментаций, чтобы проверка была честной
def get_val_transforms():
    return transforms.Compose(
        [
            # перевод в grayscale
            transforms.Grayscale(),
            # изменение размера
            transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
            # преобразование в tensor
            transforms.ToTensor(),
            # нормализация
            transforms.Normalize((0.5,), (0.5,)),
        ]
    )


# Функция случайно изменяет толщину линий буквы
# используется для увеличения разнообразия датасета
def random_thickness(img):

    # случайно выбираем действие
    if random.random() < 0.5:
        # MinFilter делает линии тоньше
        img = img.filter(ImageFilter.MinFilter(3))
    else:
        # MaxFilter делает линии толще
        img = img.filter(ImageFilter.MaxFilter(3))

    return img


# Функция возвращает список классов
# Например:
# ["A", "B", "C", "D", ...]
def get_classes():

    # ImageFolder автоматически читает папки
    # каждая папка = один класс
    full_dataset = datasets.ImageFolder(DATASET_PATH)

    # classes — список названий папок
    return full_dataset.classes


# Создание DataLoader для обучения
def get_dataloader():

    # Загружаем датасет
    # ImageFolder ожидает структуру:
    #
    # DATASET_PATH/
    #     A/
    #         img1.png
    #         img2.png
    #     B/
    #         img1.png
    #     C/
    #         img1.png
    #
    dataset = datasets.ImageFolder(
        DATASET_PATH,
        transform=get_train_transforms(),  # применяем train аугментации
    )

    # DataLoader разбивает данные на батчи
    loader = DataLoader(
        dataset,
        batch_size=BATCH_SIZE,  # сколько изображений в одном батче
        shuffle=True,  # перемешивание данных
        num_workers=4,  # количество потоков загрузки данных
    )

    # возвращаем loader и список классов
    return loader, dataset.classes
