import torch
import torch.nn as nn
import torch.nn.functional as F


# SEBlock (Squeeze-and-Excitation Block)
# Этот блок реализует механизм внимания к каналам.
# Он позволяет сети автоматически определять,
# какие каналы признаков более важны.
class SEBlock(nn.Module):
    def __init__(self, channels, reduction=16):
        super().__init__()

        # Небольшая полносвязная сеть, которая вычисляет
        # веса (важность) для каждого канала
        self.fc = nn.Sequential(
            nn.Linear(channels, channels // reduction),  # уменьшаем размерность
            nn.ReLU(inplace=True),
            nn.Linear(
                channels // reduction, channels
            ),  # возвращаем размерность обратно
            nn.Sigmoid(),  # значения от 0 до 1 (веса каналов)
        )

    def forward(self, x):
        # x имеет форму:
        # (batch_size, channels, height, width)

        b, c, _, _ = x.size()

        # Global Average Pooling
        # превращает каждую карту признаков в одно число
        # (batch_size, channels, 1, 1)
        y = F.adaptive_avg_pool2d(x, 1).view(b, c)

        # пропускаем через маленькую сеть,
        # чтобы получить веса для каналов
        y = self.fc(y).view(b, c, 1, 1)

        # умножаем входные признаки на веса каналов
        return x * y


# Residual Block — основной строительный блок сети
# Использует skip connection (остаточное соединение)
# чтобы улучшить обучение глубоких сетей
class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, downsample=False):
        super().__init__()

        # Если нужно уменьшить размер карты признаков
        # используем stride = 2
        stride = 2 if downsample else 1

        # Первая свёртка
        self.conv1 = nn.Conv2d(
            in_channels,
            out_channels,
            3,  # ядро 3x3
            stride=stride,  # уменьшает размер если downsample=True
            padding=1,  # сохраняет размер
            bias=False,
        )

        # Нормализация батча
        self.bn1 = nn.BatchNorm2d(out_channels)

        # Вторая свёртка
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, padding=1, bias=False)

        self.bn2 = nn.BatchNorm2d(out_channels)

        # Блок внимания
        self.se = SEBlock(out_channels)

        # Shortcut connection (skip connection)
        # Если число каналов меняется или происходит downsample,
        # нужно привести вход к той же форме
        if downsample or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                # 1x1 свёртка меняет число каналов
                nn.Conv2d(in_channels, out_channels, 1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels),
            )
        else:
            # Если размеры совпадают — просто пропускаем вход
            self.shortcut = nn.Identity()

    def forward(self, x):
        # сохраняем вход для skip connection
        identity = self.shortcut(x)

        # основной путь
        out = F.relu(self.bn1(self.conv1(x)))  # Conv -> BN -> ReLU
        out = self.bn2(self.conv2(out))  # Conv -> BN

        # применяем SE attention
        out = self.se(out)

        # складываем с shortcut
        out += identity

        # финальная активация
        out = F.relu(out)

        return out


# Основная CNN модель для классификации букв
class LetterCNN(nn.Module):
    def __init__(self, num_classes: int):
        super().__init__()

        # Stem — начальный слой сети
        # превращает входное изображение (1 канал)
        # в 64 карты признаков
        self.stem = nn.Sequential(
            nn.Conv2d(1, 64, 3, padding=1, bias=False),  # вход: grayscale изображение
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
        )

        # Residual блоки
        # постепенно увеличивают число каналов
        # и уменьшают размер изображения

        self.layer1 = ResidualBlock(64, 64)  # размер сохраняется
        self.layer2 = ResidualBlock(64, 128, downsample=True)  # уменьшает размер /2
        self.layer3 = ResidualBlock(128, 256, downsample=True)  # уменьшает размер /2
        self.layer4 = ResidualBlock(256, 512, downsample=True)  # уменьшает размер /2

        # Global Average Pooling
        # превращает карту признаков в вектор
        self.pool = nn.AdaptiveAvgPool2d(1)

        # Классификатор
        # преобразует признаки в вероятности классов
        self.classifier = nn.Sequential(
            nn.Linear(512, 256),  # уменьшаем размер признаков
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),  # регуляризация (борьба с переобучением)
            nn.Linear(256, num_classes),  # выход: количество классов
        )

    def forward(self, x):
        # начальная обработка изображения
        x = self.stem(x)

        # извлечение признаков через residual блоки
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        # глобальное усреднение
        x = self.pool(x)

        # превращаем в вектор
        x = torch.flatten(x, 1)

        # классификация
        return self.classifier(x)
