from datetime import datetime

import torch
import torch.nn as nn
import torch.optim as optim

from letter_recognizer.dataset import get_dataloader
from letter_recognizer.model import LetterCNN

from letter_recognizer.config import (
    EPOCHS,  # количество эпох обучения
    MODEL_PATH,  # путь для сохранения модели
    DEVICE,  # устройство (cpu или cuda)
    LEARNING_RATE,  # скорость обучения
    WEIGHT_DECAY,  # коэффициент L2-регуляризации
)


def train() -> None:
    """Функция обучения CNN с выводом результатов по каждой эпохе с сохранением лучшей моделис"""
    print("Training...")

    # загружаем DataLoader и список классов
    # classes — это список названий папок (например A, B, C ...)
    train_loader, classes = get_dataloader()

    model = LetterCNN(num_classes=len(classes)).to(DEVICE)

    # функция потерь
    # CrossEntropyLoss используется для задач классификации
    # label_smoothing уменьшает переобучение модели
    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)

    # оптимизатор AdamW
    # AdamW — улучшенная версия Adam с weight decay
    optimizer = optim.AdamW(
        model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY
    )

    best_train_acc = 0.0

    for epoch in range(EPOCHS):
        # переводим модель в режим обучения
        # включает dropout и batchnorm обучение
        model.train()

        train_loss = 0
        train_correct = 0
        train_total = 0

        for images, labels in train_loader:
            images = images.to(DEVICE, non_blocking=True)
            labels = labels.to(DEVICE, non_blocking=True)

            # обнуляем градиенты перед новым шагом
            optimizer.zero_grad()

            # прямой проход (forward pass)
            # получаем предсказания модели
            outputs = model(images)

            # вычисляем функцию потерь
            loss = criterion(outputs, labels)

            # обратное распространение ошибки (backpropagation)
            loss.backward()

            # обновление весов модели
            optimizer.step()

            # суммируем loss
            # умножаем на размер батча для корректного среднего
            train_loss += loss.item() * images.size(0)

            # получаем индекс класса с максимальной вероятностью
            predicted = outputs.argmax(1)

            # увеличиваем количество обработанных примеров
            train_total += labels.size(0)
            train_correct += (predicted == labels).sum().item()

        train_loss /= train_total
        train_acc = 100 * train_correct / train_total

        print(
            f"[{datetime.now().strftime('%d.%m.%Y %H:%M:%S')}] | "
            f"Epoch {epoch + 1}/{EPOCHS} | "
            f"Loss: {train_loss:.4f} | "
            f"Acc: {train_acc:.2f}%"
        )

        if train_acc > best_train_acc:
            best_train_acc = train_acc
            torch.save(
                {
                    # веса модели
                    "model_state": model.state_dict(),
                    # состояние оптимизатора (нужно для продолжения обучения)
                    "optimizer_state": optimizer.state_dict(),
                    # список классов (букв)
                    "classes": classes,
                },
                MODEL_PATH,
            )

            print("🔥 Best model saved")

    print("Training finished.")
    print(f"Best Accuracy: {best_train_acc:.2f}%")


# запуск обучения если файл запущен напрямую
if __name__ == "__main__":
    train()
