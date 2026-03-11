# Handwritten Letter Recognizer

Проект для **распознавания рукописных букв на изображении**.

Используемые технологии:

- **PyTorch** — нейронная сеть для классификации букв
- **OpenCV** — сегментация изображения
- **CNN (Residual + SE Attention)** — модель распознавания
- **uv** — управление зависимостями и запуск проекта

---

# Как работает система

Pipeline обработки изображения:

Image -> Preprocessing (OpenCV) -> Letter Segmentation -> Normalization -> CNN Classification -> Recognized Text

---

# Возможности

- сегментация рукописного текста на буквы
- нормализация символов
- CNN модель для распознавания
- обучение на собственном датасете
- debug режим для анализа сегментации

---

# Структура проекта
```
letter_recognizer/
│
├── config.py # настройки проекта
├── dataset.py # загрузка датасета и аугментации
├── model.py # CNN модель
├── train.py # обучение модели
├── predict.py # распознавание
├── segment.py # сегментация букв
├── utils.py # вспомогательные функции
│
main.py # CLI интерфейс
```
---
# Установка:

Проект использует **uv**.

Установка зависимостей:

```bash
uv sync
```

# Подготовка датасета:

Используется формат ImageFolder из torchvision.

Структура датасета:
```
dataset/
│
├── A/
│   ├── img1.png
│   ├── img2.png
│
├── B/
│   ├── img1.png
│   ├── img2.png
│
├── C/
│   ├── img1.png
```
Название папки = класс буквы.

# Обучение модели:

Для обучения выполните:

```bash
uv run python -m letter_recognizer.train
```

Во время обучения выводится:

* loss
* accuracy
* информация по эпохам

Лучшая модель автоматически сохраняется.

После обучения появится файл:
```
model.pth
```

# Распознавание текста:

CLI интерфейс позволяет распознавать текст на изображении.

Запуск:
```bash
uv run python main.py -i image.png
```
Вывод:
```
Recognized: HELLO
```
# Аргументы CLI:
**Путь к изображению для распознавания.**
```
--image / -i
```

Пример:
```
-i image.png
```

**Режим отладки.**
```
--debug / -d
```

Сохраняются промежуточные изображения:
* бинаризация
* сегментированные буквы

Пример:
```
-d
```

**Файл для сохранения результата.**
```
--output / -o
```

Пример:
```
-o result.txt
```

# Примеры использования:

**Простое распознавание**
```bash
uv run python main.py -i image.png
```
**Распознавание с debug режимом**
```bash
uv run python main.py -i image.png -d
```

**Будут сохранены:**

* debug_binary.png
* debug_letter_0.png
* debug_letter_1.png

**Сохранение результата в файл**
```bash
uv run python main.py -i image.png -o result.txt
```

**Полный пример запуска:**
```bash
uv run python main.py -i my_test_image.jpg -d -o result.txt
```

# Как работает распознавание:

* Загружается изображение
* Выполняется preprocessing
  * grayscale
  * blur
  * adaptive threshold
  * morphology

* Выполняется сегментация букв
  * connected components
  * merge boxes

* Каждая буква нормализуется
  * resize
  * center
  * padding

* Буквы передаются в CNN модель
* Модель предсказывает текст

# Debug режим:

Debug режим помогает анализировать сегментацию.

Сохраняются изображения:

* debug_binary.png
* debug_letter_0.png
* debug_letter_1.png
* ...

# Требования:

* Python 3.10+
* PyTorch
* torchvision
* OpenCV
* numpy

# Возможные улучшения:
* поддержка нескольких строк текста
* улучшенная сегментация
* более сложная CNN архитектура
* поддержка разных алфавитов