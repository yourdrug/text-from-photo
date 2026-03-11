import cv2
import numpy as np

from letter_recognizer.config import IMAGE_SIZE, DEBUG_PATH


# параметры для адаптивного threshold
ADAPTIVE_BLOCK_SIZE = 31
ADAPTIVE_C = 10

# минимальная площадь компоненты (фильтрация шума)
MIN_COMPONENT_AREA = 200

# пороги объединения близких компонент
MERGE_X_THRESHOLD = 50
MERGE_Y_THRESHOLD = 100

# отступ вокруг буквы при нормализации
DEFAULT_MARGIN_RATIO = 0.1


def segment_letters(
    image_path: str,
    min_area: int = MIN_COMPONENT_AREA,
    output_size: int = IMAGE_SIZE,
    debug: bool = False,
) -> list[np.ndarray]:
    """Сегментирует изображение на отдельные буквы."""

    # загрузка изображения
    image = _load_image(image_path)

    # бинаризация и очистка изображения
    binary = _preprocess_image(image, debug=debug)

    # поиск connected components
    boxes = _extract_components(binary, min_area=min_area)

    # объединение близких компонент (например точки над i)
    boxes = merge_close_components(
        boxes, x_threshold=MERGE_X_THRESHOLD, y_threshold=MERGE_Y_THRESHOLD
    )

    # нормализация и ресайз букв
    letters = _normalize_letters(binary, boxes, output_size)

    # сохранение отладочных изображений
    if debug:
        print(f"Найдено {len(letters)}")
        for i, letter in enumerate(letters):
            cv2.imwrite(f"{DEBUG_PATH}/debug_letter_{i}.png", letter)

    return letters


def _load_image(image_path: str) -> np.ndarray:
    """Загружает изображение."""
    image = cv2.imread(image_path)
    if image is None:
        raise ValueError(f"Не удалось загрузить изображение: {image_path}")
    return image


def _preprocess_image(image: np.ndarray, debug: bool = False) -> np.ndarray:
    """Подготовка изображения: grayscale → blur → threshold → morphology."""

    # перевод в grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # небольшое сглаживание
    gray = cv2.GaussianBlur(gray, (3, 3), 0)

    # адаптивная бинаризация
    binary = cv2.adaptiveThreshold(
        gray,
        255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY_INV,
        ADAPTIVE_BLOCK_SIZE,
        ADAPTIVE_C,
    )

    # морфологические операции для очистки изображения
    binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, np.ones((2, 2), np.uint8))
    binary = cv2.morphologyEx(
        binary, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    )
    binary = cv2.morphologyEx(
        binary, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_RECT, (1, 5))
    )

    # расширение символов
    binary = cv2.dilate(binary, np.ones((2, 2), np.uint8), iterations=1)

    binary = cv2.morphologyEx(
        binary, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_RECT, (5, 1))
    )

    binary = cv2.dilate(binary, np.ones((3, 3), np.uint8), iterations=1)

    # сохранение бинарного изображения для отладки
    if debug:
        cv2.imwrite(f"{DEBUG_PATH}/debug_binary.png", binary)

    return binary


def _extract_components(
    binary: np.ndarray, min_area: int
) -> list[tuple[int, int, int, int]]:
    """Находит connected components и фильтрует шум."""

    # поиск компонент
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(
        binary, connectivity=8
    )

    boxes = []

    # пропускаем фон (label = 0)
    for i in range(1, num_labels):
        x, y, w, h, area = stats[i]

        # фильтрация мелкого шума
        if area < min_area or w < 5 or h < 10:
            continue

        boxes.append((x, y, w, h))

    return boxes


def _normalize_letters(
    binary: np.ndarray, boxes: list[tuple[int, int, int, int]], target_size: int
) -> list[np.ndarray]:
    """Вырезает и нормализует каждую букву."""

    letters = []

    for x, y, w, h in boxes:
        # вырезаем букву
        letter_img = binary[y : y + h, x : x + w]

        # нормализуем размер
        letters.append(_normalize_letter(letter_img, target_size))

    return letters


def _normalize_letter(
    letter: np.ndarray, target_size: int, margin_ratio: float = DEFAULT_MARGIN_RATIO
) -> np.ndarray:
    """Центрирует букву и приводит к фиксированному размеру."""

    h, w = letter.shape

    # вычисляем масштаб
    margin = int(target_size * margin_ratio)
    scale = (target_size - 2 * margin) / max(h, w)

    new_w, new_h = int(w * scale), int(h * scale)

    # изменение размера
    resized = cv2.resize(letter, (new_w, new_h), interpolation=cv2.INTER_AREA)

    # создаем пустой canvas
    canvas = np.zeros((target_size, target_size), dtype=np.uint8)

    # центрирование буквы
    x_offset = (target_size - new_w) // 2
    y_offset = (target_size - new_h) // 2

    canvas[y_offset : y_offset + new_h, x_offset : x_offset + new_w] = resized

    # инвертируем цвета (фон чёрный → белый)
    return cv2.bitwise_not(canvas)


def merge_close_components(
    boxes: list[tuple[int, int, int, int]],
    x_threshold: int = MERGE_X_THRESHOLD,
    y_threshold: int = MERGE_Y_THRESHOLD,
) -> list[tuple[int, int, int, int]]:
    """Объединяет близкие bounding boxes."""

    if not boxes:
        return []

    # сортируем по X координате
    boxes = sorted(boxes, key=lambda b: b[0])

    merged = []
    current = boxes[0]

    for box in boxes[1:]:
        x1, y1, w1, h1 = current
        x2, y2, w2, h2 = box

        # центры компонент
        cx1, cy1 = x1 + w1 / 2, y1 + h1 / 2
        cx2, cy2 = x2 + w2 / 2, y2 + h2 / 2

        # если компоненты близко — объединяем
        if abs(cx1 - cx2) < x_threshold and abs(cy1 - cy2) < y_threshold:
            nx1, ny1 = min(x1, x2), min(y1, y2)
            nx2, ny2 = max(x1 + w1, x2 + w2), max(y1 + h1, y2 + h2)

            current = (nx1, ny1, nx2 - nx1, ny2 - ny1)

        else:
            merged.append(current)
            current = box

    merged.append(current)

    return merged
