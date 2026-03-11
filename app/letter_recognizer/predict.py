import torch
from PIL import Image

from letter_recognizer.segmentation import segment_letters
from letter_recognizer.dataset import get_val_transforms
from letter_recognizer.config import MODEL_PATH, DEVICE
from letter_recognizer.utils import load_model


def predict(image_path: str, debug_mode: bool = False) -> str:
    """
    Основная функция по предсказанию букв на фотке

    Flow:
        - загрузка модели из файла
        - разделяем буквы на фотографии
        - применяем "трансформацию" к каждой букве
        - выводим уверенность модели (для debug режима)
        - сохраняем результат в файл
    """
    model, classes = load_model(MODEL_PATH, DEVICE)
    model.to(DEVICE)
    model.eval()

    transform = get_val_transforms()
    letters = segment_letters(image_path, debug=debug_mode)

    result = ""

    for letter in letters:
        img = Image.fromarray(letter)
        tensor = transform(img).unsqueeze(0).to(DEVICE)

        with torch.no_grad():
            output = model(tensor)
            pred = torch.argmax(output, dim=1).item()

            probs = torch.softmax(output, dim=1)
            confidence, pred = torch.max(probs, 1)

            if debug_mode:
                print(confidence.item())

        result += classes[pred]

    return result
