import torch

from letter_recognizer.model import LetterCNN


def save_result(text: str, output_path: str = "result.txt") -> None:
    """Функция для сохранения результатов предсказания, по умолчанию в result.txt"""
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(text)


def load_model(model_path: str, device: torch.device) -> tuple[LetterCNN, list[str]]:
    """Загрузка модели из файла и перевод ее в режим 'предсказания'"""
    checkpoint = torch.load(model_path, map_location=device)

    classes = checkpoint["classes"]

    model = LetterCNN(len(classes))
    model.load_state_dict(checkpoint["model_state"])

    model.to(device)
    model.eval()

    return model, classes
