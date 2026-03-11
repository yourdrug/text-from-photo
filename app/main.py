import argparse

from letter_recognizer.predict import predict
from letter_recognizer.utils import save_result


def main():
    parser = argparse.ArgumentParser(description="Распознавание текста с изображения")
    parser.add_argument(
        "--image",
        "-i",
        type=str,
        required=True,
        help="Путь к изображению для распознавания",
    )
    parser.add_argument(
        "--debug",
        "-d",
        action="store_true",
        help="Включить debug режим (сохраняются промежуточные изображения)",
    )
    parser.add_argument(
        "--output",
        "-o",
        type=str,
        default=None,
        help="Файл для сохранения результата (по умолчанию печать в консоль)",
    )

    args = parser.parse_args()

    # Распознавание
    text = predict(args.image, debug_mode=args.debug)

    # Сохранение результата
    if args.output:
        save_result(text, args.output)
    else:
        print("Recognized:", text)


if __name__ == "__main__":
    main()
