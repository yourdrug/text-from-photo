from pathlib import Path
import torch

BASE_DIR = Path(__file__).resolve().parent.parent.parent

DATASET_PATH = BASE_DIR / "app" / "data" / "dataset"
MODEL_PATH = BASE_DIR / "app" / "models" / "model_ver.pth"
DEBUG_PATH = BASE_DIR / "app" / "debug"

IMAGE_SIZE = 64  # лучший размер исходя из опыта
BATCH_SIZE = 64
EPOCHS = 40  # достаточно для достижения confidence ~ 95% на каждую букву
LEARNING_RATE = 3e-4
WEIGHT_DECAY = 1e-4

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
