import os
import sys

from ..config import BASE_PATH

MODELS_PATH = os.path.join(BASE_PATH, "models/bin")

models_name = list(map(
    lambda x: x.split("model ")[1][:-4], filter(
        lambda x: x.startswith("model"), os.listdir(MODELS_PATH)
        )
    )
)

