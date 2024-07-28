from ..models import models_name, MODELS_PATH

import pickle as pk
import os

class Model:

    @staticmethod
    def from_pretrained(
        model_id: str
    ) -> tuple[str, None]:
        return Model._get_model_by_id(model_id)
    

    @staticmethod  
    def get_model_id(model_name: str):
        return models_name.index(model_name)
    

    @staticmethod
    def _get_model_by_id(model_id) -> tuple[str, None]:

        model_path = os.path.join(MODELS_PATH, f"model {models_name[model_id]}.pkl")

        with open(model_path, "rb") as f:
            model = pk.load(f)

        return model
    

class LoadLabelEncoder:

    def from_pretrained():
        return LoadLabelEncoder._get_label_encoder() 

    
    @staticmethod
    def _get_label_encoder():

        model_path = os.path.join(MODELS_PATH, "label_encoding.pkl")

        with open(model_path, "rb") as f:
            model = pk.load(f)

        return model





