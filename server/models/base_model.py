
from abc import ABC, abstractmethod
import os
import pickle


class BaseModel(ABC):
    def __init__(self, model_name, params):
        self.model_name = model_name
        self.params = params

    @abstractmethod
    def train(self, X, y):
        pass

    @abstractmethod
    def predict(self, X):
        pass

    # https://nerdit.ru/sokhranieniie-modieliei-v-pickle-format/
    def save(self, model_dir):
        model_path = os.path.join(model_dir, f"{self.model_name}.pkl")
        with open(model_path, "wb") as f: 
            pickle.dump(self, f)

    # https://nerdit.ru/sokhranieniie-modieliei-v-pickle-format/
    @classmethod
    def load(cls, model_dir, model_name):
        model_path = os.path.join(model_dir, f"{model_name}.pkl")
        with open(model_path, "rb") as f:
            return pickle.load(f)
    
