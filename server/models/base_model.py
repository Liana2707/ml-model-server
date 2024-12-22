
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
    
