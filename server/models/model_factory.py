
from abc import ABC

from models.logistic_regression import LogisticRegression
from models.linear_regression import LinearRegression


class ModelFactory(ABC):
    models = {'LogisticRegression': LogisticRegression,
             'LinearRegression': LinearRegression,
            }
    
    @classmethod
    def register_model(cls, model_type, model_class):
        cls.models[model_type] = model_class

    @classmethod
    def create_algorithm(cls, model_name, model_type, params):
        if params is None:
            params = {}
        model_class = cls.models.get(model_type)
        if model_class:
            return model_class(model_name, params)
        else:
            raise ValueError(f"Неизвестная модель: {model_type}")