from functools import lru_cache
import multiprocessing
from typing import Any, Dict
from fastapi import Depends, FastAPI, HTTPException
from typing_extensions import Annotated

from config import Settings
from schemas import ErrorResponse, FitRequest, LoadRequest, PredictRequest, Response, UnloadRequest
from models.model_factory import ModelFactory

app = FastAPI()

model_processes: Dict[str, multiprocessing.Process] = {}
loaded_models: Dict[str, Any] = {}

@lru_cache
def get_settings():
    return Settings()

@app.post("/load", response_model=Response, responses={400: {"model": ErrorResponse}, 404: {"model": ErrorResponse}})
async def load(load_request: LoadRequest, settings: Annotated[Settings, Depends(get_settings)]):
    # 
    if len(loaded_models) >= settings.max_inference_models:
            raise HTTPException(status_code=400, detail="Достигнуто максимальное количество загруженных моделей")
    
    model_name = load_request.model_name

    if model_name in loaded_models:
        raise HTTPException(status_code=400,detail=f"Модель с именем `{model_name}` уже загружена")
    
    try:
        model_wrapper = ModelFactory.load(settings.model_dir, model_name)
        loaded_models[model_name] = model_wrapper
    except FileNotFoundError:
         raise HTTPException(status_code=400, detail=f"Модель с именем `{model_name}` не найдена на диске")
    except Exception as e:
         raise HTTPException(status_code=400, detail=f"Ошибка во время загрузки модели: {e}")
    
    return Response(message=f"Модель {model_name} загружена.")

@app.post("/unload", response_model=Response, responses={404: {"model": ErrorResponse}})
async def unload(unload_request: UnloadRequest):
    # выгрузка модели
    model_name = unload_request.model_name
    if model_name not in loaded_models:
        raise HTTPException(status_code=404, detail=f"Модель с именем {model_name} не загружена")
    
    del loaded_models[model_name]
    return Response(message=f"Модель {model_name} выгружена.")

# https://github.com/Liana2707/TimeSeriesForecasting/tree/master
@app.post("/fit", response_model=Response, responses={400: {"model": ErrorResponse}})
async def fit(fit_request: FitRequest, settings: Annotated[Settings, Depends(get_settings)]):
    # создание, обучение и сохранение модели на диск
    if len(model_processes) >= settings.num_cores:
        raise HTTPException(status_code=404, detail="Достигнуто максимальное количество задач обучения")
    
    model_name = fit_request.model_name
    model_type = fit_request.model_type

    if not model_name or not model_type:
        raise HTTPException(status_code=404, detail="Необходимо указать имя модели и тип модели")
    
    if model_name in model_processes:
        raise HTTPException(status_code=404, detail=f"Модель с именем `{model_name}` уже обучается")

    params = fit_request.config.get("params")
    if model_type not in ModelFactory.models.keys():
         raise HTTPException(status_code=400, detail="Указан неправильный тип модели")
    
    try:
        model = ModelFactory.create_algorithm(model_name, model_type, params)
        model.train(fit_request.X, fit_request.y)
        model.save(settings.model_dir)
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Произошла ошибка при обучении и сохранении модели: {e}")
    return Response(message=f"Обучение модели `{model_name}` выполнено.")


@app.post("/predict", response_model=Response, responses={400: {"model": ErrorResponse}, 404: {"model": ErrorResponse}})
async def predict(predict_request: PredictRequest):
    model_name = predict_request.model_name
    if model_name not in loaded_models:
        raise HTTPException(status_code=404, detail=f"Модель с именем {model_name} не загружена")
    try:
        predictions = loaded_models[model_name].predict(predict_request.X)
        return Response(message=f"Предсказание для модели {model_name}: {predictions}")
    except Exception as e:
          raise HTTPException(status_code=400, detail=f"Ошибка во время предсказания модели: {e}")








