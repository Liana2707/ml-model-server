from asyncio import Semaphore, create_task
import asyncio
from concurrent.futures import ProcessPoolExecutor
import glob
import os
from typing import Any, Dict
from fastapi import FastAPI, HTTPException
from fastapi.concurrency import asynccontextmanager

from config import Settings
from schemas import ErrorResponse, FitRequest, LoadRequest, PredictRequest, Response, UnloadRequest
from models.model_factory import ModelFactory

@asynccontextmanager
async def lifespan(app: FastAPI):
    global settings
    settings = Settings()
    global active_processes
    active_processes = asyncio.Semaphore(settings.num_cores - 1)  
    yield

app = FastAPI(lifespan=lifespan)

loaded_models = {}


@app.post("/load", response_model=Response, responses={400: {"model": ErrorResponse}, 404: {"model": ErrorResponse}})
async def load(load_request: LoadRequest):
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
async def fit(fit_request: FitRequest):
    # создание, обучение и сохранение модели на диск    
    model_name = fit_request.model_name
    model_type = fit_request.model_type

    if not model_name or not model_type:
        raise HTTPException(status_code=404, detail="Необходимо указать имя модели и тип модели")

    if model_type not in ModelFactory.models.keys():
         raise HTTPException(status_code=400, detail="Указан неправильный тип модели")
  
    if not active_processes.locked():
        async with active_processes:
            try:
                loop = asyncio.get_event_loop()
                with ProcessPoolExecutor() as executor:
                    params = fit_request.config.get("params")
                    model = ModelFactory.create_algorithm(model_name, model_type, params)
                    model = await loop.run_in_executor(executor, model.train, fit_request.X, fit_request.y)
                    ModelFactory.save(model, model_name, settings.model_dir)
                    return Response(message=f"Обучение модели `{fit_request.model_name}` выполнено.")
            except Exception as e:
                raise HTTPException(status_code=400, detail=f"Произошла ошибка при обучении и сохранении модели: {e}")


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
    
@app.delete("/models/remove_all", response_model=Response)
async def remove_all_models():
    model_dir = settings.model_dir
    try:
        pkl_files = glob.glob(os.path.join(model_dir, "*.pkl"))
        for file_path in pkl_files:
            os.remove(file_path)
        return Response(message=f"Все модели удалены с диска")
    except OSError as e:
        raise HTTPException(status_code=400, detail=f"Ошибка удаления моделей: {e}")
    
@app.delete("/models/remove", response_model=Response, responses={400: {"model": ErrorResponse}})
async def remove_model(model_name: str):
    model_path = os.path.join(settings.model_dir, f"{model_name}.pkl")

    if not os.path.exists(model_path):
        raise HTTPException(status_code=404, detail=f"Модель {model_name}.pkl не найдена на диске")
    try:
        os.remove(model_path)
        return Response(message=f"Модель {model_name}.pkl удалена с диска.")
    except OSError as e:
        raise HTTPException(status_code=400, detail=f"Ошибка удаления модели с диска: {e}")
    