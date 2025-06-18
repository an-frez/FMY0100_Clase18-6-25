from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles

import os
import numpy as np
import pickle

from fastapi import Request
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates


app = FastAPI()
templates = Jinja2Templates(directory="static")
app.mount("/static", StaticFiles(directory="static"), name="static")

def ValuePredictor(to_predict_list):
    to_predict = np.array(to_predict_list).reshape(1, 4)
    loaded_model = pickle.load(open("checkpoints/model.pkl","rb"))
    result = loaded_model.predict(to_predict)
    return result[0]

@app.post("/result", response_class=HTMLResponse)
async def result(request: Request):
    form = await request.form()
    to_predict_list = list(form.values())  # Extraemos los valores del form

    try:
        to_predict_list = list(map(float, to_predict_list))
        result = ValuePredictor(to_predict_list)
        if int(result) == 0:
            prediction = 'Iris-Setosa'
        elif int(result) == 1:
            prediction = 'Iris-Virginica'
        elif int(result) == 2:
            prediction = 'Iris-Versicolour'
        else:
            prediction = f'{int(result)} No-definida'
    except ValueError:
        prediction = 'Error en el formato de los datos'

    return templates.TemplateResponse("result.html", {"request": request, "prediction": prediction})

@app.get("/")
async def root():
    return {"message": "Bye World"}