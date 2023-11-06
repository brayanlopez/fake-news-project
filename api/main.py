# TODO: adapt this api for the project

from fastapi import FastAPI, Body, Path, Query, status, Request, HTTPException, Depends
from fastapi.responses import HTMLResponse, JSONResponse

from app.view import get_prediction

app = FastAPI()


@app.get('/prediction', tags=['home'], status_code=status.HTTP_200_OK)
def message(data_to_predict: str = Query()):
    return get_prediction(data_to_predict)
