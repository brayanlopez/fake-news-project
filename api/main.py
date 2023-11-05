# TODO: adapt this api for the project

from fastapi import FastAPI, Body, Path, Query, status, Request, HTTPException, Depends
from fastapi.responses import HTMLResponse, JSONResponse
app = FastAPI()

@app.get('/', tags=['home'], status_code=status.HTTP_200_OK)
def message():
    return HTMLResponse('<h1>Hello world</h1>')

