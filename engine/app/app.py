import io

import cv2
import numpy as np
import pytesseract
from fastapi import FastAPI
from fastapi.templating import Jinja2Templates
from starlette.requests import Request

app = FastAPI()


@app.get("/")
def home(request: Request):
    return {
        "ping": "pong"
    }


def read_img(img):
    text = pytesseract.image_to_string(img)
    return (text)


@app.post("/extract_text")
async def extract_text(request: Request):
    label = ""
    if request.method == "POST":
        form = await request.form()
        # file = form["upload_file"].file
        contents = await form["upload_file"].read()
        image_stream = io.BytesIO(contents)
        image_stream.seek(0)
        file_bytes = np.asarray(bytearray(image_stream.read()), dtype=np.uint8)
        frame = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
        label = read_img(frame)

        # return {"label": label}
    print(label)
    return {"request": request, "label": label}
