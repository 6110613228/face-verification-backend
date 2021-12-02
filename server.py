from fastapi import FastAPI, WebSocket, WebSocketDisconnect, UploadFile, File, Form
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse

import time
import shutil
from pydantic import BaseModel
import os
import cv2 as cv
import numpy as np


from feature_utils import face_utils
#from MLs.Model_Controller import models

# test run command:  uvicorn server:app --reload
app = FastAPI()

origins = [
    '*'
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=['*'],
    allow_headers=['*'],
)


@app.get("/")
async def main():

    return {"hello": "Hello world"}


@app.post("/register")
async def regis(image: UploadFile = File(...), video: UploadFile = File(...), label: str = Form(...)):

    # -- gen file path --
    video_path, image_path = gen_file_path(label)

    # -- save file --
    with open(video_path, "wb") as buffer:
        shutil.copyfileobj(video.file, buffer)
    with open(image_path, "wb") as buffer:
        shutil.copyfileobj(image.file, buffer)

    return {
        "result": "True",
        "message": "Registation success"
    }


@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):

    #model = models['m']

    await websocket.accept()
    while True:
        try:
            data = await websocket.receive_bytes()

            nparr = np.frombuffer(data, np.uint8)
            image = cv.imdecode(nparr, 1)

            faces_bb = face_utils.detect_face(image)
            cropped_images = face_utils.crop_face(image, faces_bb)

            count_found_faces = len(cropped_images)

            if (count_found_faces == 2):
                result = model.face_verification(cropped_images)
            else:
                result = False
            #img_str = cv.imencode('.png', image)[1].tobytes()

            await websocket.send_json({
                'count_face': count_found_faces,
                'found_faces_bb': faces_bb,
                'is_same_person': result
            })

        except WebSocketDisconnect:
            await websocket.close()
            break


def gen_file_path(label):
    raw_data = "raw_data/"
    video_pathDir = raw_data+label+"/"
    image_pathDir = raw_data+label+"/"
    try:
        os.makedirs(video_pathDir)
    except FileExistsError:
        # directory already exists
        pass
    try:
        os.makedirs(image_pathDir)
    except FileExistsError:
        # directory already exists
        pass
    return video_pathDir+"video_"+label+".mp4", image_pathDir+"image_"+label+".png"
