from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware

import cv2 as cv
import numpy as np

from feature_utils import face_utils
from MLs.Model_Controller import models

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
    return {"message": "Hello World"}


@app.post("/resgis")
def regis():
    return {"regis": "success"}


@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):

    model = models['m']

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
