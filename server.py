from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware

import cv2 as cv
import numpy as np

from feature_utils import face_utils
from Models.Model_Controller import models

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


@app.websocket("/ws/{option}")
async def websocket_endpoint(websocket: WebSocket):
    count = 0
    await websocket.accept()
    while True:
        try:
            data = await websocket.receive_bytes()
            option = websocket.path_params['option']

            nparr = np.frombuffer(data, np.uint8)
            image = cv.imdecode(nparr, 1)

            faces_bb = face_utils.detect_face(image)
            results = face_utils.crop_face(image, faces_bb)

            for i, f in enumerate(results):
                cv.imwrite('image' + str(i) + '.png', f)
            model = models['wave']

            #img_str = cv.imencode('.png', image)[1].tobytes()

            await websocket.send_json({
                'message': 'test',
                'face_is_found': True,
                'count_face': 0,
                'found_faces_bb': faces_bb
            })

        except WebSocketDisconnect:
            await websocket.close()
            print('Closed connection')
            break
