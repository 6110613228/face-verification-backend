from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware

import cv2 as cv
import numpy as np

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
    return { "message": "Hello World" }


count = 0
@app.websocket("/ws/{option}")
async def websocket_endpoint(websocket: WebSocket):
    count = 0
    await websocket.accept()
    while True:
        try:
            data = await websocket.receive_bytes()
            option = websocket.path_params['option']

            nparr = np.frombuffer(data, np.uint8)
            image = cv.imdecode(nparr, -1)
            #cv.imwrite('test.png', image)

            #img_str = cv.imencode('.png', image)[1].tobytes()

            print('Got image', option)
            count += 1
            await websocket.send_json({
                'message': 'test',
                'face_is_found': True,
                'count_face': count,
                'found_faces_bb' : []
            })

        except WebSocketDisconnect:
            await websocket.close()
            print('Closed connection')
            break
