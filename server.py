from fastapi import FastAPI, WebSocket, WebSocketDisconnect, UploadFile, File, Form, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
import shutil

import os
import cv2 as cv
import numpy as np
import tensorflow as tf

from feature_utils import face_utils
from MLs.Model_Controller import models

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

model = models['wave']
face_detector = face_utils.load_mtcnn.mtcnn

SAVE_DIR = os.getcwd() + '/MLs/models/bnet/database/'
model.face_registration()

@app.get("/")
async def main():

    return {"hello": "Hello world"}


@app.post("/register")
async def regis(image: UploadFile = File(...), video: UploadFile = File(...), label: str = Form(...) ):

    main_dir = os.getcwd()+'/raw_data/'
    # -- gen file path --
    video_path, imgID_path = gen_file_path(label)

    # -- save file --
    with open(video_path, "wb") as buffer:
        shutil.copyfileobj(video.file, buffer)
    with open(imgID_path, "wb") as buffer:
        shutil.copyfileobj(image.file, buffer)

    res_vid, id_path, face_path = detect_from_vid(vid_path=video_path, saveas="test_face2_60.avi",
                                                  n_sample=30, fps=30, capture=True, label_id=label)
    print(imgID_path)
    print(id_path)

    res_pic = detect_img_from_file(
        filename=imgID_path, des_path=id_path, conf_t=0.95, label=label)
    
    if res_pic and res_vid:
        message = "Registration success"
    elif res_pic:
        message = "We can't detect face from your video"
    elif res_vid:
        message = "We can't detect face from your image"
    else:
        message = "We can't detect face from your video and image"
    
    model.face_registration()

    return {
        "result": res_pic and res_vid,
        "message": message
    }


@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):

    await websocket.accept()
    while True:
        try:
            data = await websocket.receive_bytes()

            nparr = np.frombuffer(data, np.uint8)
            image = cv.imdecode(nparr, 1)
            image = cv.cvtColor(image,cv.COLOR_BGR2RGB)

            faces_bb = face_utils.detect_face(image)
            cropped_images = face_utils.crop_face(image, faces_bb)
            count_found_faces = len(cropped_images)
            # for i,img in enumerate(cropped_images) :
            #     cv.imwrite(f"test_img{i}.jpg",img)
            # Face recognition
            if (count_found_faces > 0):
                result_classes = model.face_recognition(cropped_images)
                for i, f in enumerate(faces_bb):
                    f['label'] = result_classes[i]

            # Face verification
            if (count_found_faces == 2):
                try:
                    result = model.face_verification(cropped_images)
                except:
                    result=False 
            else:
                result = False
            #img_str = cv.imencode('.png', image)[1].tobytes()

            await websocket.send_json({
                'count_face': count_found_faces,
                'found_faces': faces_bb,
                'is_same_person': result
            })

        except WebSocketDisconnect:
            await websocket.close()
            break


def gen_file_path(label):
    raw_data = "raw_data/"
    video_pathDir = raw_data+label+"/"
    imgID_pathDir = os.getcwd()+'/'+raw_data+label+"/"

    # raw_data/label/face_detected/pic.png --format--
    try:
        os.makedirs(video_pathDir)
    except FileExistsError:
        # directory already exists
        pass
    try:
        os.makedirs(imgID_pathDir)
    except FileExistsError:
        # directory already exists
        pass
    return video_pathDir+"video_"+label+".avi", imgID_pathDir+"image_"+label+".png"


def detect_from_vid(vid_path, label_id, saveas: str, conf_t=0.85, fps: int = 30, n_sample: int = 10, capture: bool = False):

    vc = cv.VideoCapture(vid_path)
    # frame_width = int(vc.get(3))
    # frame_height = int(vc.get(4))
    fps = int(vc.get(cv.CAP_PROP_FPS))
    print("vid fps:",fps)
    n = 0
    try:
        os.makedirs(SAVE_DIR)
    except OSError:
        print("Folder already exists. continue ...")

    print(f"Processing from {vid_path}")
    i = 0
    while True:
        
        ret = vc.grab()
        if i % 5 == 0:
            ret, frame = vc.retrieve()
            print("*", end="")
            if not ret:
                break
            frame_rgb = cv.cvtColor(frame, cv.COLOR_BGR2RGB)
            results = face_detector.detect_faces(frame_rgb)
            if n == n_sample:
                capture = False
                break
            if capture:

                if len(results) == 2:
                    conf1 = results[0]['confidence']
                    conf2 = results[1]['confidence']
                    if ((conf1 > conf_t) and (conf2 > conf_t)) :  # 3 Frame interval
                        face1 = capture_face(results[0], frame)
                        face2 = capture_face(results[1], frame)
                        try:
                            os.makedirs(
                                SAVE_DIR + f"{label_id}/face_only")
                            os.makedirs(
                                SAVE_DIR + f"{label_id}/id_only")
                        except OSError:
                            print("Folder already exists. continue ...")

                        cv.imwrite(
                            SAVE_DIR + f"{label_id}/face_only/{label_id}-face_sample{n+1}.jpg", face1)
                    
                        cv.imwrite(
                            SAVE_DIR + f"{label_id}/id_only/{label_id}-id_sample{n+1}.jpg", face2)

                        print(f"Save sample{n+1}")
                        n += 1
        i+=1

    print("\nDone processing")
    response = n != 0
    vc.release()
    id_path = SAVE_DIR + f"{label_id}/id_only/"
    face_path = SAVE_DIR + f"{label_id}/face_only/"
    return response, id_path, face_path


def capture_face(res, frame):
    x1, y1, width, height = res['box']
    x1, y1 = abs(x1), abs(y1)
    x2, y2 = x1 + width, y1 + height
    crop_img = frame[y1:y2, x1:x2]
    print("Cropped a Face")
    return crop_img


def bounding_box(res, frame):
    confidence = res['confidence']
    x1, y1, width, height = res['box']
    x1, y1 = abs(x1), abs(y1)
    x2, y2 = x1 + width, y1 + height
    key_points = res['keypoints'].values()
    cv.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 0), thickness=2)
    cv.putText(frame, f'conf: {confidence:.3f}',
               (x1, y1), cv.FONT_ITALIC, 1, (0, 0, 255), 1)
    for point in key_points:
        cv.circle(frame, point, 5, (0, 255, 0), thickness=-1)
    return frame


def detect_img_from_file(filename, des_path, conf_t=0.9999, label="x"):
    count_all = 0
    count_detected = 0
    count_undetected = 0
    try:
        os.makedirs(des_path)
    except OSError:
        print("Folder already exists. continue ...")

    img = cv.imread(filename)

    count_all += 1
    results = face_detector.detect_faces(
        cv.cvtColor(img, cv.COLOR_BGR2RGB))
    # print(filenames[i],results)

    if len(results) > 0:
        cond = 1  # If found
        for n, res in enumerate(results):
            x1, y1, width, height = res['box']
            x1, y1 = abs(x1), abs(y1)
            x2, y2 = x1 + width, y1 + height

            confidence = res['confidence']
            if confidence > conf_t:
                crop_img = img[y1:y2, x1:x2]
                print("Detected", n+1)
                cv.imwrite(des_path+label+".jpg", crop_img)
                print(des_path+label+".jpg")
                count_detected += 1
                cond = 0
                return True
        if cond:  # If found but lower than threshold
            print("NOT Detected (Lower than threshold)")
            count_undetected += 1
            return False
    else:
        print("NOT Detected")
        count_undetected += 1
        return False
