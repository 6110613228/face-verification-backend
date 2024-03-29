from mtcnn import MTCNN
import cv2 as cv


def detect_face(image,conf_th = 0.85):
    """
    @param image openCV image object
    @return array of 'dictionary' of found face(s) inside image with key 'box' : (x, y, width, height)
    """
    face_detector = load_mtcnn.mtcnn

    downsize = 1

    image_width = image.shape[1]
    image_height = image.shape[0]

    resize_image = cv.resize(image, (int(
        image_width/downsize), int(image_height/downsize)), interpolation=cv.INTER_AREA)

    result = face_detector.detect_faces(resize_image)
    faces = []
    for face in result:

        if face['confidence'] > conf_th :

            del face['confidence']
            del face['keypoints']

            face['box'] = [x * downsize for x in face['box']]
            faces.append(face)
            # face['box'][0] = face['box'][0] - 20
            # face['box'][1] = face['box'][1] - 20
            # face['box'][2] = face['box'][2] + 40
            # face['box'][3] = face['box'][3] + 40
    return faces


def crop_face(image, bounding_box):
    """
    @param image openCV: image object
    @param bounding_box: Array of bounding box of faces inside image

    @return array of cropped image
    """
    cropped_faces = []

    for face in bounding_box:
        x, y, width, height = face['box']
        cropped_faces.append(image[y:y + height, x:x + width])

    return cropped_faces


class load_mtcnn():
    mtcnn = MTCNN()
