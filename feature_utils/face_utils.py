from mtcnn import MTCNN
import cv2 as cv


def detect_face(image):
    """
    @param image openCV image object
    @return array of 'dictionary' of found face(s) inside image with key 'box' : (x, y, width, height)
    """
    face_detector = load_mtcnn.mtcnn

    downsize = 2

    image_width = image.shape[1]
    image_height = image.shape[0]

    resize_image = cv.resize(image, (int(
        image_width/downsize), int(image_height/downsize)), interpolation=cv.INTER_AREA)


    result = face_detector.detect_faces(resize_image)

    for face in result:
        del face['confidence']
        del face['keypoints']

        face['box'] = [x * 2 for x in face['box']]

    return result

def crop_face(image, bounding_box):
    """
    @param image openCV image object
    @param bounding_box array of bounding box of faces inside image

    @return array of cropped image
    """
    cropped_faces = []

    for face in bounding_box:
        x, y, width, height = face['box']
        cropped_faces.append(image[y:y + height, x:x + width])

    return cropped_faces

class load_mtcnn():
    mtcnn = MTCNN()
