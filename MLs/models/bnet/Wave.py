from MLs.models.Model import Skel
import tensorflow as tf

import os
import numpy as np
import cv2 as cv

from tensorflow_similarity.losses import CircleLoss

CUR_DIR = os.getcwd()


class Wave(Skel):

    def face_verification(self, images):

        model = load_model.model

        class_names = self.get_classes_name(CUR_DIR+"/MLs/models/bnet/database")
        class_names.append("Unknown")

        images[0] = cv.resize(images[0], (224, 224), interpolation = cv.INTER_AREA)
        images[1] = cv.resize(images[1], (224, 224), interpolation = cv.INTER_AREA)

        label1 = self.find_face(model, class_names, images[0])
        label2 = self.find_face(model, class_names, images[1])

        if (label1 == label2) and ((label1 != class_names[-1]) or (label2 != class_names[-1])):
            return True
        else:
            return False

    def face_registration():
        pass

    def face_recognition(images: list) -> list:
        return super().face_recognition()

    def get_classes_name(self, path):
        for i, y in enumerate(os.walk(path)):
            subdirs, dirs, files = y
            if i == 0:
                return dirs

    def find_face(self, model, classes, face, th=0.0982):
        found = model.single_lookup(face, k=1)
        # Find Nearest with distance threshold
        if found[0].distance < th:
            return classes[found[0].label]
        else:
            return classes[len(classes) - 1]


class load_model():
    model = tf.keras.models.load_model(
        CUR_DIR + "/MLs/models/bnet/face_model",custom_objects={'circle_loss_fixed': CircleLoss()})

    model.load_index(CUR_DIR + '/MLs/models/bnet/index')