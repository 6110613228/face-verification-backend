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

        class_names = self.get_classes_name(
            CUR_DIR+"/MLs/models/bnet/database")
        class_names.append("Unknown")

        images[0] = cv.resize(images[0], (224, 224),
                              interpolation=cv.INTER_AREA)
        images[1] = cv.resize(images[1], (224, 224),
                              interpolation=cv.INTER_AREA)

        label1 = self.find_face(model, class_names, images[0])
        label2 = self.find_face(model, class_names, images[1])

        if (label1 == label2) and ((label1 != class_names[-1]) or (label2 != class_names[-1])):
            return True
        else:
            return False

    def face_registration(self):

        model = load_model.model

        index_ds = tf.keras.preprocessing.image_dataset_from_directory(
            CUR_DIR+"/MLs/models/bnet/database",
            shuffle=True,
            labels='inferred',
            label_mode='int',
            image_size=(224, 224),
            color_mode='rgb',
            batch_size=1)

        x_index, y_index = self.split_xy(index_ds)

        model.reset_index()
        model.index(x_index, y_index, data=x_index)
        model.save_index(CUR_DIR + '/MLs/models/bnet/face_model')

        return True

    def face_recognition(self, images: list) -> list:

        model = load_model.model

        class_names = self.get_classes_name(
            CUR_DIR+"/MLs/models/bnet/database")
        class_names.append("Unknown")
        labels = []
        for img in images:
            try:
                img = cv.resize(img, (224, 224), interpolation=cv.INTER_AREA)
                label = self.find_face(model, class_names, img)
            except:
                label = "Not found"
            labels.append(label)

        return labels

    def get_classes_name(self, path):
        for i, y in enumerate(os.walk(path)):
            subdirs, dirs, files = y
            if i == 0:
                return dirs
    # 0.0982 0.1383 0.108
    def find_face(self, model, classes, face, th=0.15):
        found = model.single_lookup(face, k=1)
        # Find Nearest with distance threshold
        print("dist:",found[0].distance,end ="")
        if found[0].distance < th:
            print("class:",classes[found[0].label])
            return classes[found[0].label]
        else:
            print("class:",classes[len(classes) - 1])
            return classes[len(classes) - 1]

    def split_xy(self, data_set):
        # loop batch
        images = list()
        labels = list()
        for img_batch, label_batch in data_set:
            for i in range(len(img_batch)):
                images.append(img_batch[i].numpy().astype("uint8"))
                labels.append(label_batch[i].numpy().astype("uint8"))
        images = np.array(images)
        labels = np.array(labels)
        return images.squeeze(), labels.reshape(-1)


class load_model():
    model = tf.keras.models.load_model(
        CUR_DIR + "/MLs/models/bnet/face_model", custom_objects={'circle_loss_fixed': CircleLoss()})

    model.load_index(CUR_DIR + '\\MLs\\models\\bnet\\face_model')

    def reload_model(self):
        self.model = tf.keras.models.load_model(
            CUR_DIR + "/MLs/models/bnet/face_model", custom_objects={'circle_loss_fixed': CircleLoss()})
