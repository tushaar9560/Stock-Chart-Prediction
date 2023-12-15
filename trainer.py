import os
import cv2
import numpy as np
from tensorflow.keras.applications import VGG16
from tensorflow.keras.layers import Flatten, Dense, Input
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.preprocessing.image import img_to_array, load_img
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from config import Config

def load_data():
    rows = open(Config.annots_path).read().strip().split('\n')

    data = []
    targets = []
    filenames = []

    for row in rows:
        row = row.split(',')
        filename = int(row[0].strip('"'))
        coordinates = [float(coord.strip('"')) for coord in row[1:]]
        (startX, startY, endX, endY) = coordinates
        imagePath = os.path.sep.join([Config.images_path, str(filename)+'.jpg'])
        image = cv2.imread(imagePath)
        image = load_img(imagePath, target_size = (224, 224))
        image = img_to_array(image)
        data.append(image)
        targets.append(coordinates)
        filenames.append(filename)

    data = np.array(data,dtype='float32') / 255.0
    targets = np.array(targets, dtype='float32')

    trainImages, testImages, trainTargets, testTargets, trainFilenames, testFilenames = train_test_split(
                data, targets, filenames, test_size=0.1, random_state=42)

    return trainImages, trainTargets, testImages, testTargets

def train_model(train):
    trainImages, trainTargets, testImages, testTargets = load_data()
    if train == 'Y':
        # load the VGG16 network, ensuring the head FC layers are left off
        vgg = VGG16(weights="imagenet", include_top=False,
        	input_tensor=Input(shape=(224, 224, 3)))
        # freeze all VGG layers so they will *not* be updated during the
        # training process
        vgg.trainable = False

        flatten = vgg.output
        flatten = Flatten()(flatten)
        # construct a fully-connected layer header to output the predicted
        # bounding box coordinates
        bboxHead = Dense(128, activation="relu")(flatten)
        bboxHead = Dense(64, activation="relu")(bboxHead)
        bboxHead = Dense(32, activation="relu")(bboxHead)
        bboxHead = Dense(4, activation="sigmoid")(bboxHead)
        # construct the model we will fine-tune for bounding box regression
        model = Model(inputs=vgg.input, outputs=bboxHead)

        opt = Adam(lr=Config.init_lr)
        model.compile(loss="mse", optimizer=opt)
        print(model.summary())
        # train the network for bounding box regression

        H = model.fit(
        	trainImages, trainTargets,
        	validation_data=(testImages, testTargets),
        	batch_size=Config.batch_size,
        	epochs=Config.epochs,
        	verbose=1)

        print("[INFO] saving object detector model...")
        model.save(Config.model_path, save_format="h5")

    return testImages, testTargets

if __name__ == '__main__':
    train_model(train='Y')
