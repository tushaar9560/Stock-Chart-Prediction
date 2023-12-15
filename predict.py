from tensorflow.keras.preprocessing.image import img_to_array ,load_img
from tensorflow.keras.models import load_model
import mimetypes
import argparse
import imutils
import os
import numpy as np
import cv2
from config import Config

def predict(image_path):
  image = load_img(image_path, target_size = (224,224))
  image = img_to_array(image)/255.0
  image = np.expand_dims(image, axis=0)
  if image is not None:
    pred = model.predict(image)[0]
    (startX, startY, endX, endY) = pred

    print("Coordinates: ", pred)


if __name__ == '__main__':
    model = load_model(Config.model_path) 
    image_path = input("Enter the path to the image: ")

    if os.path.isfile(image_path):
      predict(image_path)
    else:
      print("Invalid file path.")