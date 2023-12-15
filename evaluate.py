from tensorflow.keras.models import load_model
from sklearn.metrics import classification_report, confusion_matrix
import numpy as np
from config import Config
from trainer import train_model



def evaluate_model(model, testImages, testTargets):
    # Predict bounding box coordinates on the test dataset
    predictions = model.predict(testImages)

    # Initialize lists to store evaluation metrics
    true_positives = 0
    false_positives = 0
    false_negatives = 0

    for i in range(len(testImages)):
        # Calculate the absolute difference between predicted and true bounding box coordinates
        diff = np.abs(predictions[i] - testTargets[i])

        # Define a threshold for considering a prediction as a true positive
        threshold = 0.25  # Adjust this threshold as needed

        if np.all(diff <= threshold):
            true_positives += 1
        else:
            false_positives += 1
            false_negatives += 1

    # Calculate precision, recall, and F1-score
    precision = true_positives / (true_positives + false_positives)
    recall = true_positives / (true_positives + false_negatives)
    f1 = 2 * (precision * recall) / (precision + recall)

    return precision, recall, f1

if __name__ == '__main__':
    model = load_model(Config.model_path)
    testImages,testTargets = train_model(train = input("Wants to train the model (Y/N): "))
    precision, recall, f1 = evaluate_model(model, testImages, testTargets)
    print("Precision: {:.2f}".format(precision))
    print("Recall: {:.2f}".format(recall))
    print("F1-score: {:.2f}".format(f1))
