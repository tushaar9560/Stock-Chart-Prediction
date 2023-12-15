1. Dataset Preparation
    Prepare your dataset, including images and bounding box annotations in a CSV file using annotEdit.py.
    Update the dataset paths in the config.py file.

2. Install the required libraries specified in requirements.txt
   using pip install requirements.txt
   
2. Training the Model
    Run the evaluate.py script to evaluate the model. 
    You'll be prompted to ask if you want to train the model 
    Use 'Y' to train the model first time and model will be saved for prediction.
    Evaluation result will be displayed for the trained model.

3. Prediction
    Run the predict.py script to make predictions on new images. 
    You'll be prompted to enter the path to the image you want to predict.