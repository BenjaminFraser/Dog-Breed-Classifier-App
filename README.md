# Dog-Breed-Classifier-App

A Python application implemented in Flask, using Keras and Tensorflow, to recognise the breed of any uploaded dog image.

The application makes use of pre-trained keras models by importing the selected model during initialisation. The model used in this example is a Deep Convolutional Neural Network, trained using transfer learning on an existing Xception imagenet model.

The data preprocessing and model training for the dog breed classifier is illustrated in the Jupyter notebook provided in the model_training/ directory.

![example image](app_images/image_1.png?raw=True "Basic page layout of the flask app.")