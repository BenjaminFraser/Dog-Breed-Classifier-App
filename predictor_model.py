from keras import models, layers, optimizers
from keras.applications import xception
from keras.preprocessing.image import img_to_array, ImageDataGenerator
from keras.preprocessing import image
from keras.applications import imagenet_utils
from PIL import Image
import numpy as np
import io
import pickle
import os
from werkzeug.utils import secure_filename

UPLOAD_FOLDER = 'static/uploaded_images'
IMAGE_INFO_JSON = os.path.join(UPLOAD_FOLDER, 'image_info.json')
CURRENT_IMAGE_INFO = os.path.join(UPLOAD_FOLDER, 'current_image_info.json')
ALLOWED_EXTENSIONS = set(['png', 'jpg', 'jpeg'])
PRETRAINED_MODEL_PATH = 'dog_breeds_xception_tl.h5'
MODEL_LABELS_PATH = 'dog_class_labels.pickle'
PROJECT_PATH = os.path.dirname(os.path.abspath(__file__))

#####################################################################################
#### PREDICTOR MODEL CLASS - TO BE IMPORTED FROM SEPERATE FILE ON PRODUCTION APP ####
#####################################################################################
class PredictorModel(object):
    """ A keras model with pretrained weights loaded from file """

    def __init__(self):
        self.model = None
        self.class_labels = None 


    def load_model(self, model_path, class_labels_path):
        """ Load a given Keras pretrained model into our predictor class.
            The given model file must be of the filetype .h5. An accompanying
            dict containing a mapping of the label classes to their associated
            indices must also be given for decoding of predictions.

        Arguments:
            model_path: A string object pointing to the directory location of model
            class_labels_path: A string pointing to location of class labels dict
        """
        if os.path.exists(model_path):
            try:
                self.model =  models.load_model(model_path)
                self.model._make_predict_function()
            except Exception, e:
                raise ImportError("There was a problem loading the model: {0}. " 
                    "Error message: {1}".format(model_path, e.message))
        else: 
            raise NameError("The specified model path {} was not found.".format(model_path))

        if os.path.exists(class_labels_path):
            try:
                with open(class_labels_path, 'rb') as handle:
                    self.class_labels = pickle.load(handle)
            except Exception, e:
                raise ImportError("There was a problem loading the class dict: {0}. " 
                    "Error message: {1}".format(class_labels_path, e.message))
        else:
            raise NameError("The specified labels path {} was not found.".format(class_labels_path))
        return


    def prepare_image(self, img, target_size, http_request=False):
        """ Preprocess and standardise images so they work with our model 

        Arguments:
            image: An image object in PIL format
            target: a tuple containing the desired image size, e.g. (299, 299)
        """

        # if image from http request, preprocess using PIL
        if http_request:
            # open image in PIL from given path
            img = Image.open(io.BytesIO(img))

            # if the image mode is not RGB, convert it
            if img.mode != "RGB":
                img = img.convert("RGB")

            # resize the input image and preprocess it
            img = img.resize(target_size)

        std_img = img_to_array(img)
        std_img = np.expand_dims(std_img, axis=0)
        test_datagen = ImageDataGenerator(rescale=1. / 255)
        std_img = test_datagen.standardize(std_img)

        # return the processed image
        return std_img


    def predictions_to_labels(self, predictions, top_n_labels=3):
        """ takes an array of softmax prediction values and returns the top n labels 

        Argument:
            predictions: An array of softmax output probabilities representing label likelihood 
            top_n_labels: The number of most likely output labels to return (default 3)

        Returns:
            A python dictionary containing the top n labels and associated probabilities as 
            tuple values for each label, e.g. output_labels['label_1'] = ('boxer', 0.934532224)
        """
        # obtain top n prediction indices for array of predictions
        top_indices = np.argpartition(predictions[0], -top_n_labels)[-top_n_labels:]

        # negate prediction array to sort in descending order
        sorted_top = top_indices[np.argsort(-predictions[0][top_indices])]

        # dict comp to create dict of labels and probs
        output_labels = {"label_" + str(i + 1) : (self.decode_prediction(index), float(predictions[0][index])) 
                            for i, index in enumerate(sorted_top)}

        return output_labels

    def decode_prediction(self, index):
        """ Decode predictions from a given index to provide predicted class 

        Argument:
            index: An integer representing the index of the top prediction (argmax)
        """
        class_label = self.class_labels.keys()[self.class_labels.values().index(index)]
        class_label = class_label.replace('_', ' ')
        return class_label

#####################################################################################