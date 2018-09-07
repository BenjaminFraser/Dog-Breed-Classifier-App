# import the necessary packages
import flask
from flask import Flask, flash, request, json,\
        render_template, redirect, url_for, send_from_directory
from flask_bootstrap import Bootstrap
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

app = Flask(__name__)
bootstrap = Bootstrap(app)

UPLOAD_FOLDER = 'static/uploaded_images'
IMAGE_INFO_JSON = os.path.join(UPLOAD_FOLDER, 'image_info.json')
CURRENT_IMAGE_INFO = os.path.join(UPLOAD_FOLDER, 'current_image_info.json')
ALLOWED_EXTENSIONS = set(['png', 'jpg', 'jpeg'])
PRETRAINED_MODEL_PATH = 'dog_breeds_xception_tl.h5'
MODEL_LABELS_PATH = 'dog_class_labels.pickle'
PROJECT_PATH = os.path.dirname(os.path.abspath(__file__))

app.secret_key = 'some_secret'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

global predictor
predictor = None


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

def init_image_info():
    """Create image directory if not already existing """
    if not os.path.exists(UPLOAD_FOLDER):
        os.makedirs(UPLOAD_FOLDER)


@app.route('/', methods=['GET', 'POST'])
def make_prediction():
    """ Receive user images and generate predictions """

    if request.method == 'POST':
        # ensure file has been provided
        if 'file' not in request.files:
            flash('No file part')
            return redirect(request.url)
        file = request.files['file']

        # ensure a valid file has been chosen by user
        if file.filename == '':
            flash('No selected file')
            return redirect(request.url)
        if file and allowed_file(file.filename):
            # security concerns
            filename = secure_filename(file.filename)

            # save image for prediction and rendering
            file_path = save_image(file, filename)

            img = image.load_img(file_path, target_size=(299, 299))
            img = predictor.prepare_image(img, target_size=(299, 299))

            # obtain prediction from our predictor model
            predictions = predictor.model.predict(img)

            predict_labels = predictor.predictions_to_labels(predictions, top_n_labels=3)

            # save image info
            save_image_info(filename, predict_labels)

            # keep record of current prediction 
            info = {'file_name': filename, 
                    'labels' : predict_labels
                    }

            with open(CURRENT_IMAGE_INFO, 'w') as f:
                json.dump(info, f, indent=4)

            return render_template(
                    'index.html',
                    labels=predict_labels,
                    cur_image_path=file_path,
                    show_image=True)

    # get information of gallery when receive GET request
    images, num_stored_images = get_recent_images()
    return render_template(
            'index.html', images=images,
            num_stored_images=num_stored_images)


@app.route("/predict", methods=["POST"])
def predict():
    """ Make a prediction based on the POST request image data. """
    # initialize the data dictionary that will be returned from the
    # view
    data = {"success": False}

    # ensure an image was properly uploaded to our endpoint
    if flask.request.method == "POST":
        if flask.request.files.get("image"):
            # read the image in PIL formats
            img = flask.request.files["image"].read()
            img = Image.open(io.BytesIO(img))

            # preprocess the image and prepare it for classification
            img = predictor.prepare_image(img, target_size=(299, 299), http_request=True)

            # classify the input image and then initialize the list
            # of predictions to return to the client
            predictions = predictor.model.predict(img)

            dog_label = predictor.decode_prediction(np.argmax(predictions, axis=-1)[0])
            print(dog_label)
            result = {"label" : str(dog_label), "probability" : float(np.max(predictions[0]))}
            data["predictions"] = result

            # indicate that the request was a success
            data["success"] = True

    # return the data dictionary as a JSON response
    return flask.jsonify(data)


@app.route('/uploads/<filename>')
def uploaded_file(filename):
    """generate url for user uploaded file"""
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename)

@app.errorhandler(404)
def page_not_found(e):
    return render_template('404_page.html'), 404


def allowed_file(filename):
    """Check whether a uploaded file is valid and allowed"""
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


def uploaded_image_path(filename):
    """generate file path for user uploaded image"""
    return '/'.join((app.config['UPLOAD_FOLDER'], filename))


def save_image(file, filename):
    """ Save current image to app.config["UPLOAD_FOLDER"] and return the
    corresponding file path. 

    Parameters:
        file: werkzeug.datastructures.FileStorage
        filename: string representing file name, including extension
    
    Returns
        file_path: string representing the path of the saved image
    """

    # create folder for storing images if not exist
    if not os.path.exists(app.config['UPLOAD_FOLDER']):
        os.makedirs(app.config['UPLOAD_FOLDER'])

    #img = file.read()
    #img = Image.open(io.BytesIO(img))
    #img = img.resize((299, 299))


    # save image to local directory
    file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    file.save(file_path)

    return file_path


def save_image_info(filename, class_labels):
    """ Save predicted result of the image in a json file locally.

    Parameters:
        filename: string representing file name, including extension
        class_labels: A Python dict containing top label predictions, with a class label
                        and probability given for each label as a tuple.
    """

    # save prediction info locally
    with open(IMAGE_INFO_JSON, 'r') as f:
        image_info = json.load(f)
        image_info[filename] = class_labels

    with open(IMAGE_INFO_JSON, 'w') as f:
        json.dump(image_info, f, indent=4)


def get_recent_images(num_images=30):
    """Return information of recent uploaded images for galley rendering
    Parameters
    ----------
    num_images: int
        number of images to show at once
    Returns
    -------
    image_stats: list of dicts representing images in last modified order
        path: str
        label: str
        prob: float
    num_stored_images: int
        indepenent of num_images param, the total number of images available
    """
    folder = app.config['UPLOAD_FOLDER']

    init_image_info()

    # get list of last modified images - ignore .json file and files start with .
    files = ['/'.join((folder, file)) \
        for file in os.listdir(folder) if ('json' not in file) \
        and not (file.startswith('.')) ]

    # list of tuples (file_path, timestamp)
    last_modified_files = [(file, os.path.getmtime(file)) for file in files]
    print(last_modified_files)
    last_modified_files = sorted(last_modified_files,
                            key=lambda t: t[1], reverse=True)
    num_stored_images = len(last_modified_files)

    # build a list of image information
    image_stats = []

    print("THE NUMBER OF STORED IMAGES IS: {}".format(num_stored_images))

    if num_stored_images != 0:

        # read in image info
        with open(IMAGE_INFO_JSON, 'r') as f:
            info = json.load(f)

        for i, f in enumerate(last_modified_files):
            # set limit for rendering pictures
            if i > num_images: break

            path, filename = f[0], f[0].replace(folder, '').replace('/', '')
            cur_image_info = info.get(filename, {})

            print("CURRENT IMAGE INFO IS: {}".format(cur_image_info))

            img = {
                'path': path,
                'labels': cur_image_info
            }
            print("CURRENT IMG LABEL DATA IS: {}".format(img['labels']))
            image_stats.append(img)

    return image_stats, num_stored_images

# if this is the main thread of execution first load the model and
# then start the server
if __name__ == "__main__":
    print(("* Loading Keras model and Flask starting server..."
        "please wait until server has fully started"))
    predictor = PredictorModel()
    predictor.load_model(PRETRAINED_MODEL_PATH, MODEL_LABELS_PATH)
    print(predictor.model)
    print(predictor.class_labels)
    print("Predictor model successfully loaded.")
    app.run(debug=True)