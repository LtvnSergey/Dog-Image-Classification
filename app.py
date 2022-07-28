import os
from model import load_model
from utils import load_image
from flask import Flask, render_template, request
import urllib
import uuid
import numpy as np

app = Flask(__name__)

# Set base directory
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Load model
model = load_model(os.path.join(BASE_DIR, 'models', 'model_v1.pt'))

# Set allowed extensions of images and check file
ALLOWED_EXT = set(['jpg', 'jpeg', 'png', 'jfif', 'JPG', 'JPEG', 'PNG', 'JFIF'])
def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 2)[1] in ALLOWED_EXT

# Set classes
classes = ['Australian terrier',
            'Border terrier',
            'Samoyed',
            'Beagle',
            'Shih-Tzu',
            'English foxhound',
            'Rhodesian ridgeback',
            'Dingo',
            'Golden retriever',
            'Old English sheepdog']

# Predict
def predict(filename, model):

    # Load image
    image = load_image(filename)
    # Predict class and probabilities
    pred, predict_proba = model.predict(image)

    return pred, predict_proba


# Home page
@app.route('/')
def home():
    return render_template('index.html')


@app.route('/success', methods=['GET', 'POST'])
def success():
    error = ''


    target_img = os.path.join(os.getcwd(), 'static/images')

    if request.method == 'POST':
        if (request.form):
            link = request.form.get('link')
            try:
                # Select image file
                resource = urllib.request.urlopen(link)
                unique_filename = str(uuid.uuid4())
                filename = unique_filename+".jpg"

                img_path = os.path.join(target_img, filename)

                output = open(img_path, "wb")
                output.write(resource.read())
                output.close()

                pred_class, pred_prob = predict(img_path, model)

                top_prob_class = sorted(zip(np.around(np.array(pred_prob[0]), 2)*100, classes), reverse=True)[:3]

                predictions = {
                                "class1": top_prob_class[0][1],
                                "class2": top_prob_class[1][1],
                                "class3": top_prob_class[2][1],
                                "prob1": top_prob_class[0][0],
                                "prob2": top_prob_class[1][0],
                                "prob3": top_prob_class[2][0]
                                }

            # ERROR: Inappropriate input
            except Exception as e:
                print(str(e))
                error = 'This image from this site is not accesible or inappropriate input'

            # Check errors
            if (len(error) == 0):
                # If everything id successful go to 'success' page
                return render_template('success.html', img=filename, predictions=predictions)
            else:
                # If error stay on home page
                return render_template('index.html', error=error)

        elif (request.files):
            file = request.files['file']
            if file and allowed_file(file.filename):
                file.save(os.path.join(target_img , file.filename))
                img_path = os.path.join(target_img , file.filename)
                img = file.filename

                pred_class, pred_prob = predict(img_path, model)

                top_prob_class = sorted(zip(np.around(np.array(pred_prob[0]), 2)*100, classes), reverse=True)[:3]

                predictions = {
                                "class1": top_prob_class[0][1],
                                "class2": top_prob_class[1][1],
                                "class3": top_prob_class[2][1],
                                "prob1": top_prob_class[0][0],
                                "prob2": top_prob_class[1][0],
                                "prob3": top_prob_class[2][0]
                                }
            else:
                error = "Please upload images of jpg, jpeg and png extension only"

            if (len(error) == 0):
                return render_template('success.html', img=img , predictions=predictions)
            else:
                return render_template('index.html', error=error)
    else:
        return render_template('index.html')


if __name__ == "__main__":
    # Run app
    app.run(debug=True)