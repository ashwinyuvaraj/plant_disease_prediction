import os
from flask import Flask, request, render_template, redirect, url_for
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np
import logging
from werkzeug.utils import secure_filename


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__)

UPLOAD_FOLDER = 'uploads'
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}  

os.makedirs(UPLOAD_FOLDER, exist_ok=True)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  


try:
    model = load_model('plant_disease_model.h5')
    
    logger.info("Model loaded successfully.")
except Exception as e:
    
    logger.error(f"Error loading the model: {e}")
    
class_names = [ 'Black spot','Damping off','Early blight','Fruit borer','Fruit rot','healthy','Late blight','Leaf miner','Septoria Leaf spot']


def allowed_file(filename):
    """Checks if the uploaded file has an allowed extension."""
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def prepare_image(img_path):
    """Loads and preprocesses an image for the model."""
    try:
        img = image.load_img(img_path, target_size=(128, 128))
        img_array = image.img_to_array(img)
        img_array = np.expand_dims(img_array, axis=0)
        img_array = img_array / 255.0 
        return img_array
    except Exception as e:
        logger.error(f"Error preparing image: {e}")
        return None

def predict_disease(img_path):
    """Predicts the plant disease from the given image path."""
    img = prepare_image(img_path)
    if img is None:
        return "Error: Unable to process image."  

    try:
        predictions = model.predict(img)
        predicted_class_index = np.argmax(predictions[0])
        confidence = predictions[0][predicted_class_index] * 100
        predicted_class_name = class_names[predicted_class_index]
        return predicted_class_name, confidence
    except Exception as e:
        logger.error(f"Error during prediction: {e}")
        return "Error: Prediction failed."

@app.route('/', methods=['GET', 'POST'])
def upload_and_predict():
    """Handles file upload, prediction, and result display."""
    if request.method == 'POST':
        if 'file' not in request.files:
            logger.warning("No file part")
            return render_template('index.html', error='No file part')
        file = request.files['file']
        
        if file.filename == '':
            logger.warning("No selected file")
            return render_template('index.html', error='No selected file')
        if file and allowed_file(file.filename):
            try:
                filename = secure_filename(file.filename)
                filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
                file.save(filepath) 
                logger.info(f"File saved successfully at {filepath}")

                prediction_result = predict_disease(filepath)
                if isinstance(prediction_result, str) and prediction_result.startswith("Error:"):
                    os.remove(filepath) 
                    return render_template('index.html', error=prediction_result) 
                else:
                    predicted_class_name, confidence = prediction_result
                   
                    return render_template('result.html',
                                           filename=filename, 
                                           predicted_class=predicted_class_name,
                                           confidence=confidence)
            except Exception as e:
                logger.error(f"Error during file processing: {e}")
                return render_template('index.html', error=f'Error: {e}')
        else:
            logger.warning("Invalid file type")
            return render_template('index.html', error='Invalid file type. Allowed types are png, jpg, jpeg.')
    return render_template('index.html', error=None) 

@app.route('/uploads/<filename>')
def display_image(filename):
    """Route to display the uploaded image."""
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename)

if __name__ == '__main__':
    
    app.run(debug=True, host='0.0.0.0')