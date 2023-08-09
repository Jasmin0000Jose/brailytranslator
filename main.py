from flask import Flask, render_template, request, redirect, url_for
import tensorflow as tf
import numpy as np

app = Flask(__name__)

# Load the TensorFlow.js model
model = None
def load_braille_model():
    global model
    model = tf.keras.models.load_model('models/BrailleNet.h5')  # Update with the correct path

load_braille_model()

# Preprocess the uploaded image
def preprocess_image(image):
    resized_image = tf.image.resize(image, [224, 224])
    normalized_image = resized_image / 255.0
    batched_image = np.expand_dims(normalized_image, axis=0)
    return batched_image

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        uploaded_file = request.files['upload']
        if uploaded_file:
            image = tf.image.decode_image(uploaded_file.read())
            preprocessed_image = preprocess_image(image)
            prediction = model.predict(preprocessed_image)
            output = np.argmax(prediction, axis=1).item()
            
            # Redirect to translations.html with the prediction result as parameter
            return redirect(url_for('show_translation', prediction=output))
        else:
            return "No image uploaded"
    except Exception as e:
        return "Error: " + str(e)

@app.route('/templates/translation.html')
def show_translation(prediction):
    return render_template('translations.html', prediction=prediction)

if __name__ == '__main__':
    app.run(debug=True)
