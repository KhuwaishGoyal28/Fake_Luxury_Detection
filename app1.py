from flask import Flask, request, jsonify
import tensorflow as tf
from tensorflow.keras.preprocessing import image
import numpy as np
import os
import google.generativeai as genai

app = Flask(__name__)

# Load trained model
model = tf.keras.models.load_model("luxury_brand_detector.h5")

# 🔹 Directly Use Gemini API Key (Replace with your actual key)
GEMINI_API_KEY = "AIzaSyBQLRa5R88g3iWhOoE5uKCN1Obi89YGyII"  # ⚠️ Use with caution!
genai.configure(api_key=GEMINI_API_KEY)

LABELS = {0: "Fake", 1: "Real"}

def preprocess_image(img_path):
    img = image.load_img(img_path, target_size=(224, 224))
    img_array = image.img_to_array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)
    return img_array

def check_with_gemini(product_name, brand):
    prompt = f"Is the product '{product_name}' from the brand '{brand}' genuine or counterfeit?"
    response = genai.GenerativeModel("gemini-pro").generate_content(prompt)
    return response.text.strip()

@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files or 'product_name' not in request.form or 'brand' not in request.form:
        return jsonify({"error": "File, product name, and brand are required"})

    file = request.files['file']
    product_name = request.form['product_name']
    brand = request.form['brand']

    img_path = "uploads/" + file.filename
    file.save(img_path)

    img_array = preprocess_image(img_path)
    prediction = model.predict(img_array)
    cnn_result = LABELS[int(prediction[0][0] > 0.5)]

    gemini_result = check_with_gemini(product_name, brand)

    return jsonify({
        "cnn_prediction": cnn_result,
        "gemini_verification": gemini_result
    })

if __name__ == '__main__':
    os.makedirs("uploads", exist_ok=True)
    app.run(host="0.0.0.0", port=5003, debug=True)  # Running on port 5001
