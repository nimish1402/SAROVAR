from flask import Flask, render_template, request
import joblib
import numpy as np

app = Flask(__name__)

# Load the enhanced model
model = joblib.load("model/crop_model.pkl")

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    try:
        # Get user input
        features = [float(request.form[field]) for field in ["N", "P", "K", "temperature", "humidity", "ph", "rainfall"]]
        input_data = np.array([features])

        # Predict crop
        prediction = model.predict(input_data)[0]

        return render_template("index.html", prediction_text=f"Recommended Crop: {prediction}")

    except Exception as e:
        return render_template("index.html", prediction_text=f"Error: {str(e)}")

if __name__ == "__main__":
    app.run(debug=True)
