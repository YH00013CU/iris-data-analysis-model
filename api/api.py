from flask import Flask, request, jsonify
import pickle
import numpy as np

app = Flask(__name__)

# Load the Iris model
with open("iris_model.pkl", "rb") as f:
    model = pickle.load(f)

@app.route("/", methods=["GET"])
def home():
    return "🌼 Iris Prediction API is Running"

@app.route("/predict", methods=["POST"])
def predict():
    try:
        # Expecting JSON like: {"feature_array": [5.1, 3.5, 1.4, 0.2]}
        data = request.get_json()
        features = np.array(data["feature_array"]).reshape(1, -1)
        prediction = model.predict(features)[0]

        # Add class label
        class_names = ["Setosa", "Versicolor", "Virginica"]
        response = {
            "prediction": int(prediction),
            "flower_name": class_names[prediction]
        }
        return jsonify(response)

    except Exception as e:
        return jsonify({"error": str(e)}), 400

if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=5000)
