from flask import Flask, request, render_template
import pickle
import numpy as np
import joblib  # For loading the scaler

app = Flask(__name__)

# Load the model and scaler from the pickle files
model_path = 'model.pkl'
scaler_path = 'scaler.pkl'

with open(model_path, 'rb') as f:
    model = pickle.load(f)

with open(scaler_path, 'rb') as f:
    scaler = joblib.load(f)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Capture the form data
        data = request.form

        # Check if all required fields are present
        required_fields = ['age', 'gender', 'height', 'weight', 'systolic_bp', 'diastolic_bp', 'cholesterol', 'glucose', 'smoker', 'alcohol', 'physical']
        for field in required_fields:
            if field not in data or not data[field]:
                return render_template('index.html', result=f"Missing or empty field: {field}")

        # Convert form data to appropriate types and create the features array
        features = np.array([[
            float(data['age']),
            int(data['gender']),
            float(data['height']),
            float(data['weight']),
            int(data['systolic_bp']),
            int(data['diastolic_bp']),
            int(data['cholesterol']),
            int(data['glucose']),
            int(data['smoker']),
            int(data['alcohol']),
            int(data['physical'])
        ]])

        # Apply the same scaling as used during training
        features = scaler.transform(features)

        # Predict using the model
        prediction = model.predict(features)

        # Interpret the result
        result = 'Positive for Cardiovascular Disease' if prediction[0] == 1 else 'Negative for Cardiovascular Disease'
        return render_template('index.html', result=result)

    except ValueError as ve:
        return render_template('index.html', result=f"Value error: {ve}")
    except Exception as e:
        return render_template('index.html', result=f"An error occurred: {e}")

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8080, debug=True)
