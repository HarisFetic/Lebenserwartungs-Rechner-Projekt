from flask import Flask, request, render_template
import joblib
import numpy as np

app = Flask(__name__)

# Load the model
model = joblib.load('linreg.joblib')

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    # Extract and preprocess data from form
    features = ['Year', 'Sex_0', 'Sex_1'] # Include all the relevant features
    input_data = [float(request.form.get(feature)) for feature in features]

    # Make a prediction
    prediction = model.predict([input_data])[0]

    return render_template('index.html', prediction_text=f'Predicted Remaining Life Expectancy: {prediction} years')

if __name__ == "__main__":
    app.run(debug=True)
