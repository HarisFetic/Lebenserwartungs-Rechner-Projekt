from flask import Flask, request, render_template
import pandas as pd
import numpy as np
import torch
import joblib
import torch.nn as nn


# Define Neural Network class
class NeuralNet(nn.Module):
    def __init__(self, input_size):
        super(NeuralNet, self).__init__()
        self.fc1 = nn.Linear(input_size, 256)
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Linear(256, 128)
        self.relu2 = nn.ReLU()
        self.fc3 = nn.Linear(128, 1)

    def forward(self, x):
        x = self.fc1(x)
        x = self.relu1(x)
        x = self.fc2(x)
        x = self.relu2(x)
        x = self.fc3(x)
        return x
    
app = Flask(__name__)

# Load models
xgb_model_binary = joblib.load('xgb_model_binary.joblib')
fnn_model = NeuralNet(22)
fnn_model.load_state_dict(torch.load('FNN_model.pth'))
fnn_model.eval()

# Load the StandardScaler
scaler = joblib.load('scaler.save')

# Read country names and their encodings from the CSV file
countries_df = pd.read_csv('countries.csv')
country_encodings = countries_df.set_index('Entity').to_dict(orient='index')

# Load the dataset for comparison
comparison_data = pd.read_csv('final_merged_data.csv')

@app.route('/')
def home():
    return render_template('index.html', countries=countries_df['Entity'].tolist())

@app.route('/predict', methods=['POST'])
def predict():
    # User inputs
    input_age = float(request.form.get('Age'))
    sex = request.form.get('Sex')  # 'Male' or 'Female'
    country_name = request.form.get('Country')
    smoking_status = request.form.get('Smoking')
    # height = float(request.form.get('Height'))  # in meters
    # weight = float(request.form.get('Weight'))  # in kilograms
    
    # Binary encode sex
    sex_encoding = [1, 0] if sex == 'Male' else [0, 1]

    # Convert country name to binary encoding
    country_encoding = list(country_encodings[country_name].values())

    # Predefined age groups and find the closest age group to the user's input
    age_groups = [0, 10, 15, 25, 45, 65, 80]
    closest_age = min(age_groups, key=lambda x: abs(x - input_age))

    # Set Smoking Adults (% of population) based on smoking status
    if smoking_status == 'Yes':
        smoking_value = 100
    else:
        smoking_value = 0

    # Calculate if persoon is obese
    # if weight / (height ** 2) > 30:
        # bmi =100
        # else:
        # bmi = 0

    # Debug: Print column names of the loaded DataFrame
    print("Columns in comparison_data:", comparison_data.columns.tolist())

    # Filter data for the year 2020, closest age, matching sex, and country
    matched_row = comparison_data[(comparison_data['Year'] == 2020) &
                                  (comparison_data['Entity'] == country_name) &
                                  (comparison_data['Age'] == closest_age)]

    if matched_row.empty:
        return render_template('index.html', prediction_text='No matching data found.',
                               countries=countries_df['Entity'].tolist())


    # Convert the matched row to a DataFrame
    feature_vector_df = pd.DataFrame([matched_row.iloc[0]])

    # Update feature vector 
    feature_vector_df['Sex_0'], feature_vector_df['Sex_1'] = sex_encoding
    for i, val in enumerate(country_encoding):
        feature_vector_df[f'Entity_{i}'] = val
    feature_vector_df['Smoking Adults (% of population)'] = smoking_value
    # feature_vector_df['ObesityRate (BMI > 30)'] = bmi
    

    # Define required columns
    required_columns = ['Entity_0', 'Entity_1', 'Entity_2', 'Entity_3', 'Entity_4', 
                        'Entity_5', 'Entity_6', 'Year', 'Sex_0', 'Sex_1', 'Age', 
                        'Gov expenditure on education (%)', 'Internet usage (% of population)', 
                        'Access to electricity (% of population)', 'Access to Sanitation (% of population)', 
                        'Smoking Adults (% of population)', 'GDP ($)', 
                        'Meat consumption in kg per year per capita', 'ObesityRate (BMI > 30)', 
                        'Healthcare spending (% of GDP)', 
                        'air pollution, annual exposure (micrograms per cubic meter)', 
                        'Electoral democracy index']
    
    # Get the selected model from the form
    selected_model = request.form.get('Model')

    # Prepare the feature vector so its in the right order
    feature_vector = feature_vector_df[required_columns].to_numpy().flatten()
    
    # Scaled features for the neural network
    scaled_features = scaler.transform([feature_vector])

    # Get the selected model from the form
    selected_model = request.form.get('Model')

    # Make a prediction based on the selected model
    if selected_model == 'FNN':
        input_tensor = torch.tensor(scaled_features, dtype=torch.float32)
        with torch.no_grad():
            prediction_tensor = fnn_model(input_tensor)
            prediction = prediction_tensor.item()
    elif selected_model == 'XGBoost':
        # For XGBoost, use the unscaled feature vector
        prediction = xgb_model_binary.predict([feature_vector])[0]



    # Break down the prediction into years, days, hours, minutes, and seconds
    years = int(prediction)
    days_fraction = (prediction - years) * 365
    days = int(days_fraction)
    hours_fraction = (days_fraction - days) * 24
    hours = int(hours_fraction)
    minutes_fraction = (hours_fraction - hours) * 60
    minutes = int(minutes_fraction)
    seconds = int((minutes_fraction - minutes) * 60)


    # Extract the components of the remaining life expectancy
    components = {
        'years': years,
        'days': days,
        'hours': hours,
        'minutes': minutes,
        'seconds': seconds
    }

    

    return render_template('index.html', components=components,
                           countries=countries_df['Entity'].tolist(), request=request)
   

if __name__ == "__main__":
    app.run(debug=True)
