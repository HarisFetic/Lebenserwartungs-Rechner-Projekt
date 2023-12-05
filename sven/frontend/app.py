from flask import Flask, request, render_template
import pandas as pd
import joblib
import numpy as np

app = Flask(__name__)

# Load the model
model = joblib.load('linreg.joblib')

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

    # Binary encode sex
    sex_encoding = [1, 0] if sex == 'Male' else [0, 1]

    # Convert country name to binary encoding
    country_encoding = list(country_encodings[country_name].values())

    # Predefined age groups and find the closest age group to the user's input
    age_groups = [0, 10, 15, 25, 45, 65, 80]
    closest_age = min(age_groups, key=lambda x: abs(x - input_age))

    # Filter data for the year 2020, closest age, matching sex, and country
    matched_row = comparison_data[(comparison_data['Year'] == 2020) &
                                  (comparison_data['Entity'] == country_name) &
                                  (comparison_data['Age'] == closest_age)]

    if matched_row.empty:
        return render_template('index.html', prediction_text='No matching data found.',
                               countries=countries_df['Entity'].tolist())

    # Update the matched row with binary encodings for sex, country, and year
    feature_vector = matched_row.iloc[0]
    feature_vector['Sex_0'], feature_vector['Sex_1'] = sex_encoding
    for i, val in enumerate(country_encoding):
        feature_vector[f'Entity_{i}'] = val
    feature_vector['Year'] = 2021

    # Drop columns not used in training and ensure correct order
    required_columns = ['Entity_0', 'Entity_1', 'Entity_2', 'Entity_3', 'Entity_4', 
                        'Entity_5', 'Entity_6', 'Year', 'Sex_0', 'Sex_1', 'Age', 
                        'Gov expenditure on education (%)', 'Internet usage (% of population)', 
                        'Access to electricity (% of population)', 'Access to Sanitation (% of population)', 
                        'Smoking Adults (% of population)', 'GDP ($)', 
                        'Meat consumptionm in kg per year per capita', 'ObesityRate (BMI > 30)', 
                        'Healthcare spending (% of GDP)', 
                        'air pollution, annual exposure (micrograms per cubic meter)', 
                        'Electoral democracy index']
    feature_vector = feature_vector[required_columns]

    # Make a prediction
    prediction = model.predict([np.array(feature_vector)])[0]

    # Break down the prediction into years, days, hours, minutes, and seconds
    years = int(prediction)
    days_fraction = (prediction - years) * 365
    days = int(days_fraction)
    hours_fraction = (days_fraction - days) * 24
    hours = int(hours_fraction)
    minutes_fraction = (hours_fraction - hours) * 60
    minutes = int(minutes_fraction)
    seconds = int((minutes_fraction - minutes) * 60)

    prediction_text = f'Predicted Remaining Life Expectancy: {years} years, {days} days, {hours} hours, {minutes} minutes, {seconds} seconds'

    return render_template('index.html', prediction_text=prediction_text,
                       countries=countries_df['Entity'].tolist())

if __name__ == "__main__":
    app.run(debug=True)
