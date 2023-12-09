# Lebenserwartungs-Rechner-Projekt ⏳💓
This project is a Flask-based web application designed to predict an individual's remaining life expectancy using different machine learning models. It utilizes a feedforward neural network and XGBoost for predictions.

## Features ✨
- **Web Interface**: Built with Flask, the application provides an easy-to-use interface for users to input their data and receive life expectancy predictions.
- **Advanced Machine Learning Models**: Incorporates a feedforward neural network and XGBoost model to predict life expectancy.
- **Data Analysis and Model Training**: Extensive Exploratory Data Analysis (EDA) cleansing, analysis, Featrue Selection & Engneering, model training and XAI, performed in Jupyter Notebooks. Also including Random Forest and Linear Regression models.

## Technical Stack 🔬
- **Flask**: For creating the web application.
- **Pandas & NumPy**: For data manipulation and analysis.
- **PyTorch**: For building and training the neural network model.
- **XGBoost**: For building the gradient boosting model.
- **Joblib**: For model serialization and deserialization.
- **Scikit-learn**: For various machine learning utilities.
- **Seaborn & Matplotlib**: For data visualization.
- **Category Encoders**: For encoding categorical variables.
- **SHAP**: For model interpretability.

## Installation ⌨️
To set up this project, you need to have Python installed on your system. After cloning the repository, install the required dependencies:

```bash
pip install -r requirements.txt
```

## Usage 🔧
To run the Flask application, execute:

```bash
python app.py
```

Navigate to http://localhost:5000 in your web browser to access the application.

## File Structure 📁
- **app.py**: The Flask application.
- **final.ipynb**: Jupyter notebook containing EDA, transformation, cleansing, analysis, Featrue Selection & Engneering, model traing and evaluation aswell as XAI.

## License ⚖️

This project is licensed under the MIT License.



