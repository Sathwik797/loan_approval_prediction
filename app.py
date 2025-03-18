from flask import Flask, render_template, request
import pickle
import numpy as np
import pandas as pd

app = Flask(__name__)

# Load the trained model and feature names
try:
    with open('model.pkl', 'rb') as file:
        model, feature_names = pickle.load(file)
except FileNotFoundError:
    model = None
    feature_names = []

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Get user input from form
        age = float(request.form['age'])
        income = float(request.form['income'])
        credit_score = float(request.form['credit_score'])
        loan_amount = float(request.form['loan_amount'])
        loan_term = int(request.form['loan_term'])
        employment_status = request.form['employment_status']

        # Encode Employment Status (Ensure it matches training data)
        employment_encoded = {
            'Employed': [1, 0],
            'Unemployed': [0, 1],
            'Self-Employed': [0, 0]  # Default case since we used drop_first=True
        }
        employment_features = employment_encoded.get(employment_status, [0, 0])

        # Construct input data and align with feature names
        user_input = pd.DataFrame([[age, income, credit_score, loan_amount, loan_term] + employment_features], columns=feature_names)
        user_input = user_input.fillna(0)  # Fill missing values if any

        # Make prediction
        prediction = model.predict(user_input)

        result = "Loan Approved" if prediction[0] == 1 else "Loan Rejected"
        return render_template('result.html', result=result)

    except Exception as e:
        return str(e)

if __name__ == '__main__':
    app.run(debug=True)