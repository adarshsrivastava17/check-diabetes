from django.shortcuts import render
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor

# Load dataset
dataset = pd.read_csv('C:/Users/adars/OneDrive/Desktop/Yukti-project/Diabetic/diabetes.csv')

# Preprocessing
x = dataset.drop(columns='Outcome', axis=1)
y = dataset['Outcome']

scaler = StandardScaler()
scaler.fit(x)
x_scaled = scaler.transform(x)

# Train-Test Split
x_train, x_test, y_train, y_test = train_test_split(x_scaled, y, test_size=0.2, random_state=2)

# Train Model
model = DecisionTreeRegressor(random_state=0)
model.fit(x_train, y_train)

def predict_diabetes(request):
    if request.method == 'POST':
        # Get user input
        values = [
            float(request.POST['pregnancies']),
            float(request.POST['glucose']),
            float(request.POST['bp']),
            float(request.POST['skin_thickness']),
            float(request.POST['insulin']),
            float(request.POST['bmi']),
            float(request.POST['dpf']),
            float(request.POST['age'])
        ]

        # Scale input
        values_scaled = scaler.transform([values])

        # Predict
        result = model.predict(values_scaled)[0]
        prediction = "The person is diabetic" if result == 1 else "The person is not diabetic"

        return render(request, 'index.html', {'result': prediction})

    return render(request, 'index.html')
