import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.preprocessing import StandardScaler
import joblib

data = pd.read_csv('./diabetes.csv')

zero_columns = ['Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI']
for col in zero_columns:
    data[col] = data[col].replace(0, np.nan)
    data[col].fillna(data[col].mean(), inplace=True)

x = data.drop(columns='Outcome')
y = data['Outcome']


scaler = StandardScaler()
x = scaler.fit_transform(x)


x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=0)


model = GaussianNB()
model.fit(x_train, y_train)


joblib.dump(model, 'diabetes_model.pkl')

joblib.dump(scaler, 'scaler.pkl')

print("Model and Scaler saved successfully!")