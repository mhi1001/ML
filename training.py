import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
import numpy as np
import joblib

def load_data():
    """
    load dataset
    """
    df = pd.read_csv("./sample_data/diabetes.csv")
    return df

def train_model(X_train, y_train):
    """
    train the RandomForestClassifier model
    """
    model = RandomForestClassifier()
    model.fit(X_train, y_train)
    return model

def make_prediction(model, user_input):
    """
    Make a prediction based on user input.
    """
    prediction = model.predict(user_input)
    return prediction

def main():

    # Load data
    df = load_data()

    # Split data
    X = df.drop('Outcome', axis=1)
    y = df['Outcome']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.15, random_state=42)
    X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.1765, random_state=42)

    print("##############TRAINING SET INFO############################")
    print("Training set shape:", X_train.shape, y_train.shape)
    print("Validation set shape:", X_val.shape, y_val.shape)
    print("Testing set shape:", X_test.shape, y_test.shape)

    # Train model
    model = train_model(X_train, y_train)
    
    # Save the model for future use
    joblib.dump(model, 'diabetes_model.pkl')
    
    # Prompt the user for input for each feature
    print("Please enter the following data for diabetes prediction:")
    pregnancies = float(input("Number of Pregnancies: "))
    glucose = float(input("Glucose:"))
    blood_pressure = float(input("BloodPressure: "))
    skin_thickness = float(input("SkinThickness (mm): "))
    insulin = float(input("Insulin: "))
    bmi = float(input("BMI: "))
    diabetes_pedigree_function = float(input("Diabetes pedigree function: "))
    age = float(input("Age (years): "))
    
    # Create a DataFrame with the user input
    user_input_df = pd.DataFrame([[
        pregnancies, glucose, blood_pressure, skin_thickness, insulin, bmi, diabetes_pedigree_function, age
    ]], columns=['Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI', 'DiabetesPedigreeFunction', 'Age'])

    # Make the actual prediction using our RandomForestClassifier model
    prediction = make_prediction(model, user_input_df)

    print(f"Prediction (0 for No Diabetes, 1 for Diabetes): {prediction[0]}")

if __name__ == "__main__":
    main()