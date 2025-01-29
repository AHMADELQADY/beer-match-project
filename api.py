from fastapi import FastAPI
import joblib
import pandas as pd

app = FastAPI()

# Caricare il modello e i LabelEncoders
model = joblib.load("random_forest_beer_model.pkl")
label_encoders = joblib.load("label_encoders.pkl")

def encode_input_data(input_data):
    """ Converte le feature categoriche in numeri e aggiunge eventuali feature mancanti. """
    expected_features = ["Età", "Grado di alcol", "Colore", "Gusto", "Corpo/consistenza", "Carbonatazione", "Origine"]
    for feature in expected_features:
        if feature not in input_data:
            input_data[feature] = "Sconosciuto"

    for col in label_encoders:
        if col in input_data:
            input_data[col] = label_encoders[col].transform([input_data[col]])[0]

    return input_data

@app.post("/predict")
def predict(data: dict):
    input_data = encode_input_data(data)
    input_df = pd.DataFrame([input_data])
    input_df = input_df[["Età", "Grado di alcol", "Colore", "Gusto", "Corpo/consistenza", "Carbonatazione", "Origine"]]
    
    prediction = model.predict(input_df)[0]
    return {"prediction": prediction}