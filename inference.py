import logging
import os
import joblib
import pandas as pd

# 📂 Creazione della cartella per i log
log_dir = "logs"
os.makedirs(log_dir, exist_ok=True)

# 🔍 Configurazione del logger per inferenza
logging.basicConfig(
    filename=os.path.join(log_dir, "inference_log.txt"),
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - INPUT: %(message)s"
)

# 🧠 1. Caricare il modello addestrato
model = joblib.load("random_forest_beer_model.pkl")

# 🔢 2. Caricare i LabelEncoders salvati
label_encoders = joblib.load("label_encoders.pkl")

def encode_input_data(input_data):
    """ Converte le feature categoriche in numeri e aggiunge eventuali feature mancanti. """
    
    # 📌 Feature attese dal modello (ordine esatto delle colonne usate in training)
    expected_features = ["Età", "Grado di alcol", "Colore", "Gusto", "Corpo/consistenza", "Carbonatazione", "Origine"]

    # 📌 Aggiungere eventuali feature mancanti con valori predefiniti
    for feature in expected_features:
        if feature not in input_data:
            input_data[feature] = "Sconosciuto"  # Valore predefinito per evitare errori

    # 📌 Convertire le feature categoriche in numeri
    for col in label_encoders:
        if col in input_data:
            input_data[col] = label_encoders[col].transform([input_data[col]])[0]

    return input_data

def predict(input_data):
    """ Effettua una predizione e logga il risultato. """
    input_data = encode_input_data(input_data)  # Convertire le variabili categoriali
    input_df = pd.DataFrame([input_data])  # Convertire in DataFrame per sklearn
    
    # 📌 Assicurarsi che l'ordine delle feature sia corretto
    input_df = input_df[["Età", "Grado di alcol", "Colore", "Gusto", "Corpo/consistenza", "Carbonatazione", "Origine"]]

    prediction = model.predict(input_df)[0]
    logging.info(f"Input: {input_data} --> Prediction: {prediction}")
    return prediction

# 📥 Esempio di input per test (aggiunto tutte le feature richieste)
input_example = {
    'Età': 30,
    'Grado di alcol': 5.5,
    'Colore': 'Chiara',
    'Gusto': 'Bilanciato',
    'Corpo/consistenza': 'Media',  # Aggiunto per evitare errori
    'Carbonatazione': 'Alta',       # Aggiunto per evitare errori
    'Origine': 'Belga'              # Aggiunto per evitare errori
}

# 🔍 Effettua la predizione
pred = predict(input_example)

print(f"✔ Predizione effettuata: {pred}. Log salvato in logs/inference_log.txt")